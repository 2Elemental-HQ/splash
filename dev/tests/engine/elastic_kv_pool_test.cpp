#include "TestKvPool.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using splash::engine::KvPool;
using splash::metal::AllocationFailure;
using splash::test::TestKvStorage;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Error, typename Function>
void requireThrows(Function &&function, const char *message) {
    try {
        function();
    } catch (const Error &) {
        return;
    }
    throw std::runtime_error(message);
}

void release(KvPool &pool, const std::vector<uint32_t> &pages,
             bool prefix = false) {
    for (uint32_t page : pages) pool.releasePage(page, prefix);
}

uint32_t reclaimEvery(KvPool &pool, bool keepRunway) {
    return pool.reclaimEmptyExtents(keepRunway, std::numeric_limits<uint32_t>::max());
}

void testGrowthPacksAllocatedExtents() {
    TestKvStorage storage(12, 100, 4);
    KvPool pool(storage, 4);
    auto pages = pool.acquirePages(5, false);
    require(pages.granted() && pages.pages.size() == 5,
            "elastic pool did not acquire requested pages");
    require(pages.pages[0] == 0 && pages.pages[3] == 3 &&
                pages.pages[4] == 4,
            "elastic pool did not fill its allocated runway first");
    auto live = pool.snapshot();
    require(live.pagesAllocated == 8 && live.pagesActive == 5 &&
                live.pagesFree == 3 &&
                live.allocatedBytes == 800,
            "elastic growth accounting is incorrect");

    release(pool, pages.pages);
    require(reclaimEvery(pool, true) == 1,
            "reclaim did not retain exactly one warm runway");
    auto reclaimed = pool.snapshot();
    require(reclaimed.pagesAllocated == 4 &&
                reclaimed.reclaimableExtents == 1 &&
                storage.releasedExtents == 1,
            "empty extent was not returned exactly");
    require(reclaimed.extentAllocations == 2 && reclaimed.extentReleases == 1,
            "the pool did not count its runway, growth and release");
}

// The runway is allocated through the pool like every later extent, so the
// pool's counts cover all the storage holds. A runway the budget refuses
// fails with its cause, and one longer than the pool is invalid.
void testRunwayIsAllocatedThroughThePool() {
    TestKvStorage storage(16, 100, 4);
    KvPool pool(storage, 6);
    auto status = pool.snapshot();
    require(status.extentAllocations == 2 && status.pagesAllocated == 8 &&
                status.reclaimableExtents == 2 && storage.allocated(0) &&
                storage.allocated(1) && !storage.allocated(2),
            "the runway was not the extents of its pages, allocated through the pool");
    require(reclaimEvery(pool, true) == 1, "an empty runway extent was not released");
    status = pool.snapshot();
    require(status.extentAllocations - status.extentReleases == 1 &&
                status.pagesAllocated == 4 && storage.allocatedPages() == 4,
            "the pool's extent counts differ from the extents allocated");

    TestKvStorage blocked(16, 100, 4);
    blocked.growthBlocked = true;
    bool refused = false;
    try {
        KvPool refusedPool(blocked, 4);
    } catch (const splash::metal::MetalAllocationError &error) {
        refused = error.failure() == AllocationFailure::EngineBudget;
    }
    require(refused, "a refused runway did not fail with the budget's cause");
    requireThrows<std::invalid_argument>([&] { KvPool longPool(blocked, 17); },
                                         "a runway longer than the pool was accepted");
}

// Page ids cover every extent the budget could hold, so running out of them
// is a broken invariant rather than a refusal; the pages taken go back.
void testIdExhaustionIsALogicError() {
    TestKvStorage storage(8, 100, 4);
    KvPool pool(storage, 8);
    requireThrows<std::logic_error>(
        [&] { static_cast<void>(pool.acquirePages(9, false)); },
        "running out of page ids was not a logic error");
    const auto status = pool.snapshot();
    require(status.pagesActive == 0 && status.pagesFree == 8,
            "running out of page ids kept the pages taken");
}

// An extent the storage refuses denies the acquisition, which holds no page
// and keeps the extents it allocated, reclaimable: the retry takes their
// pages instead of allocating them again, and a reclaim pass returns them if
// nothing does.
void testFailedGrowthKeepsItsExtentsForTheRetry() {
    TestKvStorage storage(16, 100, 4);
    storage.growthAllowed = [](uint32_t extent) { return extent != 2; };
    KvPool pool(storage, 0);
    const auto before = pool.snapshot().extentReleases;
    auto pages = pool.acquirePages(9, false);
    auto status = pool.snapshot();
    require(!pages.granted() && pages.failure == AllocationFailure::EngineBudget &&
                storage.allocationAttempts == 3 &&
                storage.releasedExtents == 0 &&
                pool.snapshot().extentReleases == before &&
                status.pagesAllocated == 8 && status.reclaimableExtents == 2 &&
                status.pagesFree == 8 && status.pagesActive == 0 &&
                status.pagesPrefix == 0 && status.extentAllocations == 2,
            "a failed acquisition was not denied, held pages or did not "
            "keep its extents");
    storage.growthAllowed = nullptr;
    pages = pool.acquirePages(9, false);
    require(pages.granted() && pool.snapshot().extentAllocations == 3 &&
                storage.releasedExtents == 0,
            "the retry allocated again the extents it was denied with");
    release(pool, pages.pages);
    require(reclaimEvery(pool, false) == 3 &&
                pool.snapshot().pagesAllocated == 0,
            "a reclaim pass did not return the extents the retry left");
}

// A storage that throws while allocating leaves every page free and the
// accounting whole; the extent allocated before it stays, reclaimable, and
// the pool keeps serving.
void testThrowingStorageKeepsAccounting() {
    TestKvStorage storage(12, 100, 4);
    storage.growthAllowed = [](uint32_t extent) {
        if (extent == 1) throw std::runtime_error("test allocation failure");
        return true;
    };
    KvPool pool(storage, 0);
    bool threw = false;
    try {
        static_cast<void>(pool.acquirePages(5, false));
    } catch (const std::runtime_error &) {
        threw = true;
    }
    auto status = pool.snapshot();
    require(threw && status.pagesActive == 0 && status.pagesFree == 4 &&
                status.pagesAllocated == 4 && status.reclaimableExtents == 1,
            "throwing storage leaked pages or broke the extent accounting");
    storage.growthAllowed = nullptr;
    auto pages = pool.acquirePages(5, false);
    require(pages.granted() && pool.snapshot().pagesActive == 5,
            "pool did not serve after a throwing storage");
}

void testPressureReusesFreePagesAndDeniesGrowth() {
    TestKvStorage storage(8, 100, 4);
    KvPool pool(storage, 4);
    auto active = pool.acquirePages(2, false);
    require(active.granted() && active.pages.size() == 2,
            "pressure setup did not acquire active pages");
    storage.growthBlocked = true;
    auto reused = pool.acquirePages(1, false);
    require(reused.granted() && reused.pages.size() == 1 &&
                reused.pages.front() == 2,
            "critical pressure rejected a free page of an allocated extent");
    auto denied = pool.acquirePages(2, false);
    require(!denied.granted() && denied.failure == AllocationFailure::EngineBudget,
            "critical pressure admitted a new extent");
    require(pool.activeReferences(active.pages[0]) == 1 &&
                pool.activeReferences(active.pages[1]) == 1 &&
                pool.activeReferences(reused.pages.front()) == 1 &&
                pool.snapshot().pagesActive == 3,
            "critical pressure corrupted existing active references");
    release(pool, reused.pages);
    release(pool, active.pages);
    require(reclaimEvery(pool, false) == 1 &&
                pool.snapshot().pagesAllocated == 0,
            "pressure cleanup did not reclaim the empty extent");
}

// One pass releases every empty extent but the runway, however many there
// are; a pass without the runway releases that one too.
void testPassReleasesEveryEmptyExtent() {
    constexpr uint32_t extents = 200;
    TestKvStorage storage(4 * extents, 100, 4);
    KvPool pool(storage, 4 * extents);
    auto pages = pool.acquirePages(4 * extents, false);
    require(pages.granted() && pool.snapshot().pagesAllocated == 4 * extents,
            "release setup did not acquire every page");
    release(pool, pages.pages);
    require(pool.snapshot().reclaimableExtents == extents,
            "every empty extent was not reclaimable");
    const auto before = pool.snapshot().extentReleases;
    require(reclaimEvery(pool, true) == extents - 1 &&
                storage.releasedExtents == extents - 1 &&
                pool.snapshot().extentReleases == before + extents - 1 &&
                pool.snapshot().reclaimableExtents == 1,
            "a pass did not release every empty extent but the runway");
    require(reclaimEvery(pool, false) == 1 &&
                pool.snapshot().pagesAllocated == 0 &&
                pool.snapshot().extentReleases == extents,
            "a pass without the runway did not release it");
}

void testFullestExtentFillsFirstSoColdExtentsDrain() {
    TestKvStorage storage(12, 100, 4);
    KvPool pool(storage, 12);
    auto all = pool.acquirePages(12, false);
    require(all.granted() && all.pages.size() == 12,
            "fill setup did not acquire every page");
    // Leave extent 0 with three holes, extent 1 with one and extent 2 with two.
    release(pool, {0, 1, 2, 5, 8, 9});
    require(pool.snapshot().pagesFree == 6 &&
                pool.snapshot().reclaimableExtents == 0,
            "partial release accounting is incorrect");

    // New pages come from the fullest extents; the coldest keeps its holes.
    auto refill = pool.acquirePages(2, false);
    require(refill.granted() && refill.pages.size() == 2 &&
                refill.pages[0] == 5 &&
                (refill.pages[1] == 8 || refill.pages[1] == 9),
            "refill did not take pages from the fullest extents first");
    auto again = pool.acquirePages(1, false);
    require(again.granted() && again.pages.front() / 4 == 2,
            "allocation did not continue with the fullest extent");

    // Its last page going cold empties the extent so it can be released.
    release(pool, {3});
    require(pool.snapshot().reclaimableExtents == 1 &&
                reclaimEvery(pool, false) == 1 &&
                pool.snapshot().pagesAllocated == 8 &&
                storage.releasedExtents == 1,
            "drained extent was not released");
}

// A release the storage refuses, as PageStorage refuses one while a command
// is in flight, leaves the pool's record of the extent as it was.
void testRefusedReleaseChangesNothing() {
    TestKvStorage storage(8, 100, 4);
    KvPool pool(storage, 8);
    storage.commandInFlight = [] { return true; };
    requireThrows<std::logic_error>([&] { static_cast<void>(reclaimEvery(pool, false)); },
                                    "an extent was released while a command was in flight");
    const auto status = pool.snapshot();
    require(status.pagesAllocated == 8 && status.pagesFree == 8 &&
                status.reclaimableExtents == 2 && status.extentReleases == 0 &&
                storage.allocatedPages() == 8,
            "a refused release changed the pool's record of its extent");
    storage.commandInFlight = nullptr;
    require(reclaimEvery(pool, false) == 2 && pool.snapshot().pagesAllocated == 0,
            "the extents a refused release kept were not released afterwards");
}

void testPrefixAndActiveReferencesHoldTheExtent() {
    TestKvStorage storage(8, 100, 4);
    KvPool pool(storage, 4);
    auto active = pool.acquirePages(1, false);
    require(active.granted(), "shared reference setup failed");
    pool.retainPage(active.pages.front(), true);
    pool.releasePage(active.pages.front(), false);
    require(reclaimEvery(pool, false) == 0 &&
                pool.snapshot().pagesPrefix == 1,
            "prefix-owned extent was reclaimed while live");
    pool.releasePage(active.pages.front(), true);
    require(reclaimEvery(pool, false) == 1,
            "last prefix release did not make extent reclaimable");
}

}  // namespace

int main() {
    try {
        testGrowthPacksAllocatedExtents();
        testRunwayIsAllocatedThroughThePool();
        testIdExhaustionIsALogicError();
        testFailedGrowthKeepsItsExtentsForTheRetry();
        testThrowingStorageKeepsAccounting();
        testPressureReusesFreePagesAndDeniesGrowth();
        testPassReleasesEveryEmptyExtent();
        testFullestExtentFillsFirstSoColdExtentsDrain();
        testRefusedReleaseChangesNothing();
        testPrefixAndActiveReferencesHoldTheExtent();
        std::cout << "elastic KV pool tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "elastic KV pool test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
