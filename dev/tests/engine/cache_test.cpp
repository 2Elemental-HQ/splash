#include "TestKvPool.hpp"
#include "engine/Cache.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

using namespace splash;
using namespace splash::engine;

namespace {

class State final : public CompositeState {
public:
  explicit State(uint64_t bytes) : bytes_(bytes) {}
  uint64_t bytes() const noexcept override { return bytes_; }

private:
  uint64_t bytes_;
};

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

CacheNamespace cacheNamespace() {
  CacheNamespace result;
  result.digest.fill(0x5a);
  return result;
}

void publish(engine::Cache &resources, uint64_t block,
             uint64_t bytes) {
  resources.publishCompositeState(block, std::make_shared<State>(bytes));
}

std::vector<uint32_t> tokens(uint32_t count, uint32_t salt = 0) {
  std::vector<uint32_t> result(count);
  for (uint32_t i = 0; i < count; ++i)
    result[i] = salt + i + 1;
  return result;
}

void testCanonicalPagesAndSparseState() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, cacheNamespace());
  auto prompt = tokens(65);
  resources.beginRequest(1);
  require(resources.ensureTokens(1, prompt.size()).granted(),
          "request pages were not admitted");
  uint64_t deepest = resources.publishCommittedBlocks(1, prompt, 64);
  require(deepest && resources.blockAt(1, 64) == deepest,
          "complete Page32 chain was not published");
  publish(resources, deepest, 100);
  resources.endRequest(1);

  auto lookup = resources.lookup(prompt);
  require(lookup.kvBoundary == 64 && lookup.resumeBoundary() == 64 &&
              !lookup.junctionBoundary(),
          "KV-first lookup did not coordinate the sparse state");
  resources.beginRequest(2);
  require(resources.restoreRequest(2, lookup).granted(), "restore pages were denied");
  require(resources.pageTable(2).pages.size() == 2,
          "restored request did not retain the state KV chain");
  resources.endRequest(2);
}

void testKvDeeperThanStateAndDependencyEviction() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, cacheNamespace());
  auto prompt = tokens(97);
  resources.beginRequest(1);
  require(resources.ensureTokens(1, 96).granted(), "KV allocation failed");
  static_cast<void>(resources.publishCommittedBlocks(1, prompt, 96));
  const uint64_t middle = resources.blockAt(1, 64);
  publish(resources, middle, 100);
  resources.endRequest(1);
  auto lookup = resources.lookup(prompt);
  require(lookup.kvBoundary == 96 && lookup.resumeBoundary() == 64 &&
              lookup.junctionBoundary() == 96,
          "dense KV did not expose the lazy state junction");
  lookup.state.reset();
  require(resources.reclaimCache(1, false, false) >= 100,
          "unreferenced composite state was not reclaimed first");
  require(resources.snapshot().kvCache.blocks == 2,
          "LRU reclaim did not remove the older fragmented KV leaf first");
  require(resources.reclaimCache(1, false, false) != 0,
          "KV was not reclaimed after cached state");
  require(resources.snapshot().stateCache.entries == 0,
          "composite state outlived its KV dependency");
}

void testActiveTipProtectsTheContentChain() {
  test::TestKvStorage storage(8, 4096, 4);
  storage.budgetPages = 4;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, cacheNamespace());
  auto prompt = tokens(97, 1000);
  resources.beginRequest(1);
  require(resources.ensureTokens(1, 64).granted(),
          "active request KV allocation failed");
  static_cast<void>(resources.publishCommittedBlocks(1, prompt, 64));
  resources.beginRequest(2);

  engine::TokenAdmission blocked = resources.ensureTokens(2, 96);
  require(!blocked.granted() &&
              !resources.reclaimOne(CacheReclaimMode::KeepExtents).madeProgress &&
              resources.snapshot().kvCache.blocks == 2,
          "memory pressure evicted an active request KV tip");

  resources.endRequest(1);
  require(resources.reclaimOne(CacheReclaimMode::KeepExtents).madeProgress &&
              resources.ensureTokens(2, 96).granted(),
          "released KV tip did not become reclaimable");
  resources.endRequest(2);
}

void testGrowthReclaimsOneWholeCachedExtent() {
  test::TestKvStorage storage(8, 4096, 4);
  storage.budgetPages = 4;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, cacheNamespace());
  auto prompt = tokens(129, 2000);
  resources.beginRequest(1);
  require(resources.ensureTokens(1, 128).granted(),
          "initial KV extent allocation failed");
  static_cast<void>(resources.publishCommittedBlocks(1, prompt, 128));
  resources.endRequest(1);
  require(resources.snapshot().kvCache.blocks == 4 &&
              resources.snapshot().pool.pagesAllocated == 4,
          "cached extent setup is wrong");

  resources.beginRequest(2);
  require(!resources.ensureTokens(2, 1).granted(),
          "growth bypassed engine-coordinated reclaim");
  require(resources.reclaimCache(1, false, false) != 0 &&
              resources.ensureTokens(2, 1).granted(),
          "explicit reclaim did not release the cached KV extent");
  const auto snapshot = resources.snapshot();
  require(snapshot.kvCache.blocks == 0 && snapshot.pool.pagesAllocated == 4 &&
              snapshot.pool.pagesActive == 1,
          "growth reclaim did not atomically replace the cached extent");
  resources.endRequest(2);
}

void testFragmentedColdKvPrecedesNewerState() {
  test::TestKvStorage storage(8, 4096, 4);
  storage.budgetPages = 4;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, cacheNamespace());
  const auto prompt = tokens(129);
  resources.beginRequest(1);
  require(resources.ensureTokens(1, 128).granted(), "fixture allocation failed");
  static_cast<void>(resources.publishCommittedBlocks(1, prompt, 128));
  const uint64_t stateBlock = resources.blockAt(1, 32);
  resources.endRequest(1);
  publish(resources, stateBlock, 100);

  const auto reclaimed = resources.reclaimOne();
  require(reclaimed.madeProgress && reclaimed.reclaimedBytes == 0 &&
              resources.snapshot().kvCache.blocks == 3 &&
              resources.snapshot().stateCache.entries == 1 &&
              storage.releasedExtents == 0,
          "released-byte preference evicted newer state before cold KV");
}

void testReplacementKeepsTheExtentItEmpties() {
  test::TestKvStorage storage(8, 4096, 4);
  storage.budgetPages = 4;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, cacheNamespace());
  resources.beginRequest(1);
  const auto prompt = tokens(33);
  require(resources.ensureTokens(1, 32).granted(), "fixture allocation failed");
  static_cast<void>(resources.publishCommittedBlocks(1, prompt, 32));
  resources.endRequest(1);

  const auto reclaimed = resources.reclaimOne(CacheReclaimMode::KeepExtents);
  require(reclaimed.madeProgress && reclaimed.reclaimedBytes == 0 &&
              resources.snapshot().kvCache.blocks == 0 &&
              storage.allocatedPages() == 4 && storage.releasedExtents == 0,
          "replacement released the newly reusable extent");
  resources.beginRequest(2);
  require(resources.ensureTokens(2, 128).granted() &&
              pool.snapshot().extentAllocations == 1 && storage.releasedExtents == 0,
          "replacement allocated the reusable extent again");
  resources.endRequest(2);
  require(resources.reclaimCache(0, false, false) == 4 * 4096 &&
              storage.allocatedPages() == 0 && storage.releasedExtents == 1,
          "zero-target shrink did not release the empty extent");
}

// One reclaim pass releases every empty extent first, however many there
// are, then evicts the cache and releases the extents that empties.
void testReclaimPassReleasesEveryEmptyExtent() {
  constexpr uint32_t empty = 200;
  test::TestKvStorage storage(4 * (empty + 1), 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, cacheNamespace());
  const auto prompt = tokens(33);
  resources.beginRequest(1);
  require(resources.ensureTokens(1, 32).granted(), "cached page allocation failed");
  static_cast<void>(resources.publishCommittedBlocks(1, prompt, 32));
  resources.endRequest(1);
  // Request 2 fills the cached block's extent and every other one, then
  // ends: every extent but the cached block's is empty.
  resources.beginRequest(2);
  require(resources.ensureTokens(2, 4 * empty * 32 + 3 * 32).granted(),
          "empty extent allocation failed");
  resources.endRequest(2);
  require(resources.snapshot().pool.reclaimableExtents == empty &&
              resources.snapshot().kvCache.blocks == 1,
          "release setup geometry changed");

  require(resources.reclaimCache(uint64_t{empty} * 4 * 4096, false, false) ==
                  uint64_t{empty} * 4 * 4096 &&
              storage.releasedExtents == empty &&
              resources.snapshot().pool.reclaimableExtents == 0 &&
              resources.snapshot().kvCache.blocks == 1,
          "a pass did not release every empty extent before evicting");
  require(resources.reclaimCache(1ULL << 40, false, false) == 4 * 4096 &&
              storage.releasedExtents == empty + 1 &&
              storage.allocatedPages() == 0 &&
              resources.snapshot().kvCache.blocks == 0,
          "a pass did not evict the cache and release its extent");
}

// A pass that releases extents as its evictions empty them reports the
// longest release of one extent, as growth reports the longest allocation of
// one; the loop's longest tick covers the whole pass.
void testReleaseTimeCoversOneExtent() {
  constexpr uint32_t extents = 6;
  test::TestKvStorage storage(4 * extents, 4096, 4);
  storage.releaseTime = std::chrono::milliseconds(5);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, cacheNamespace());
  for (uint32_t chain = 0; chain < extents; ++chain) {
    const uint64_t id = chain + 1;
    resources.beginRequest(id);
    require(resources.ensureTokens(id, 128).granted(),
            "cached chain allocation failed");
    static_cast<void>(
        resources.publishCommittedBlocks(id, tokens(129, 1000 * chain), 128));
    resources.endRequest(id);
  }
  require(resources.snapshot().pool.reclaimableExtents == 0 &&
              resources.snapshot().kvCache.blocks == 4 * extents,
          "release time setup geometry changed");
  static_cast<void>(resources.evictAll());
  // The pool's timer runs around one release: no less than the storage saw
  // its longest release take, and less than all of them took together. No
  // bound in milliseconds holds on a loaded machine.
  const double longest = resources.snapshot().pool.extentReleaseMaxMilliseconds;
  require(storage.releasedExtents == extents && longest >= storage.longestRelease &&
              longest < storage.totalRelease,
          "the release time is not one extent's");
}

// KV gives a publication in use memory only through the extent it leaves
// empty. An empty extent goes first. Then leaves go, oldest first, until one
// leaves an extent empty, which is released before the call returns: the
// snapshot that follows needs the memory at once.
void testPublicationReleasesTheExtentItEmpties() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, cacheNamespace());
  // The publishing request runs on the first extent.
  const auto running = tokens(129);
  resources.beginRequest(1);
  require(resources.ensureTokens(1, 128).granted(), "the running request got no pages");
  static_cast<void>(resources.publishCommittedBlocks(1, running, 128));
  const uint64_t point = resources.blockAt(1, 128);
  StateUse use = resources.useState(point);
  // Another conversation's chain fills the second extent and half the third.
  const auto other = tokens(193, 1000);
  resources.beginRequest(2);
  require(resources.ensureTokens(2, 192).granted(), "the other chain got no pages");
  static_cast<void>(resources.publishCommittedBlocks(2, other, 192));
  resources.endRequest(2);
  // A request that cached nothing leaves the fourth extent empty.
  resources.beginRequest(3);
  require(resources.ensureTokens(3, 96).granted(), "the empty extent was not allocated");
  resources.endRequest(3);
  require(resources.snapshot().pool.reclaimableExtents == 1 && storage.allocatedPages() == 16,
          "fixture geometry changed");

  // An extent is room only for a snapshot that can allocate its bytes.
  require(!resources.reclaimOneState(false, point, false) && storage.releasedExtents == 0 &&
              resources.snapshot().kvCache.blocks == 10,
          "a publication that cannot allocate released an extent or took KV");
  StateRoom room = resources.reclaimOneState(false, point);
  require(room && room.extent && storage.releasedExtents == 1 &&
              resources.snapshot().kvCache.blocks == 10,
          "the publication did not release the empty extent first");
  room = resources.reclaimOneState(false, point);
  require(room && room.extent && storage.releasedExtents == 2 &&
              storage.allocatedPages() == 8 && resources.snapshot().kvCache.blocks == 8,
          "the publication took more KV than its extent or kept the extent it emptied");
  resources.endRequest(1);
}

} // namespace

int main() {
  try {
    testReclaimPassReleasesEveryEmptyExtent();
    testReleaseTimeCoversOneExtent();
    testCanonicalPagesAndSparseState();
    testKvDeeperThanStateAndDependencyEviction();
    testActiveTipProtectsTheContentChain();
    testGrowthReclaimsOneWholeCachedExtent();
    testFragmentedColdKvPrecedesNewerState();
    testReplacementKeepsTheExtentItEmpties();
    testPublicationReleasesTheExtentItEmpties();
    std::cout << "engine cache tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "engine cache tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
