#pragma once

#include "ops/PagedKv.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace splash::engine {

struct KvPoolSnapshot {
  // Pages of allocated extents: all of them, those nothing holds, and those
  // requests and the cache hold.
  uint32_t pagesAllocated = 0;
  uint32_t pagesFree = 0;
  uint32_t pagesActive = 0;
  uint32_t pagesPrefix = 0;
  // Extents none of whose pages is held, which a reclaim releases at once.
  uint32_t reclaimableExtents = 0;
  uint64_t allocatedBytes = 0;
  uint64_t reclaimableBytes = 0;
  // Extents allocated and released through the pool, and the longest
  // allocation and release of one: what memory costs per extent. The counts
  // and the longest allocation include the runway the pool allocates when it
  // is built, before the serving loop runs. How long a whole reclaim pass
  // holds the loop shows in its longest tick.
  uint64_t extentAllocations = 0;
  uint64_t extentReleases = 0;
  double extentAllocateMaxMilliseconds = 0.0;
  double extentReleaseMaxMilliseconds = 0.0;
};

struct KvPageAcquisition {
  std::vector<uint32_t> pages;
  // Why the pool could not grow by an extent; None when granted.
  metal::AllocationFailure failure = metal::AllocationFailure::None;

  [[nodiscard]] bool granted() const noexcept {
    return failure == metal::AllocationFailure::None;
  }
};

// Sole owner of KV page references and of which extents are allocated. It
// alone allocates and releases extents, starting with the runway its
// constructor allocates. Resource policy may ask for pages or release
// references, but cannot directly allocate or release Metal memory. Free
// pages are handed out from the allocated extent with the most live pages
// first, so partially used extents fill up, empty extents are touched last,
// and cold extents drain to empty, the only state in which an extent can be
// released.
class KvPool final {
public:
  // Allocates the runway, the extents that hold pages [0, runwayPages), or
  // throws metal::MetalAllocationError with the budget's cause;
  // std::invalid_argument for a runway longer than the pool.
  KvPool(kv::ExtentStorage &storage, uint32_t runwayPages);

  [[nodiscard]] KvPageAcquisition acquirePages(uint32_t count,
                                               bool prefixOwner);
  void retainPage(uint32_t page, bool prefixOwner);
  void releasePage(uint32_t page, bool prefixOwner);

  [[nodiscard]] uint32_t pageCount() const noexcept;
  [[nodiscard]] uint64_t bytesPerPage() const noexcept;
  // Free pages of allocated extents; acquisition hands these out first.
  [[nodiscard]] uint32_t freePageCount() const noexcept;
  [[nodiscard]] uint32_t activeReferences(uint32_t page) const;
  [[nodiscard]] bool pageFree(uint32_t page) const;
  [[nodiscard]] uint64_t allocatedBytes() const noexcept;

  // Releases completely unreferenced extents, at most `limit` of them.
  // keepRunway keeps one empty extent warm, so the next request does not wait
  // for an allocation; unlike the runway the constructor allocates, it is
  // always a single extent.
  [[nodiscard]] uint32_t reclaimEmptyExtents(bool keepRunway, uint32_t limit);
  [[nodiscard]] KvPoolSnapshot snapshot() const;

private:
  static constexpr uint32_t noIndex = std::numeric_limits<uint32_t>::max();

  struct PageRecord {
    uint32_t activeReferences = 0;
    uint32_t prefixReferences = 0;
    uint32_t previousFree = noIndex;
    uint32_t nextFree = noIndex;
    bool onFreeList = false;
  };

  struct IndexList {
    uint32_t head = noIndex;
    uint32_t count = 0;
  };

  struct ExtentRecord {
    uint32_t usedPages = 0;
    uint32_t previousReclaimable = noIndex;
    uint32_t nextReclaimable = noIndex;
    // The extent's pages nothing holds; only an allocated extent lists any.
    IndexList freePages;
    bool allocated = false;
    bool reclaimable = false;
  };

  [[nodiscard]] uint32_t extentOf(uint32_t page) const noexcept {
    return page / extentPages_;
  }
  [[nodiscard]] uint32_t firstPage(uint32_t extent) const noexcept {
    return extent * extentPages_;
  }
  // The lowest extent that is not allocated; noIndex when all of them are.
  [[nodiscard]] uint32_t unallocatedExtent() const noexcept;
  void insertFree(uint32_t page) noexcept;
  void removeFree(uint32_t page) noexcept;
  [[nodiscard]] uint32_t popFree() noexcept;
  [[nodiscard]] uint32_t packingExtent() noexcept;
  void markUsed(uint32_t page) noexcept;
  void markFree(uint32_t page) noexcept;
  void setExtentAllocated(uint32_t extent, bool allocated) noexcept;
  void setExtentReclaimable(uint32_t extent, bool reclaimable) noexcept;

  // Every allocation and release of an extent, timed and counted; the
  // extent's record follows the storage.
  [[nodiscard]] metal::AllocationResult allocateExtent(uint32_t extent);
  void releaseExtent(uint32_t extent);

  kv::ExtentStorage &storage_;
  uint32_t extentPages_ = 0;
  uint64_t extentAllocations_ = 0;
  uint64_t extentReleases_ = 0;
  double extentAllocateMaxMilliseconds_ = 0.0;
  double extentReleaseMaxMilliseconds_ = 0.0;
  std::vector<PageRecord> pages_;
  std::vector<ExtentRecord> extents_;
  uint32_t freePages_ = 0;
  // The extent currently being filled. Stays valid while only this extent
  // changes, so a burst of allocations rescans the extents once per extent
  // it moves into.
  uint32_t packingExtent_ = noIndex;
  IndexList reclaimableExtents_;
  uint32_t activePages_ = 0;
  uint32_t prefixPages_ = 0;
  uint32_t allocatedExtents_ = 0;
};

} // namespace splash::engine
