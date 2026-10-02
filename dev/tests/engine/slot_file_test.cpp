#include "model/SlotFile.hpp"

#include <algorithm>
#include <future>
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using splash::model::DiskBudget;
using splash::model::SlotFile;

// A slot is its payload rounded up to the alignment of uncached IO.
static_assert(SlotFile::slotBytesFor(1) == SlotFile::kAlignmentBytes &&
              SlotFile::slotBytesFor(SlotFile::kAlignmentBytes) == SlotFile::kAlignmentBytes &&
              SlotFile::slotBytesFor(SlotFile::kAlignmentBytes + 1) ==
                  2 * SlotFile::kAlignmentBytes);

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Call>
static bool throws(Call call) {
  try { call(); } catch (const Exception &) { return true; }
  return false;
}

static void testFailedWriteStopsWriting() {
  const pid_t child = fork();
  require(child >= 0, "failed to isolate file limit test");
  if (!child) {
    int result = 0;
    try {
      constexpr size_t size = SlotFile::kAlignmentBytes;
      // The engine starts with the default disposition, which kills the
      // process on a write past the limit, so the file has to change it.
      signal(SIGXFSZ, SIG_DFL);
      SlotFile file(size, 2 * size);
      auto complete = file.acquire();
      auto partial = file.acquire();
      std::vector<std::byte> source(size, std::byte{1}), output(size);
      require(file.write(complete, {source}, {})->wait(), "initial write failed");
      struct rlimit original;
      require(getrlimit(RLIMIT_FSIZE, &original) == 0, "file limit unavailable");
      auto limited = original;
      // The second slot starts at size: allow half its payload before failure.
      limited.rlim_cur = size + size / 2;
      require(setrlimit(RLIMIT_FSIZE, &limited) == 0, "file limit could not be set");
      std::fill(source.begin(), source.end(), std::byte{2});
      require(!file.write(partial, {source}, {})->wait(), "partial write reported success");
      require(!file.read(partial, {output}, {})->wait(), "partially written slot was readable");
      require(file.writtenBytes() == size + size / 2 && file.readBytes() == 0,
              "IO counters lost a partial write or counted an invalid read");
      require(setrlimit(RLIMIT_FSIZE, &original) == 0, "file limit restore failed");
      require(!file.writable() && file.write(partial, {source}, {}) == nullptr,
              "storage failure did not stop further writes");
      // A closed file still rejects a write that does not fit its slots.
      std::vector<std::byte> oversized(size + 1);
      require(throws<std::invalid_argument>(
                  [&] { static_cast<void>(file.write(partial, {oversized}, {})); }),
              "a closed file absorbed a write of more than a slot");
      require(file.read(complete, {output}, {})->wait() && output.front() == std::byte{1},
              "complete slot became unreadable after a storage failure");
      require(file.readBytes() == size, "successful read bytes were not counted");
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      result = 1;
    }
    _exit(result);
  }
  int status = 0;
  require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "file failure/integrity test failed");
}

// Callers' memory need not be aligned or fill a slot: spans of any size and
// alignment, from a few bytes to more than one of the worker's chunks, move
// through the worker's own buffer. A write zeros the slot past its spans,
// and a read may take less than was written, scattered differently.
static void testScatteredSpans() {
  // Three of the worker's 1 MiB chunks, the last one partial.
  constexpr size_t size = (2 << 20) + 3 * SlotFile::kAlignmentBytes;
  SlotFile file(size, size);
  auto slot = file.acquire();
  std::vector<std::byte> source(size);
  for (size_t index = 0; index < source.size(); ++index)
    source[index] = static_cast<std::byte>(index * 131 + 7);
  // Odd addresses and lengths, an empty span, and one span across the first
  // chunk's end, adding up to less than a slot.
  const std::span<const std::byte> bytes(source);
  const std::vector<std::span<const std::byte>> spans{
      bytes.subspan(1, 7), bytes.subspan(4099, 0), bytes.subspan(4099, (1 << 20) + 12345),
      bytes.subspan(33, 999'983)};
  std::vector<std::byte> expected;
  for (auto span : spans) expected.insert(expected.end(), span.begin(), span.end());
  require(expected.size() < size, "the scattered spans fill the slot");
  expected.resize(size);
  require(file.write(slot, spans, {})->wait() && file.writtenBytes() == size,
          "a scattered write failed or did not store the whole slot");

  // The whole slot through one unaligned span: the spans in order, zeros after.
  std::vector<std::byte> whole(size + 1);
  require(file.read(slot, {std::span(whole).subspan(1)}, {})->wait() &&
              std::equal(expected.begin(), expected.end(), whole.begin() + 1),
          "a scattered write did not store its spans in order with zeros after");
  // Less than was written, across the first chunk's end into odd addresses:
  // the read takes the two chunks its spans reach and nothing around them.
  std::vector<std::byte> head(3 * 4096 + 3), tail((1 << 20) + 5);
  const uint64_t readBefore = file.readBytes();
  require(file.read(slot, {std::span(head).subspan(3), std::span(tail).subspan(5)}, {})->wait() &&
              file.readBytes() - readBefore == 2 << 20,
          "a partial read failed or did not take exactly the chunks it needs");
  require(std::all_of(head.begin(), head.begin() + 3, [](std::byte b) { return b == std::byte{0}; }) &&
              std::all_of(tail.begin(), tail.begin() + 5, [](std::byte b) { return b == std::byte{0}; }) &&
              std::equal(head.begin() + 3, head.end(), expected.begin()) &&
              std::equal(tail.begin() + 5, tail.end(), expected.begin() + (head.size() - 3)),
          "a partial read returned the wrong bytes or wrote outside its spans");
}

// Spans of the given sizes, carved from `storage`: each starts at an address
// aligned like slot offsets, as a Metal buffer does, or where `aligned` is
// false one byte past one.
using Shape = std::vector<std::pair<size_t, bool>>;
static std::vector<std::span<std::byte>> placeSpans(std::vector<std::byte> &storage,
                                                    const Shape &shape) {
  constexpr size_t unit = SlotFile::kAlignmentBytes;
  const auto room = [](size_t bytes) { return (bytes + 1 + unit - 1) / unit * unit; };
  size_t total = unit;
  for (const auto &[bytes, aligned] : shape) total += room(bytes);
  storage.assign(total, std::byte{0});
  void *base = storage.data();
  size_t space = storage.size();
  auto *cursor = static_cast<std::byte *>(std::align(unit, total - unit, base, space));
  std::vector<std::span<std::byte>> spans;
  for (const auto &[bytes, aligned] : shape) {
    spans.emplace_back(cursor + (aligned ? 0 : 1), bytes);
    cursor += room(bytes);
  }
  return spans;
}

// A run of at least one chunk of aligned memory moves straight between the
// memory and the file; the rest of the spans, and a run the slot would hold
// at an unaligned offset, move through the worker's buffer. Either way a
// write stores the spans in order with zeros after them, counts the slot
// once, and a read brings them back into spans of any shape.
static void testAlignedRunsMoveDirectly() {
  constexpr size_t chunk = 1 << 20;
  constexpr size_t size = 3 * chunk;
  SlotFile file(size, size);
  auto slot = file.acquire();
  const std::vector<Shape> shapes{
      // Two chunks straight, and the run's last half chunk through the buffer.
      {{5 * chunk / 2, true}},
      // The last run starts at an unaligned offset of the slot.
      {{3 * chunk / 2, true}, {100, false}, {chunk, true}},
      // The buffer takes only the alignment units before a run, so the run
      // starts at an aligned offset of the slot.
      {{3 * SlotFile::kAlignmentBytes, false}, {5 * chunk / 4, true}},
  };
  uint32_t state = 1;
  for (const Shape &shape : shapes) {
    std::vector<std::byte> sourceMemory;
    const auto sources = placeSpans(sourceMemory, shape);
    std::vector<std::byte> expected;
    for (const auto span : sources) {
      for (std::byte &value : span) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<std::byte>(state >> 24);
      }
      expected.insert(expected.end(), span.begin(), span.end());
    }
    expected.resize(size);
    const uint64_t writtenBefore = file.writtenBytes();
    require(file.write(slot, {sources.begin(), sources.end()}, {})->wait() &&
                file.writtenBytes() - writtenBefore == size,
            "a write of aligned runs failed or did not store the slot exactly once");

    // The whole slot into one unaligned span, all of it through the buffer.
    std::vector<std::byte> wholeMemory;
    const auto whole = placeSpans(wholeMemory, {{size, false}}).front();
    require(file.read(slot, {whole}, {})->wait() &&
                std::equal(expected.begin(), expected.end(), whole.begin()),
            "aligned runs did not land in order with zeros after them");
    // Back into spans of the same shape, the aligned runs straight.
    std::vector<std::byte> destinationMemory;
    const auto destinations = placeSpans(destinationMemory, shape);
    require(file.read(slot, destinations, {})->wait(), "a read into aligned runs failed");
    for (size_t index = 0; index < sources.size(); ++index) {
      require(std::equal(sources[index].begin(), sources[index].end(),
                         destinations[index].begin()),
              "a read into aligned runs returned the wrong bytes");
    }
  }
}

// A queued write of aligned runs, cancelled behind a parked worker, fails and
// leaves its slot unreadable. The worker sees the cancellation before the
// write's first chunk, so this covers a queued direct write only, not one
// cancelled between two of its chunks.
static void testCancelledDirectWrite() {
  constexpr size_t size = 3 << 20;
  SlotFile file(size, size);
  auto slot = file.acquire();
  std::vector<std::byte> memory;
  const auto source = placeSpans(memory, {{size, true}}).front();
  require(file.write(slot, {source}, {})->wait(), "the first write of aligned runs failed");
  std::promise<void> reached, release;
  auto released = release.get_future().share();
  std::vector<std::byte> output(size);
  auto hold = file.read(slot, {output}, [&] { reached.set_value(); released.wait(); });
  reached.get_future().wait();
  auto cancelled = file.write(slot, {source}, {});
  cancelled->cancel();
  release.set_value();
  require(hold->wait() && !cancelled->wait(), "a cancelled write of aligned runs succeeded");
  require(file.writable() && !file.read(slot, {output}, {})->wait(),
          "a cancelled write of aligned runs left its slot readable or closed the file");
}

int main() {
  try {
    testFailedWriteStopsWriting();
    testScatteredSpans();
    testAlignedRunsMoveDirectly();
    testCancelledDirectWrite();
    constexpr size_t size = 4 * SlotFile::kAlignmentBytes;
    SlotFile file(size, size * 2 + 1);
    require(file.slotBytes() == size && file.capacityBytes() == size * 2 + 1 &&
                file.usedBytes() == 0,
            "file did not report its quota");
    auto first = file.acquire();
    auto second = file.acquire();
    require(first && second && file.usedBytes() == size * 2 && !file.acquire(),
            "a partial slot of the quota was granted");
    std::vector<std::byte> source(size, std::byte{0xa5}), restored(size);
    auto write = file.write(first, {std::span(source).first(128), std::span(source).subspan(128)}, {});
    require(write->wait(), "slot write failed");
    require(file.read(first, {restored}, {})->wait() && restored == source,
            "slot did not roundtrip across IO spans");
    require(!file.read(second, {restored}, {})->wait(), "unwritten slot was readable");
    require(file.idle(), "a file with every operation finished called itself busy");

    // Park the worker in a completion so the queue behind it is deterministic:
    // a queued write, a cancelled read and a cancelled overwrite behind it.
    std::promise<void> reached, release;
    auto released = release.get_future().share();
    auto hold = file.read(first, {restored}, [&] { reached.set_value(); released.wait(); });
    reached.get_future().wait();
    std::vector<std::byte> later(size, std::byte{0x5a});
    write = file.write(second, {later}, {});
    std::vector<std::byte> untouched(size, std::byte{0});
    auto cancelled = file.read(second, {untouched}, {});
    cancelled->cancel();
    auto cancelledWrite = file.write(first, {later}, {});
    cancelledWrite->cancel();
    require(!file.idle(), "a file with queued work called itself idle");
    release.set_value();
    require(hold->wait() && write->wait() && !cancelled->wait() && !cancelledWrite->wait(),
            "queued operations misreported");
    require(file.idle(), "the queue drained but the file still held work");
    require(untouched.front() == std::byte{0} && untouched.back() == std::byte{0},
            "cancelled queued read touched destination");
    require(file.read(second, {restored}, {})->wait() && restored == later,
            "queued write did not land in order");
    require(file.writable() && !file.read(first, {restored}, {})->wait(),
            "cancelled overwrite exposed stale data or disabled the file");
    require(file.write(first, {later}, {})->wait() &&
                file.read(first, {restored}, {})->wait() && restored == later,
            "slot did not recover after a cancelled overwrite");

    // A completion is the caller's wake-up: one that throws is not the
    // worker's to die of.
    require(file.write(second, {later}, [] { throw std::runtime_error("completion"); })->wait(),
            "write with a throwing completion failed");
    require(file.read(second, {restored}, {})->wait() && restored == later,
            "the worker did not survive a throwing completion");
    first.reset();
    second.reset();
    auto reused = file.acquire();
    second = file.acquire();
    require(reused && second && !file.acquire(), "released quota was not reusable");
    require(throws<std::invalid_argument>([&] { SlotFile(size, size - 1); }),
            "quota below one slot was accepted");
    require(throws<std::invalid_argument>([&] { SlotFile(size + 1, 4 * size); }),
            "unaligned slot size was accepted");
    std::vector<std::byte> oversized(size + 1);
    require(throws<std::invalid_argument>(
                [&] { static_cast<void>(file.read(reused, {oversized}, {})); }) &&
                throws<std::invalid_argument>([&] {
                  static_cast<void>(file.write(
                      reused, {std::span<const std::byte>(source), std::span<const std::byte>(oversized).first(1)},
                      {}));
                }),
            "a transfer of more than a slot was accepted");
    {
      // Two files of different slot sizes draw on one budget.
      auto budget = std::make_shared<DiskBudget>(4 * size);
      SlotFile small(size, budget);
      SlotFile large(2 * size, budget);
      require(small.capacityBytes() == 4 * size && large.capacityBytes() == 4 * size &&
                  small.usedBytes() == 0,
              "files did not report the shared budget");
      auto one = large.acquire();
      auto two = small.acquire();
      require(one && two && small.usedBytes() == 3 * size && !large.acquire(),
              "the shared budget did not bound the second file");
      std::vector<std::byte> largePayload(2 * size, std::byte{3});
      require(large.write(one, {largePayload}, {})->wait() &&
                  small.write(two, {source}, {})->wait(), "shared IO writes failed");
      require(small.writtenBytes() == 3 * size && large.writtenBytes() == 3 * size &&
                  budget->writtenBytes() == 3 * size,
              "shared IO counters did not include both slot files");
      auto three = small.acquire();
      require(three && !small.acquire() && budget->usedBytes() == 4 * size,
              "the last slot of the budget was not granted exactly once");
      one.reset();
      require(budget->usedBytes() == 2 * size && large.acquire() && !small.acquire(),
              "a released slot did not return its bytes to the budget");
      require(throws<std::invalid_argument>(
                  [&] { SlotFile(8 * size, budget); }),
              "a file whose slot exceeds the shared budget was accepted");
    }
    std::cout << "Slot file tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
