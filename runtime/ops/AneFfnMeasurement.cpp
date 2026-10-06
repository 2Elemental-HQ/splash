#include "ops/AneFfnMeasurement.hpp"

#include "AwakeClock.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops::ane_ffn {
namespace {

// The shares calibration times the split's parts at.
constexpr std::array<double, 2> kTimedShares{0.4, 0.8};
// The layers of the longer command of each duration time() takes; one more
// layer gives the last one's next weights to stage.
constexpr uint32_t kTimedLayers = 4;
// The rows of layers of each command of chunks(), and its runs of each.
constexpr uint32_t kChunkLayerRows = 2 * AneFfn::kMaximumRows, kChunkRuns = 3;

// The per-layer milliseconds of `run`, which runs a number of layers and
// returns its milliseconds.
template <class Run> double perLayer(const Run &run) {
  const auto least = [&](uint32_t count) { return std::min(run(count), run(count)); };
  return usable((least(kTimedLayers) - least(1)) / (kTimedLayers - 1));
}

} // namespace

Measurement::Measurement(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers,
                         const PrefillFfnBuffers &ffn, const std::array<metal::MetalBuffer, 2> &hidden,
                         std::function<bool()> interrupted)
    : backend_(backend), layers_(layers), ffn_(ffn), hidden_(hidden), interrupted_(std::move(interrupted)),
      linear_(backend.capabilities()) {
  if (const char *reason = AneFfn::unsupported(layers)) throw std::invalid_argument(reason);
  // Timing does not depend on the values, but on their being finite.
  AneFfn::fillNormalized(ffn_.normalized);
  for (const metal::MetalBuffer &buffer : {ffn_.sums, hidden_[0], hidden_[1]})
    if (void *data = buffer.contents()) std::memset(data, 0, buffer.sizeBytes());
}

// Neither part waits on the other while timed: a command buffer that hands
// off to the ANE after little GPU work can start the next one late, by tens
// of milliseconds at random on the M5 Max with Qwen3.8-27B resident, which
// would time Metal's scheduling instead.
Timings Measurement::time() {
  const uint32_t units = AneFfn::units(layers_);
  // A model of fewer layers times some twice.
  std::vector<SwiGluProjections> sampled;
  for (uint32_t index = 0; index <= kTimedLayers; ++index)
    sampled.push_back(layers_[index * (layers_.size() - 1) / kTimedLayers]);
  constexpr uint32_t kRows = AneFfn::kMaximumRows;
  // The split timed: the GPU's part of a number of layers in one command
  // buffer, and the ANE's evaluations back to back, the first starting at
  // once.
  const auto measure = [&](double share) {
    const uint32_t aneUnits = nearestUnits(share, units);
    AneFfn split(backend_, sampled, aneUnits, interrupted_, std::array{kRows});
    const auto parts = [&](bool gpu, bool ane) {
      return [&, gpu, ane](uint32_t count) {
        metal::CommandGraph graph;
        static_cast<void>(split.begin());
        if (gpu)
          for (uint32_t layer = 0; layer < count; ++layer)
            split.encode(graph, layer, ffn_, hidden_[layer & 1], hidden_[(layer & 1) ^ 1], kRows,
                         AneFfn::Parts::Gpu);
        std::vector<std::pair<uint32_t, uint32_t>> evaluations;
        for (uint32_t layer = 0; layer < count; ++layer) evaluations.emplace_back(0, layer & 1);
        const auto start = AwakeClock::now();
        if (ane) split.queueNow(evaluations);
        if (gpu) static_cast<void>(backend_.submitCommandAsync(graph.dispatches()).wait());
        if (!split.handoff_.finish())
          throw std::runtime_error("ANE FFN evaluations failed (" + split.reason() + ")");
        return millisecondsSince(start);
      };
    };
    // The first command compiles and wires what the others run.
    static_cast<void>(parts(true, true)(kTimedLayers));
    return Timing{static_cast<double>(aneUnits) / units, perLayer(parts(true, false)), perLayer(parts(false, true)),
                  perLayer(parts(true, true))};
  };
  Timings timings{measure(kTimedShares[0]), measure(kTimedShares[1])};
  timings.gpuAlone = perLayer([&](uint32_t count) {
    metal::CommandGraph graph;
    for (uint32_t layer = 0; layer < count; ++layer)
      linear_.addPrefillSwiGlu(graph, sampled[layer], ffn_, hidden_[layer & 1], hidden_[(layer & 1) ^ 1], kRows);
    const auto start = AwakeClock::now();
    static_cast<void>(backend_.submitCommandAsync(graph.dispatches()).wait());
    return millisecondsSince(start);
  });
  return timings;
}

// Commands of the split's layers, and of the GPU's alone, of a chunk's rows.
// Each layer follows the GPU alone's FFN of the same layer into the other
// hidden rows, which stands in for the mixer before it in a prefill: the ANE
// idles as long between its evaluations, and a layer that hands its inputs to
// the ANE after little GPU work can start the next command buffer late
// (time()). The stand-in takes as long as the GPU's layer alone, half of each
// of the GPU's command's layers. Each command takes about kChunkLayerRows
// rows of layers, more layers of fewer rows, over which the jitter of the
// handoffs averages out, and the median of kChunkRuns of each, which passes
// over a first run that wires a function, gives a layer's milliseconds. Each
// layer timed stages the next one's weights.
ChunkTimings Measurement::chunks(AneFfn &split) {
  if (layers_.size() < 2) throw std::invalid_argument("ANE FFN chunks timed on fewer than two layers");
  const auto layers = [&](uint32_t rows) {
    return static_cast<uint32_t>(std::clamp<size_t>(kChunkLayerRows / rows, 1, layers_.size() - 1));
  };
  const auto command = [&](bool splits, uint32_t rows, uint32_t count) {
    metal::CommandGraph graph;
    if (splits && !split.begin()) throw std::runtime_error("ANE FFN split stopped (" + split.reason() + ")");
    for (uint32_t layer = 0; layer < count; ++layer) {
      linear_.addPrefillSwiGlu(graph, layers_[layer], ffn_, hidden_[0], hidden_[1], rows);
      if (splits)
        split.add(graph, layer, ffn_, hidden_[1], hidden_[0], rows);
      else
        linear_.addPrefillSwiGlu(graph, layers_[layer], ffn_, hidden_[1], hidden_[0], rows);
    }
    const auto start = AwakeClock::now();
    if (splits) {
      static_cast<void>(split.commit(graph, {}).wait());
      if (!split.completed())
        throw std::runtime_error("ANE FFN split of " + std::to_string(rows) + " rows failed (" + split.reason() +
                                 ")");
    } else {
      static_cast<void>(backend_.submitCommandAsync(graph.command()).wait());
    }
    return millisecondsSince(start) / count;
  };
  const auto median = [&](bool splits, uint32_t rows) {
    std::array<double, kChunkRuns> runs{};
    for (double &run : runs) run = command(splits, rows, layers(rows));
    std::ranges::nth_element(runs, runs.begin() + kChunkRuns / 2);
    return usable(runs[kChunkRuns / 2]);
  };
  const std::vector<AneFfn::Evaluation> &functions = split.evaluations_;
  const auto gpu = [&](uint32_t rows) { return ChunkTimings::Gpu{rows, median(false, rows) / 2}; };
  ChunkTimings timings{{gpu(functions.front().rows), gpu(functions.back().rows)}, {}};
  const auto &[low, high] = timings.gpu;
  const double slope = (high.milliseconds - low.milliseconds) / (high.rows - low.rows);
  for (size_t index = functions.size(); index-- > 0;) {
    const uint32_t rows = functions[index].rows, least = index ? functions[index - 1].rows + 1 : AneFfn::kMinimumRows;
    const double standIn = low.milliseconds + slope * (rows - low.rows);
    timings.functions.push_back({least, rows, usable(median(true, rows) - standIn)});
    if (minimumRows(timings) != least) break;
  }
  return timings;
}

} // namespace splash::ops::ane_ffn
