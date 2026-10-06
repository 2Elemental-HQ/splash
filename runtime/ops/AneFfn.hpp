#pragma once

#include "ane/Program.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/Linear.hpp"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace splash::ops {

// The dense FFN of a prefill chunk split by intermediate channel between the
// GPU and the Neural Engine. The GPU runs the leading channels with its
// prefill kernels over views of the projections' affine Q4 or GGUF planes;
// the ANE runs the rest as one W8A8 program over the chunk's rows, with int8
// weights the GPU requantizes from those planes one layer ahead into
// double-buffered surfaces, and the GPU adds the ANE's partial down projection
// to its own. A shared event orders each layer's ANE evaluation between the
// GPU's packing of its inputs and that join, within the one prefill command.
class AneFfn final {
public:
  // Chunks of kMinimumRows rows or more split, up to the prefill budget; the
  // GPU alone runs smaller ones faster. A chunk runs on the smallest of the
  // program's functions that holds it, one every kProgramStep rows, since an
  // evaluation costs the ANE its function's rows.
  static constexpr uint32_t kMinimumRows = 512, kProgramStep = 128, kMaximumRows = SPLASH_PREFILL_TOKEN_BUDGET;
  static_assert(kMaximumRows >= kMinimumRows && (kMaximumRows - kMinimumRows) % kProgramStep == 0,
                "the functions cover the chunks the split takes");
  [[nodiscard]] static constexpr bool splits(uint32_t rows) noexcept {
    return rows >= kMinimumRows && rows <= kMaximumRows;
  }

  // `share` is the fraction of intermediate channels the ANE takes. The
  // program is compiled and loaded, and each of its functions evaluated once,
  // here, so that a split the ANE cannot run fails before it serves.
  AneFfn(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
         double share);
  ~AneFfn();
  AneFfn(const AneFfn &) = delete;
  AneFfn &operator=(const AneFfn &) = delete;

  [[nodiscard]] double share() const noexcept { return share_; }
  // What Metal allocated for the split, which plannedBytes() bounds.
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return allocatedBytes_; }

  // Whether the split takes `layers`: affine Q4 or quantized GGUF projections
  // of one shape (AneFfn.cpp).
  [[nodiscard]] static bool supports(std::span<const SwiGluProjections> layers);
  // The Metal memory of the split of `layers` at `share`.
  [[nodiscard]] static uint64_t plannedBytes(std::span<const SwiGluProjections> layers, double share);

  struct Calibration final {
    // The fraction of intermediate channels the ANE takes; 0 when no share
    // beats the GPU alone by enough.
    double share = 0.0;
    // A full chunk's FFN layer on the GPU alone, as timed, and split at
    // `share`, as the timed model (AneFfn.cpp) predicts it.
    double gpuMilliseconds = 0.0, splitMilliseconds = 0.0;
    // The split's output against the GPU's alone over the timed layers: the
    // RMS of their difference relative to the GPU's.
    double error = 0.0;
  };
  // The share at which full chunks of `layers` (supports()) split on this
  // device, where the GPU's part and the ANE's take equally long (AneFfn.cpp),
  // timed on layers spread over the model with full chunks in
  // `ffn` and the residual and output of alternate layers in `hidden`, all of
  // which it overwrites. Throws if the split at that share computes other
  // values than the GPU alone, beyond int8's error.
  [[nodiscard]] static Calibration calibrate(metal::MetalBackend &backend, const Linear &linear,
                                             std::span<const SwiGluProjections> layers, const PrefillFfnBuffers &ffn,
                                             const std::array<metal::MetalBuffer, 2> &hidden);

  // Starts encoding a command, dropping the jobs of one never submitted.
  void begin() noexcept { jobs_.clear(); }
  // Layer `layer`'s FFN of a chunk of `rows` rows (splits(rows)), encoded in
  // layer order from layer 0 within each command: output = residual + FFN of
  // ffn.normalized, whose Q4 sums the norm wrote.
  void add(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn, metal::MetalBuffer residual,
           metal::MetalBuffer output, uint32_t rows);
  // Queues the ANE evaluations of the command encoded since begin(), before
  // the command is committed. If one cannot be queued, those queued are
  // released and drained, and the command must not be committed.
  void submit();
  // After the command has completed: waits for its evaluations to report,
  // and throws if one failed.
  void finish();

private:
  // The split's channels: hidden and intermediate, the layers', the GPU's and
  // the ANE's, and the segments of down's inputs the ANE takes.
  struct Shape final {
    uint32_t hidden = 0, intermediate = 0, layers = 0, gpu = 0, ane = 0;
    std::vector<uint32_t> downSegments;
  };
  // A staging set of the ANE's int8 weights: gate's and up's rows over each
  // segment of the hidden inputs, down's rows over each segment of its ANE
  // inputs, and each one's per-row scales.
  struct Weights final {
    std::vector<ane::Surface> gate, up, down;
    ane::Surface gateScale, upScale, downScale;
  };
  // Everything the split allocates (allocate() in AneFfn.cpp). The chunk's
  // rotated input rows in int8 segments, their per-token scales and the ANE's
  // partial down projection, which every function reads or writes, hold
  // kMaximumRows rows, whose row stride each function declares.
  struct Memory final {
    metal::MetalBuffer signs, rowScales, rotated;
    std::vector<ane::Surface> inputs;
    ane::Surface tokenScale, partial;
    std::array<Weights, 2> sets;
  };
  // A layer and the GPU's share of it: gate and up over views of their
  // leading rows, down over the leading inputs of its rows.
  struct Layer final {
    SwiGluProjections source;
    Projection gate, up, down;
  };
  // A matrix of a layer, whose per-row int8 scales rowScales() views.
  enum class Matrix : uint8_t { Gate, Up, Down };
  // An input of the ANE program's functions: its name, the surface each
  // weight set binds to it, and the rows and width the functions read of it,
  // a width of 0 reading the chunk's rows.
  struct Input final {
    std::string name;
    std::array<const ane::Surface *, 2> surfaces;
    uint32_t rows = 0, width = 0;
  };
  // What encode() adds of a split layer: the GPU's work alone (its channels,
  // the staging of the next layer's int8 weights, and the packing and join
  // of the ANE's channels), or with the ANE's evaluation, which the shared
  // event orders between the packing and the join.
  enum class Parts : uint8_t { Gpu, Both };
  // The program's function of `rows` rows over the surfaces of each weight
  // set, by set.
  struct Evaluation final {
    uint32_t rows = 0;
    std::vector<ane::Program::Binding> bindings;
  };
  // An evaluation of weight set `set`: it starts once the GPU raises the
  // event to `ready`, and raises it to `done`.
  struct Job final {
    uint32_t evaluation, set;
    uint64_t ready, done;
  };
  // Evaluations complete in the order they are queued.
  struct Completions final {
    std::mutex mutex;
    std::condition_variable changed;
    uint64_t completed = 0;
    bool failed = false;
  };

  AneFfn(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
         double share, std::span<const uint32_t> functionRows);
  [[nodiscard]] static Shape shapeOf(std::span<const SwiGluProjections> layers, double share);
  // Every allocation of the split, which `buffer` and `surface` make:
  // plannedBytes() counts them and the constructor allocates them.
  template <class MakeBuffer, class MakeSurface>
  [[nodiscard]] static Memory allocate(const Shape &shape, MakeBuffer &&buffer, MakeSurface &&surface);
  // The program's inputs; the program over them, with a function of each of
  // `rows` rows; and its function of `rows` rows.
  [[nodiscard]] static std::vector<Input> programInputs(const Shape &shape, const Memory &memory);
  [[nodiscard]] static std::string program(const Shape &shape, std::span<const Input> inputs,
                                           const ane::Surface &output, std::span<const uint32_t> rows);
  [[nodiscard]] static std::string function(const Shape &shape, std::span<const Input> inputs,
                                            const ane::Surface &output, uint32_t rows);
  // Calibration.error of `layers` split at `share` (AneFfn.cpp).
  [[nodiscard]] static double splitError(metal::MetalBackend &backend, const Linear &linear,
                                         std::span<const SwiGluProjections> layers, double share,
                                         const PrefillFfnBuffers &ffn, const std::array<metal::MetalBuffer, 2> &hidden);
  void encode(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn, metal::MetalBuffer residual,
              metal::MetalBuffer output, uint32_t rows, Parts parts);
  void addWeights(metal::CommandGraph &graph, uint32_t layer, uint32_t set) const;
  [[nodiscard]] metal::MetalBuffer rowScales(uint32_t layer, Matrix matrix) const;
  // An event value the CPU has raised the event to: a job that waits for it
  // starts at once.
  [[nodiscard]] uint64_t reached();
  // Waits for the queued evaluations, raising the event to each one's
  // `ready` first when `release` (no Metal command will).
  [[nodiscard]] bool wait(bool release);

  metal::MetalBackend &backend_;
  // A copy: the plans' owner may move or end before the split does.
  const Linear linear_;
  double share_ = 0.0;
  Shape shape_;
  std::vector<Layer> layers_;
  Memory memory_;
  std::unique_ptr<ane::Program> program_;
  // By rows, ascending.
  std::vector<Evaluation> evaluations_;
  metal::SharedEvent event_;
  uint64_t value_ = 0;
  // Encoded but not queued, and queued (the queued-th evaluation is the
  // last of queued_).
  std::vector<Job> jobs_, queued_;
  uint64_t queuedCount_ = 0;
  std::shared_ptr<Completions> completions_ = std::make_shared<Completions>();
  uint64_t allocatedBytes_ = 0;
};

} // namespace splash::ops
