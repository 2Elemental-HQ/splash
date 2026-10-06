#pragma once

#include "ops/AneFfn.hpp"
#include "ops/AneFfnCalibration.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <span>

namespace splash::ops::ane_ffn {

// Times the prefill FFN's Neural Engine split of `layers` (AneFfn) on this
// device, with chunks in `ffn` and the residual and output of alternate
// layers in `hidden`, all of which it overwrites. A program it compiles waits
// for the ANE's service until `interrupted` returns true. Each duration
// passes usable() where it is measured.
class Measurement final {
public:
  Measurement(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers, const PrefillFfnBuffers &ffn,
              const std::array<metal::MetalBuffer, 2> &hidden, std::function<bool()> interrupted);

  // The split's parts of full chunks, apart and together, with the ANE taking
  // the channel units nearest two shares, and the GPU's FFN alone, on layers
  // spread over the model, whose formats can differ with depth: per layer,
  // from commands of one and of more layers, the least of two runs each,
  // which cancels each command's fixed costs.
  [[nodiscard]] Timings time();
  // The GPU alone over the least and the most rows of `split`'s functions,
  // and `split`, of the layers, over a chunk of each function's rows through
  // begin(), add(), commit() and its completion check, from the most rows
  // down to the first function that does not split all its chunks
  // (ane_ffn::minimumRows), on the first layers (AneFfnMeasurement.cpp).
  [[nodiscard]] ChunkTimings chunks(AneFfn &split);

private:
  metal::MetalBackend &backend_;
  std::span<const SwiGluProjections> layers_;
  PrefillFfnBuffers ffn_;
  std::array<metal::MetalBuffer, 2> hidden_;
  std::function<bool()> interrupted_;
  const Linear linear_;
};

} // namespace splash::ops::ane_ffn
