#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

// How the prefill FFN's Neural Engine split (ops/AneFfn.hpp) runs on this
// device, from what ane_ffn::Measurement times: the ANE's share of a full
// chunk's FFN, and the least chunk the split takes. Nothing here touches
// Metal or the ANE.
namespace splash::ops::ane_ffn {

// The split's per-layer milliseconds of a full chunk with the ANE taking
// `share` of the intermediate channels (its channel units over all of
// them): the GPU's part alone, the ANE's evaluations alone, and both
// together.
struct Timing final {
  double share = 0.0, gpu = 0.0, ane = 0.0, both = 0.0;
};
// What calibration times: the split at two shares, and a full chunk's FFN
// layer on the GPU alone.
struct Timings final {
  Timing low, high;
  double gpuAlone = 0.0;
};

// `milliseconds`, which a measurement takes only as a duration it measured:
// finite and positive. Throws "calibration timings are not usable"
// otherwise.
[[nodiscard]] double usable(double milliseconds);

// The ANE's channel units nearest `share`, in (0, 1), of the `units` the
// intermediate channels hold: from one to all but one of them. Throws for
// another share or fewer than two units.
[[nodiscard]] uint32_t nearestUnits(double share, uint32_t units);

struct Line final {
  double at0 = 0.0, slope = 0.0;
  [[nodiscard]] double operator()(double share) const noexcept { return at0 + slope * share; }
};
// A full chunk's split FFN layer at share s takes
//   T(s) = max(G(s), A(s), uG G(s) + uA A(s)):
// G(s) is the GPU's part alone and A(s) the ANE's evaluations, each close to
// linear in s, and the third term the memory bandwidth both parts share when
// they run together, uG and uA each one's fraction of it alone.
struct Model final {
  Line gpu, ane;
  double uG = 0.0, uA = 0.0;
  [[nodiscard]] double operator()(double share) const noexcept;
};
// The model through the timings of two shares (AneFfnCalibration.cpp).
// Throws unless the shares differ.
[[nodiscard]] Model fit(const Timing &p, const Timing &q);

// The ANE's channel units, of the `units` the intermediate channels hold, at
// which full chunks split, at most `maxAneUnits` of them (AneFfnCalibration.cpp);
// none if `model` predicts no gain worth the split on `gpuAlone`
// milliseconds of the GPU alone. `previous`, the choice of the last start,
// stays while it is nearly as good and gains as much: each share compiles a
// program of its own.
[[nodiscard]] std::optional<uint32_t> choose(const Model &model, double gpuAlone, uint32_t units,
                                             uint32_t maxAneUnits, std::optional<uint32_t> previous);

// A chunk's FFN layer in milliseconds: the GPU's alone at the least and the
// most rows of the split's functions, and the split's of each function timed,
// from the most rows down, which chunks of `least` to `rows` rows run.
struct ChunkTimings final {
  struct Gpu final {
    uint32_t rows = 0;
    double milliseconds = 0.0;
  };
  struct Function final {
    uint32_t least = 0, rows = 0;
    double split = 0.0;
  };
  std::array<Gpu, 2> gpu{};
  std::vector<Function> functions;
};
// The least rows of a chunk the split takes: those from which every larger
// chunk of the functions timed splits (AneFfnCalibration.cpp); none when not
// even a chunk of the most rows does.
[[nodiscard]] std::optional<uint32_t> minimumRows(const ChunkTimings &timings);

} // namespace splash::ops::ane_ffn
