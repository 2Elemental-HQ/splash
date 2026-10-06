#include "ops/AneFfnCalibration.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace splash::ops::ane_ffn {
namespace {

// The model. Timing each part alone at two shares gives G and A, and both
// together at those shares gives uG and uA where the bandwidth binds there. The
// split runs where G meets A, near the fastest share of a prefill of seconds
// measured on Qwen3.8-27B's FFN: the M5 Max's 0.24, the M5 Pro's 0.41 and the
// M6's 0.76. A is timed over evaluations back to back, without the mixer's
// time between them in a prefill. Short of it the ANE waits on
// the GPU, whose part the third term slows where the two share bandwidth, as on
// the M6; past it the GPU waits on the ANE, which T takes to run as fast as
// alone. It does not: on the M6 the two slow each other's memory accesses
// beyond the GPU's L2 (an ANE evaluation takes a third longer beside a GPU
// kernel that streams memory), and a minute of prefill warms the ANE a tenth
// slower. There these timings put the fastest share at 0.85-0.88, which a
// 14K-token prompt runs 7% slower than 0.76. The split takes the least share
// within kTolerance of it, for the ANE's memory and error, if T predicts a gain
// of kMinimumGain there on the GPU alone, timed as well.
//
// The excess of both parts together over the longer alone that shows the
// bandwidth binding; the most of the bandwidth either part is solved to
// take, a little above all of it for the timings' noise; and the least
// determinant of the two shares' timings, relative to its terms, that
// solves for uG and uA.
constexpr double kBindingMargin = 0.03, kMaximumUtilization = 1.05, kSingular = 1e-6;
// The split's predicted gain below which the GPU runs the FFN alone, and the
// slack of the fastest share.
constexpr double kMinimumGain = 0.05, kTolerance = 0.01;
// How much slower than the best a share's max(G, A) may be and the share
// stay the choice of the last start: keeping a share a few percent slower
// than the best costs that much of the prefill FFN; changing it costs a
// compile, 29 s on an M6. Five back-to-back calibrations of Qwen3.8-27B on an
// M5 Max put the best max(G, A) within 1.1% (MLX) and 1.7% (GGUF) of one
// another; this is about twice that noise.
constexpr double kHysteresis = 0.03;
// How much longer than the GPU's alone a chunk's split layer may take and
// the chunk still split: six timings of each function's split layer of
// Qwen3.8-27B on an M5 Max spread over ±3%, and the FFN takes about half of
// a prefill, so a chunk that splits within this prefills at most about 1%
// slower.
constexpr double kChunkNoise = 0.02;

// The GPU alone's milliseconds of a chunk's FFN layer by its rows: the line
// through its timings, which the GPU follows within 1% between them.
Line gpuAlone(const ChunkTimings &timings) {
  const auto &[low, high] = timings.gpu;
  const double slope = (high.milliseconds - low.milliseconds) / (high.rows - low.rows);
  return {low.milliseconds - slope * low.rows, slope};
}

} // namespace

double usable(double milliseconds) {
  if (!(milliseconds > 0.0 && std::isfinite(milliseconds)))
    throw std::runtime_error("calibration timings are not usable");
  return milliseconds;
}

uint32_t nearestUnits(double share, uint32_t units) {
  if (!(share > 0.0 && share < 1.0) || units < 2)
    throw std::invalid_argument("ANE FFN share must lie in (0, 1) of two units or more");
  return static_cast<uint32_t>(std::clamp<long>(std::lround(share * units), 1, units - 1));
}

double Model::operator()(double share) const noexcept {
  const double g = gpu(share), a = ane(share);
  return std::max({g, a, uG * g + uA * a});
}

Model fit(const Timing &p, const Timing &q) {
  if (!(p.share != q.share)) throw std::invalid_argument("ANE FFN calibration needs two shares");
  const auto line = [&](double Timing::*part) {
    const double slope = (q.*part - p.*part) / (q.share - p.share);
    return Line{p.*part - slope * p.share, slope};
  };
  Model model{line(&Timing::gpu), line(&Timing::ane)};
  const auto excess = [](const Timing &x) { return x.both / std::max(x.gpu, x.ane) - 1.0; };
  bool solved = false;
  if (excess(p) > kBindingMargin && excess(q) > kBindingMargin) {
    const double det = p.gpu * q.ane - p.ane * q.gpu;
    if (std::abs(det) > kSingular * (std::abs(p.gpu * q.ane) + std::abs(p.ane * q.gpu))) {
      model.uG = (p.both * q.ane - p.ane * q.both) / det;
      model.uA = (p.gpu * q.both - p.both * q.gpu) / det;
      solved = model.uG >= 0.0 && model.uG <= kMaximumUtilization && model.uA >= 0.0 &&
               model.uA <= kMaximumUtilization;
    }
  }
  if (!solved) {
    // Binding at one share: the GPU's part alone nearly saturates memory, so
    // that share's excess over it is the ANE's fraction.
    const Timing &x = excess(p) >= excess(q) ? p : q;
    model.uG = excess(x) > kBindingMargin ? 1.0 : 0.0;
    model.uA = model.uG > 0.0 ? (x.both - x.gpu) / x.ane : 0.0;
  }
  model.uG = std::clamp(model.uG, 0.0, 1.0);
  model.uA = std::clamp(model.uA, 0.0, 1.0);
  return model;
}

// The ANE units where the GPU's part and the ANE's alone take most nearly as
// long: `previous` while it is within kHysteresis of the least max(G, A) and
// gains, else the least within kTolerance of it, if that gains.
std::optional<uint32_t> choose(const Model &model, double gpuAlone, uint32_t units, uint32_t maxAneUnits,
                               std::optional<uint32_t> previous) {
  if (!maxAneUnits || maxAneUnits >= units) throw std::invalid_argument("ANE FFN choice of no feasible units");
  const auto share = [&](uint32_t aneUnits) { return static_cast<double>(aneUnits) / units; };
  const auto longer = [&](uint32_t aneUnits) { return std::max(model.gpu(share(aneUnits)), model.ane(share(aneUnits))); };
  const auto gains = [&](uint32_t aneUnits) { return model(share(aneUnits)) <= (1.0 - kMinimumGain) * gpuAlone; };
  double best = std::numeric_limits<double>::infinity();
  for (uint32_t aneUnits = 1; aneUnits <= maxAneUnits; ++aneUnits) best = std::min(best, longer(aneUnits));
  if (!(best > 0.0 && std::isfinite(best))) return std::nullopt;
  if (previous && *previous >= 1 && *previous <= maxAneUnits && longer(*previous) <= (1.0 + kHysteresis) * best &&
      gains(*previous))
    return previous;
  for (uint32_t aneUnits = 1; aneUnits <= maxAneUnits; ++aneUnits)
    if (longer(aneUnits) <= (1.0 + kTolerance) * best) return gains(aneUnits) ? std::optional(aneUnits) : std::nullopt;
  return std::nullopt;
}

// A chunk splits while its function's split layer takes at most kChunkNoise
// longer than the GPU's layer alone of the chunk's rows (gpuAlone). Every
// chunk a function runs takes about as long as the function's own rows: the
// ANE's part runs them and takes the longer. The functions do not cost the
// ANE in proportion to their rows (on an M5 Max 640 rows took 93% of 768's,
// 512 rows 73%), so each is timed, and each row count it runs checked: a
// batch's chunk sums its lanes' rows.
std::optional<uint32_t> minimumRows(const ChunkTimings &timings) {
  const auto &[low, high] = timings.gpu;
  const std::vector<ChunkTimings::Function> &functions = timings.functions;
  bool ordered = !functions.empty() && low.rows < high.rows;
  for (size_t index = 0; ordered && index < functions.size(); ++index)
    ordered = functions[index].least && functions[index].least <= functions[index].rows &&
              (!index || functions[index].rows + 1 == functions[index - 1].least);
  if (!ordered) throw std::invalid_argument("ANE FFN chunks are not those of its functions");
  const Line gpu = gpuAlone(timings);
  std::optional<uint32_t> minimum;
  for (const ChunkTimings::Function &function : functions)
    for (uint32_t rows = function.rows; rows >= function.least; --rows) {
      if (!(function.split <= (1.0 + kChunkNoise) * gpu(rows))) return minimum;
      minimum = rows;
    }
  return minimum;
}

// Each layer the GPU waits for the Neural Engine's evaluation before it joins
// the ANE's part to its own, so evaluations that take as long as the GPU's
// whole layers alone make the split slower than the GPU alone, whatever the
// rest of the layer takes: the breaker trips there, and never short of it. An
// evaluation counts against the GPU alone's layer of its function's rows,
// which the chunk pads up to, so the GPU alone would take no longer over the
// chunk; the reason scales the window's ratio to a layer of the most rows.
// At the share calibration chooses, an evaluation takes about as long as the
// GPU's part (choose), about 1 - share of the GPU alone: on an M5 Max,
// calibration's timings of Qwen3.8-27B put it at 14.0 ms against 19.0 at 9
// of 34 units, and the M5 Pro's 0.41 and the M6's 0.76 leave more room.
// Measured, another process keeping the ANE busy beside the split costs it
// 2-8% on an M5 Max, an M5 Pro and an M6, and a minute of prefill slows the
// M6's ANE by a third: far short of the trip point. Judging kWindow commands
// at once, it weighs one slow evaluation, as a function's first after its
// program loads again, among all of theirs.
Breaker::Breaker(const ChunkTimings &timings) {
  const auto &[low, high] = timings.gpu;
  if (!(low.rows < high.rows)) throw std::invalid_argument("ANE FFN breaker needs the GPU alone at two row counts");
  static_cast<void>(usable(low.milliseconds));
  static_cast<void>(usable(high.milliseconds));
  gpu_ = gpuAlone(timings);
  mostRows_ = high.rows;
}

std::string Breaker::add(uint32_t rows, uint32_t evaluations, double milliseconds) {
  if (!gpu_) return {};
  ane_ += milliseconds;
  alone_ += evaluations * (*gpu_)(rows);
  if (++commands_ < kWindow) return {};
  const double ratio = ane_ / alone_, most = (*gpu_)(mostRows_);
  commands_ = 0;
  ane_ = alone_ = 0.0;
  if (!(ratio >= 1.0)) return {};
  std::ostringstream reason;
  reason << std::fixed << std::setprecision(1) << "losing to the GPU alone (" << ratio * most
         << " ms on the Neural Engine per " << mostRows_ << "-row layer against " << most << " ms on the GPU alone)";
  return reason.str();
}

} // namespace splash::ops::ane_ffn
