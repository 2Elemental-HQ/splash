#include "tuning/AttentionTuning.hpp"

#include "tuning/AttentionFixture.hpp"
#include "tuning/HostKvExtents.hpp"
#include "tuning/LinearNumerics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace splash::ops::tuning {
namespace {

using Clock = std::chrono::steady_clock;
using Tensor = AttentionFixture::Tensor;
constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kDimension = AttentionFixturePlan::kHeadDimension;
constexpr uint32_t kVerifyRows = SPLASH_TARGET_VERIFY_ROWS;

// A workload's fixture: its one attention layer in as few extents as the
// page index allows, with scratch for every candidate's plan.
template <typename Workload> AttentionFixturePlan fixturePlan(Workload workload) {
  const kv::Layout layout{1, workload.shape.kvHeads, workload.shape.headDimension,
                          workload.shape.format};
  uint32_t lanes = 1, rows = 0;
  AttentionFixturePlan::Histories histories{};
  AttentionWorkspace scratch;
  const auto cover = [&](AttentionWorkspace workspace) {
    scratch.partialsBytes = std::max(scratch.partialsBytes, workspace.partialsBytes);
    scratch.statisticsBytes = std::max(scratch.statisticsBytes, workspace.statisticsBytes);
  };
  if constexpr (std::is_same_v<Workload, PrefillAttentionWorkload>) {
    rows = workload.rows;
    histories[0] = workload.historyTokens;
    for (auto config : PagedAttention::prefillCandidates())
      cover(PagedAttention::prefillPlan(rows, workload.shape.queryHeads, layout,
                                        workload.historyTokens, config).workspace);
  } else {
    lanes = workload.lanes;
    rows = kVerifyRows;
    histories = workload.historyTokens;
    for (auto config : PagedAttention::verifyCandidates())
      cover(PagedAttention::verifyPlan(lanes, workload.shape.queryHeads, layout, histories,
                                       config).workspace);
  }
  const uint32_t poolPages =
      AttentionFixturePlan::poolPagesFor(AttentionFixturePlan::pagesOf(lanes, rows, histories));
  return AttentionFixturePlan::make(workload.shape, lanes, rows, histories, scratch,
                                    {1, 0, HostKvExtents::aligned(layout, poolPages).extentPages});
}

struct NumericalMismatch final : std::runtime_error {
  NumericalMismatch() : std::runtime_error("attention tuning candidate failed output qualification") {}
};

struct Interrupted final {
  MeasurementStatus status;
};

// The trials of one workload on its fixture: graphs of repeated production
// attention, the reset before each run, and output qualification against the
// baseline's output, which the host keeps.
class TuningFixture final {
public:
  TuningFixture(metal::MetalBackend &backend, AttentionFixturePlan plan)
      : fixture_(backend, std::move(plan), "attention-tuning-fixture") {}

  // Called once after admission. Long histories periodically consult the same
  // sweep control callback; no GPU command has been submitted at this point.
  bool initialize(const MeasurementStop &stop) { return fixture_.fill(stop); }

  template <typename Config> metal::CommandGraph graph(Config config,
                                                       uint32_t repetitions = 1) const {
    const auto &plan = fixture_.plan();
    metal::CommandGraph result;
    const auto attentionPlan = [&] {
      if constexpr (std::is_same_v<Config, PrefillAttentionConfig>)
        return PagedAttention::prefillPlan(plan.rows, plan.shape.queryHeads,
                                           plan.layout(), plan.histories[0], config);
      else
        return PagedAttention::verifyPlan(plan.lanes, plan.shape.queryHeads,
                                          plan.layout(), plan.histories, config);
    }();
    // Each repeated subgraph begins with the deterministic KV store and ends
    // with split reduction. It never consumes the previous attention output:
    // queries/chunk K/V/history remain unchanged and current KV slots are
    // overwritten with identical values. Do not repeat individual dispatches.
    for (uint32_t repetition = 0; repetition < repetitions; ++repetition)
      fixture_.addGraph(result, attentionPlan);
    return result;
  }

  void reset() {
    for (Tensor tensor : {Tensor::Output, Tensor::Partials, Tensor::Statistics}) {
      const metal::MetalBuffer buffer = fixture_.buffer(tensor);
      std::memset(buffer.contents(), 0, buffer.sizeBytes());
    }
    // Only current-row cache slots are mutated by the production store.
    // Restore those exact slots; immutable history, input and tables remain.
    const auto clear = [](auto *keys, auto *values) {
      for (uint32_t d = 0; d < kDimension; ++d) {
        keys[d] = 0;
        values[d * kv::kPageTokens] = 0;
      }
    };
    const auto &plan = fixture_.plan();
    for (uint32_t lane = 0; lane < plan.lanes; ++lane)
      for (uint32_t row = 0; row < plan.rows; ++row)
        for (uint32_t head = 0; head < plan.shape.kvHeads; ++head) {
          const uint32_t token = plan.histories[lane] + row;
          if (plan.shape.format == kv::Format::Int8) {
            *fixture_.scale(lane, SPLASH_KV_KEY_SCALES, head, token) = 0;
            *fixture_.scale(lane, SPLASH_KV_VALUE_SCALES, head, token) = 0;
            clear(fixture_.keyRow<int8_t>(lane, head, token),
                  fixture_.valueColumn<int8_t>(lane, head, token));
          } else {
            clear(fixture_.keyRow<uint16_t>(lane, head, token),
                  fixture_.valueColumn<uint16_t>(lane, head, token));
          }
        }
  }

  void qualify(bool baseline) {
    const auto &plan = fixture_.plan();
    const metal::MetalBuffer output = fixture_.buffer(Tensor::Output);
    const auto *values = static_cast<const uint16_t *>(output.contents());
    double dot = 0, refSquared = 0, outSquared = 0;
    float maximumError = 0;
    for (uint32_t lane = 0; lane < plan.lanes; ++lane)
      for (uint32_t head = 0; head < plan.shape.queryHeads; ++head)
        for (uint32_t row = 0; row < plan.rows; ++row)
          for (uint32_t d = 0; d < kDimension; ++d) {
            const uint64_t index = plan.queryIndex(lane, head, row, d);
            const float right = bf16ToFloat(values[index]);
            if (!std::isfinite(right)) throw NumericalMismatch();
            if (reference_.empty()) continue;
            const float left = bf16ToFloat(reference_[index]);
            maximumError = std::max(maximumError, std::abs(left - right));
            dot += double(left) * right;
            refSquared += double(left) * left;
            outSquared += double(right) * right;
          }
    if (reference_.empty()) {
      if (!baseline) throw std::logic_error("attention baseline was not measured first");
      reference_.assign(values, values + output.sizeBytes() / sizeof(uint16_t));
    } else if (!(maximumError < 0.02f) || !(refSquared > 0) || !(outSquared > 0) ||
               !(dot / std::sqrt(refSquared * outSquared) > 0.9995)) {
      throw NumericalMismatch();
    }
  }

private:
  AttentionFixture fixture_;
  // The baseline's output, once it ran.
  std::vector<uint16_t> reference_;
};

template <typename Workload, typename Config>
bool equivalentToBaseline(const Workload &workload, Config baseline, Config config) {
  const kv::Layout layout{1, workload.shape.kvHeads, workload.shape.headDimension, workload.shape.format};
  const auto plan = [&](Config selected) {
    if constexpr (std::is_same_v<Config, PrefillAttentionConfig>)
      return PagedAttention::prefillPlan(workload.rows, workload.shape.queryHeads,
                                         layout, workload.historyTokens, selected);
    else
      return PagedAttention::verifyPlan(workload.lanes, workload.shape.queryHeads, layout,
                                        workload.historyTokens, selected);
  };
  const auto base = plan(baseline), candidate = plan(config);
  return base.sameExecutionAs(candidate);
}

// configurations is the tuning order: its front is the baseline every other
// entry is measured against, and candidate IDs index it.
template <typename Result, typename Workload, typename Config>
Result tune(metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
            Workload workload, std::span<const Config> configurations,
            const MeasurementOptions &options, const MeasurementStop &pressure,
            const MeasurementStop &stop) {
  if (configurations.empty()) throw std::logic_error("attention tuning has no baseline");
  const Config baseline = configurations.front();
  Result result{{workload, baseline}, {}, false, {}};
  const auto started = Clock::now();
  auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - started).count(); };
  auto interrupted = [&] {
    return (stop && stop()) || (pressure && pressure()) || elapsed() >= options.maximumWallSeconds;
  };
  try {
    const auto plan = fixturePlan(workload);
    if (!admit || !validMeasurementOptions(options) || interrupted() ||
        plan.bytes > backend.capabilities().maxBufferLengthBytes)
      return result;
    std::unique_ptr<TuningFixture> fixture;
    bool invoked = false;
    const auto admitted = admit(plan.bytes, [&] {
      if (invoked) throw std::logic_error("attention fixture admission invoked twice");
      invoked = true;
      fixture = std::make_unique<TuningFixture>(backend, plan);
    });
    if (!admitted) {
      if (fixture) throw std::logic_error("attention admission denied after retaining allocation");
      return result;
    }
    if (!invoked || !fixture)
      throw std::logic_error("attention admission succeeded without allocation");
    if (interrupted() || !fixture->initialize(interrupted))
      return result;
    const auto requireControl = [&] {
      if (stop && stop()) throw Interrupted{MeasurementStatus::Cancelled};
      if (pressure && pressure()) throw Interrupted{MeasurementStatus::UnderPressure};
      if (elapsed() >= options.maximumWallSeconds)
        throw Interrupted{MeasurementStatus::BudgetExceeded};
    };
    const auto qualificationRun = [&](Config config, uint32_t repetitions) {
      requireControl();
      fixture->reset();
      requireControl();
      const auto graph = fixture->graph(config, repetitions);
      const auto timing = backend.submitCommand(graph.dispatches());
      requireControl();
      if (!std::isfinite(timing.gpuSeconds) || timing.gpuSeconds <= 0 ||
          !std::isfinite(timing.wallSeconds) || timing.wallSeconds <= 0)
        throw std::runtime_error("invalid attention qualification timing");
      return timing;
    };
    // Reuse the ordinary single-baseline qualification as the GPU pilot.
    // The resulting count is fixed for the entire workload, never tuned per
    // candidate or inferred from the faster side of a measured pair.
    const auto baselinePilot = qualificationRun(baseline, 1);
    fixture->qualify(true);
    result.repetitions = measurementBatchRepetitions(baselinePilot.gpuSeconds);
    if (result.repetitions > 1) {
      (void)qualificationRun(baseline, result.repetitions);
      // Repeated complete graphs use the same numerical contract as a
      // single candidate; normal floating-point differences are permitted.
      fixture->qualify(false);
    }
    for (size_t candidate = 1; candidate < configurations.size(); ++candidate) {
      const Config config = configurations[candidate];
      if (equivalentToBaseline(workload, baseline, config)) {
        result.equivalentCandidates.push_back({static_cast<uint32_t>(candidate)});
        continue;
      }
      if (interrupted()) return result;
      (void)qualificationRun(config, 1);
      fixture->qualify(false);
      if (result.repetitions > 1) {
        (void)qualificationRun(config, result.repetitions);
        fixture->qualify(false);
      }
      if (interrupted()) return result;
      auto remainingOptions = options;
      remainingOptions.maximumWallSeconds = options.maximumWallSeconds - elapsed();
      const CandidateId id{static_cast<uint32_t>(candidate)};
      auto measurement = measureWorkload(
          id, {0}, [&](CandidateId selected) {
            if (pressure && pressure()) return RunTiming{0, 0, true};
            fixture->reset();
            if (pressure && pressure()) return RunTiming{0, 0, true};
            if (stop && stop()) throw Interrupted{MeasurementStatus::Cancelled};
            if (elapsed() >= options.maximumWallSeconds)
              throw Interrupted{MeasurementStatus::BudgetExceeded};
            const auto wallStart = Clock::now();
            const auto graph = fixture->graph(selected == kBaseline ? baseline : config,
                                               result.repetitions);
            const auto timing = backend.submitCommand(graph.dispatches());
            const double wallSeconds =
                std::chrono::duration<double>(Clock::now() - wallStart).count();
            return RunTiming{timing.gpuSeconds, wallSeconds,
                              pressure && pressure()};
          }, remainingOptions, stop);
      if (measurement.status == MeasurementStatus::RunFailed && measurement.failure) {
        try { std::rethrow_exception(measurement.failure); }
        catch (const Interrupted &interruption) {
          measurement.status = interruption.status;
          measurement.failure = {};
        }
        catch (...) {}
      }
      const auto status = measurement.status;
      result.measurements.push_back(std::move(measurement));
      if (status != MeasurementStatus::Completed && status != MeasurementStatus::Rejected) {
        result.failure = result.measurements.back().failure;
        return result;
      }
    }
    if (interrupted()) return result;
    std::vector<WorkloadMeasurements> gpuWorkloads, wallWorkloads;
    gpuWorkloads.reserve(result.measurements.size());
    wallWorkloads.reserve(result.measurements.size());
    for (const auto &measurement : result.measurements) {
      // Each metric independently sees every finished candidate, including
      // one rejected by the OTHER metric. Filtering those records could hide
      // disagreement between their winners.
      gpuWorkloads.push_back({{0}, measurement.rawGpuSamples()});
      wallWorkloads.push_back({{0}, measurement.rawWallSamples()});
    }
    std::vector<CandidateMeasurements> gpuCandidates, wallCandidates;
    for (size_t i = 0; i < result.measurements.size(); ++i) {
      gpuCandidates.push_back({result.measurements[i].candidate, {&gpuWorkloads[i], 1}});
      wallCandidates.push_back({result.measurements[i].candidate, {&wallWorkloads[i], 1}});
    }
    constexpr std::array required{WorkloadId{0}};
    const auto gpu = selectCandidate(gpuCandidates, required, options.policy);
    const auto wall = selectCandidate(wallCandidates, required, options.policy);
    if (gpu.verdict == SelectionVerdict::Selected &&
        wall.verdict == SelectionVerdict::Selected && gpu.candidate == wall.candidate)
      result.choice.configuration = configurations[gpu.candidate.value];
    result.complete = true;
  } catch (const Interrupted &) {
    // Prequalification control boundaries have no partial timing sample to
    // report. They still stop the entire sweep without selecting a winner.
  } catch (...) {
    result.failure = std::current_exception();
  }
  return result;
}

template <typename Config, typename Probe, typename Workload>
Config selectPolicy(std::span<const Probe> probes, std::span<const Workload> required,
                     std::span<const Config> configurations, const Policy &policy) {
  if (configurations.empty()) throw std::logic_error("attention tuning has no baseline");
  const Config baseline = configurations.front();
  if (probes.size() != required.size() || required.empty()) return baseline;
  const size_t candidates = configurations.size() - 1;
  std::vector<WorkloadId> ids;
  std::vector<std::vector<WorkloadMeasurements>> gpu(candidates), wall(candidates);
  for (size_t i = 0; i < probes.size(); ++i) {
    const auto &probe = probes[i];
    if (!probe.complete || probe.failure || probe.choice.workload != required[i] ||
        probe.measurements.size() + probe.equivalentCandidates.size() != candidates)
      return baseline;
    ids.push_back({static_cast<uint32_t>(i)});
    for (size_t c = 1; c < configurations.size(); ++c) {
      const CandidateId id{static_cast<uint32_t>(c)};
      const auto equivalent = std::count(probe.equivalentCandidates.begin(),
                                         probe.equivalentCandidates.end(), id);
      const MeasurementResult *measured = nullptr;
      for (const auto &measurement : probe.measurements) {
        if (measurement.candidate != id) continue;
        if (measured) return baseline;
        measured = &measurement;
      }
      if (equivalent > 1 || (equivalent != 0) == (measured != nullptr)) return baseline;
      if (equivalent) {
        // Re-check structural evidence at the typed-plan boundary. Never
        // infer equivalence from an absent record or a measured zero gain.
        if (!equivalentToBaseline(required[i], baseline, configurations[c])) return baseline;
        gpu[c - 1].push_back({ids.back(), {}, true});
        wall[c - 1].push_back({ids.back(), {}, true});
      } else {
        if (measured->failure || (measured->status != MeasurementStatus::Completed &&
            measured->status != MeasurementStatus::Rejected)) return baseline;
        gpu[c - 1].push_back({ids.back(), measured->rawGpuSamples()});
        wall[c - 1].push_back({ids.back(), measured->rawWallSamples()});
      }
    }
  }
  std::vector<CandidateMeasurements> gpuCandidates, wallCandidates;
  for (size_t c = 0; c < candidates; ++c) {
    gpuCandidates.push_back({{static_cast<uint32_t>(c + 1)}, gpu[c]});
    wallCandidates.push_back({{static_cast<uint32_t>(c + 1)}, wall[c]});
  }
  const auto selectedGpu = selectCandidate(gpuCandidates, ids, policy);
  const auto selectedWall = selectCandidate(wallCandidates, ids, policy);
  if (selectedGpu.verdict == SelectionVerdict::Selected &&
      selectedWall.verdict == SelectionVerdict::Selected &&
      selectedGpu.candidate == selectedWall.candidate)
    return configurations[selectedGpu.candidate.value];
  return baseline;
}

template <typename Result, typename PolicyKey, typename Config, typename Workloads,
          typename Run, typename Select>
Result tunePolicy(PolicyKey key, Config baseline, const Workloads &workloads,
                    const MeasurementOptions &options, const MeasurementStop &pressure,
                    const MeasurementStop &stop, Run run, Select select) {
  Result result{{key, baseline}, {}, false, {}};
  const auto started = Clock::now();
  const auto remaining = [&] {
    return options.maximumWallSeconds - std::chrono::duration<double>(Clock::now() - started).count();
  };
  const auto interrupted = [&] {
    return (stop && stop()) || (pressure && pressure()) || remaining() <= 0;
  };
  try {
    if (!validMeasurementOptions(options)) return result;
    for (const auto &workload : workloads) {
      if (interrupted()) return result;
      auto probeOptions = options;
      probeOptions.maximumWallSeconds = remaining();
      result.probes.push_back(run(workload, probeOptions));
      const auto &probe = result.probes.back();
      if (!probe.complete || probe.failure) {
        result.failure = probe.failure;
        return result;
      }
    }
    if (interrupted()) return result;
    result.choice.configuration = select(result.probes, options.policy);
    result.complete = true;
  } catch (...) {
    result.failure = std::current_exception();
  }
  return result;
}

} // namespace

std::array<PrefillAttentionWorkload, 4>
prefillAttentionPolicyWorkloads(AttentionShape shape) {
  constexpr uint32_t rows = SPLASH_PREFILL_TOKEN_BUDGET;
  (void)PagedAttention::prefillPlan(rows, shape.queryHeads,
                                   {1, shape.kvHeads, shape.headDimension, shape.format}, 0);
  return {{{shape, rows, 0}, {shape, rows, 2048},
           {shape, rows, 16384}, {shape, rows, 131072}}};
}

std::vector<VerifyAttentionWorkload>
verifyAttentionPolicyWorkloads(VerifyAttentionPolicy policy) {
  const auto shape = policy.shape;
  const std::array<uint32_t, kMaximumLanes> histories{};
  (void)PagedAttention::verifyPlan(policy.lanes, shape.queryHeads,
                                    {1, shape.kvHeads, shape.headDimension, shape.format}, histories);
  std::vector<VerifyAttentionWorkload> result;
  for (uint32_t history : {1U, 25U, 2048U, 131072U}) {
    VerifyAttentionWorkload workload{shape, policy.lanes, {}};
    std::fill_n(workload.historyTokens.begin(), policy.lanes, history);
    result.push_back(workload);
  }
  if (policy.lanes > 1) {
    constexpr std::array anchors{25U, 2049U, 8191U, 131072U};
    VerifyAttentionWorkload mixed{shape, policy.lanes, {}};
    for (uint32_t lane = 0; lane < policy.lanes; ++lane)
      mixed.historyTokens[lane] = anchors[lane * 3 / (policy.lanes - 1)];
    result.push_back(mixed);
    std::reverse(mixed.historyTokens.begin(), mixed.historyTokens.begin() + policy.lanes);
    result.push_back(mixed);
  }
  return result;
}

std::vector<VerifyAttentionConfig>
verifyAttentionTuningCandidates(VerifyAttentionConfig baseline) {
  const auto operators = PagedAttention::verifyCandidates();
  if (std::count(operators.begin(), operators.end(), baseline) != 1)
    throw std::invalid_argument("verify attention baseline is not a precompiled candidate");
  std::vector<VerifyAttentionConfig> result{baseline};
  for (const auto config : operators)
    if (config != baseline) result.push_back(config);
  return result;
}

PrefillAttentionConfig selectPrefillAttentionPolicy(
    std::span<const PrefillAttentionTuningResult> probes, const Policy &policy) {
  if (probes.empty()) return {};
  const auto workloads = prefillAttentionPolicyWorkloads(probes.front().choice.workload.shape);
  return selectPolicy(probes, std::span<const PrefillAttentionWorkload>(workloads),
                        PagedAttention::prefillCandidates(), policy);
}
VerifyAttentionConfig selectVerifyAttentionPolicy(
    std::span<const VerifyAttentionTuningResult> probes, VerifyAttentionConfig baseline,
    const Policy &policy) {
  const auto candidates = verifyAttentionTuningCandidates(baseline);
  if (probes.empty()) return baseline;
  const auto &first = probes.front().choice.workload;
  const auto workloads = verifyAttentionPolicyWorkloads({first.shape, first.lanes});
  return selectPolicy(probes, std::span<const VerifyAttentionWorkload>(workloads),
                        std::span<const VerifyAttentionConfig>(candidates), policy);
}

PrefillAttentionPolicyResult tunePrefillAttentionPolicy(
    metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
    PrefillAttentionPolicy policy, const MeasurementOptions &options,
    const MeasurementStop &pressure, const MeasurementStop &stop) {
  try {
    if (policy.rows != SPLASH_PREFILL_TOKEN_BUDGET)
      throw std::invalid_argument("prefill attention calibration requires the fixed chunk size");
    return tunePolicy<PrefillAttentionPolicyResult>(policy, PrefillAttentionConfig{},
        prefillAttentionPolicyWorkloads(policy.shape), options, pressure, stop,
        [&](auto workload, const auto &probeOptions) {
          return tunePrefillAttention(backend, admit, workload, probeOptions, pressure, stop);
        }, selectPrefillAttentionPolicy);
  } catch (...) {
    return {{policy, {}}, {}, false, std::current_exception()};
  }
}
VerifyAttentionPolicyResult tuneVerifyAttentionPolicy(
    metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
    VerifyAttentionPolicy policy, const MeasurementOptions &options,
    const MeasurementStop &pressure, const MeasurementStop &stop) {
  const auto baseline = VerifyAttentionConfig{};
  try {
    return tunePolicy<VerifyAttentionPolicyResult>(policy, baseline,
        verifyAttentionPolicyWorkloads(policy), options, pressure, stop,
        [&](auto workload, const auto &probeOptions) {
          return tuneVerifyAttention(backend, admit, workload, probeOptions, pressure, stop);
        },
        [&](std::span<const VerifyAttentionTuningResult> probes, const Policy &selection) {
          return selectVerifyAttentionPolicy(probes, baseline, selection);
        });
  } catch (...) {
    return {{policy, baseline}, {}, false, std::current_exception()};
  }
}

uint64_t prefillAttentionTuningFixtureBytes(PrefillAttentionWorkload workload) {
  return fixturePlan(workload).bytes;
}
uint64_t verifyAttentionTuningFixtureBytes(VerifyAttentionWorkload workload) {
  return fixturePlan(workload).bytes;
}
PrefillAttentionTuningResult tunePrefillAttention(
    metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
    PrefillAttentionWorkload workload, const MeasurementOptions &options,
    const MeasurementStop &underPressure, const MeasurementStop &shouldStop) {
  return tune<PrefillAttentionTuningResult>(backend, admit, workload,
                                           PagedAttention::prefillCandidates(),
                                           options, underPressure, shouldStop);
}
VerifyAttentionTuningResult tuneVerifyAttention(
    metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
    VerifyAttentionWorkload workload, const MeasurementOptions &options,
    const MeasurementStop &underPressure, const MeasurementStop &shouldStop) {
  const auto candidates = verifyAttentionTuningCandidates(VerifyAttentionConfig{});
  return tune<VerifyAttentionTuningResult>(backend, admit, workload,
                                          std::span<const VerifyAttentionConfig>(candidates),
                                          options, underPressure, shouldStop);
}

} // namespace splash::ops::tuning
