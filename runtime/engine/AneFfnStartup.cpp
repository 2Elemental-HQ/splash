#include "engine/AneFfnStartup.hpp"

#include "AwakeClock.hpp"
#include "StderrLine.hpp"
#include "ane/Program.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::engine {
namespace {

using Kind = AneFfnOutcome::Kind;

// `tokens` with its thousands grouped, as the server prints a context.
std::string grouped(uint32_t tokens) {
  std::string digits = std::to_string(tokens);
  for (size_t at = digits.size(); at > 3; at -= 3) digits.insert(at - 3, ",");
  return digits;
}

uint32_t contextOf(const EngineMemoryPlanResult &result) {
  return result.plan ? result.plan->maximumContextTokens() : 0;
}

// Logs `outcome` of a start of `model` as `setting` allows: a model without
// dense FFN layers and one --no-ane leaves alone without its layers said, and
// a refusal of a context the GPU alone does not hold either, which fails the
// start on its own, log nothing.
void logOutcome(const AneFfnOutcome &outcome, const AneFfnModel &model, const AneFfnSetting &setting,
                uint32_t requestedContextTokens) {
  switch (outcome.kind) {
  case Kind::Off:
    if (model.dense && model.unsupported.empty()) logLine("The GPU runs the prefill FFN alone, ", outcome.reason, ".");
    return;
  case Kind::Unsupported:
    if (model.dense) logLine("The GPU runs the prefill FFN alone: ", outcome.reason, ".");
    return;
  case Kind::NoGain:
    logLine("The GPU runs the prefill FFN alone: ", outcome.reason, ".");
    return;
  case Kind::Refused:
    if (requestedContextTokens > outcome.contextWithout) return;
    if (outcome.contextWith)
      logWarning("Neural Engine FFN split off: ", outcome.reason, ", and ",
                 setting.given ? "the given share" : "any split", " leaves at most ", grouped(outcome.contextWith),
                 " tokens; pass --max-context ", outcome.contextWith,
                 " or less to run it, or --no-ane, which also silences this line.");
    else
      logWarning("Neural Engine FFN split off: ", outcome.reason, "; --no-ane silences this line.");
    return;
  case Kind::Unavailable:
    logWarning("Neural Engine FFN split unavailable (", outcome.reason,
               "); the GPU runs the prefill FFN alone. --no-ane silences this line.");
    return;
  case Kind::Split:
    logLine("Neural Engine FFN split ", outcome.reason, ".");
    return;
  }
}

AneFfnStart decide(const AneFfnModel &model, const AneFfnSetting &setting, uint32_t requestedContextTokens,
                   const std::function<EngineMemoryPlanResult(uint64_t)> &planMemory,
                   const std::function<bool()> &cancelled) {
  const uint32_t without = contextOf(planMemory(0));
  const auto gpuAlone = [&](Kind kind, std::string reason, uint32_t with = 0) {
    return AneFfnStart{{}, {}, {kind, std::move(reason), with, without}};
  };
  if (!model.dense) return gpuAlone(Kind::Unsupported, "the target has no dense FFN layers");
  if (!setting.enabled) return gpuAlone(Kind::Off, "as given");
  if (!model.unsupported.empty()) return gpuAlone(Kind::Unsupported, model.unsupported);
  const std::optional<uint32_t> given =
      setting.given ? std::optional(ops::ane_ffn::nearestUnits(setting.given->share, model.units)) : std::nullopt;
  if (setting.given &&
      (setting.given->minimumRows < ops::AneFfn::kMinimumRows || setting.given->minimumRows > ops::AneFfn::kMaximumRows))
    throw std::invalid_argument("ANE FFN split has no function of " + std::to_string(setting.given->minimumRows) +
                                " rows");

  // Feasibility. The split's memory comes out of the KV cache, more of it
  // with each unit the ANE takes: the plan must still hold the context asked
  // for, or with none asked any.
  const auto planWith = [&](uint32_t aneUnits) { return planMemory(model.plannedBytes(aneUnits)); };
  const uint32_t needed = std::max<uint32_t>(requestedContextTokens, 1);
  const auto holds = [&](uint32_t aneUnits) { return contextOf(planWith(aneUnits)) >= needed; };
  const uint32_t least = given.value_or(1);
  if (!holds(least)) {
    const uint32_t most = contextOf(planWith(least));
    if (requestedContextTokens > without)
      return gpuAlone(Kind::Refused,
                      "the GPU alone does not hold --max-context " + std::to_string(requestedContextTokens) + " either",
                      most);
    return gpuAlone(Kind::Refused,
                    requestedContextTokens
                        ? "--max-context " + std::to_string(requestedContextTokens) + " cannot be held with it"
                        : "it leaves no memory for context",
                    most);
  }
  // The most units whose plan holds it.
  uint32_t maxAneUnits = least;
  for (uint32_t high = given ? least : model.units - 1; maxAneUnits < high;) {
    const uint32_t middle = maxAneUnits + (high - maxAneUnits + 1) / 2;
    if (holds(middle))
      maxAneUnits = middle;
    else
      high = middle - 1;
  }

  logLine("Setting up the Neural Engine FFN split (splash serve --no-ane keeps the FFN on the GPU).");
  const auto started = AwakeClock::now();
  const auto seconds = [&] {
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << std::chrono::duration<double>(AwakeClock::now() - started).count()
         << " s";
    return text.str();
  };
  const std::string layer = " ms per " + std::to_string(ops::AneFfn::kMaximumRows) + "-row FFN layer";
  try {
    uint32_t aneUnits = least;
    if (!given) {
      const ops::ane_ffn::Timings timings = model.time();
      const std::optional<uint32_t> chosen =
          ops::ane_ffn::choose(ops::ane_ffn::fit(timings.low, timings.high), timings.gpuAlone, model.units,
                               maxAneUnits, model.recall());
      if (!chosen) {
        std::ostringstream reason;
        reason << "no Neural Engine split beats its " << std::fixed << std::setprecision(1) << timings.gpuAlone
               << layer << " by enough (calibrated in " << seconds() << ")";
        return gpuAlone(Kind::NoGain, reason.str());
      }
      aneUnits = *chosen;
    }
    AneFfnPrepared prepared = model.prepare(aneUnits, !given);
    const ops::ane_ffn::ChunkTimings &chunks = prepared.chunks;
    const std::optional<uint32_t> minimumRows =
        given ? std::optional(setting.given->minimumRows) : ops::ane_ffn::minimumRows(chunks);
    std::ostringstream reason;
    reason << std::fixed << std::setprecision(1);
    if (!minimumRows) {
      reason << "no chunk of the Neural Engine split beats it by enough, " << chunks.functions.front().split << layer
             << " against " << chunks.gpu[1].milliseconds << " (calibrated in " << seconds() << ")";
      return gpuAlone(Kind::NoGain, reason.str());
    }
    if (prepared.split) {
      prepared.split->setMinimumRows(*minimumRows);
      if (!given) prepared.split->setBreaker(ops::ane_ffn::Breaker(chunks));
    }
    EngineMemoryPlanResult plan = planWith(aneUnits);
    if (!plan.plan) throw std::logic_error("the memory plan does not hold the split it chose");
    if (!given) model.remember(aneUnits);
    const uint32_t with = contextOf(plan);
    reason << std::setprecision(2) << "at " << (given ? "the given share " : "share ")
           << static_cast<double>(aneUnits) / model.units << std::setprecision(1) << " for chunks of " << *minimumRows
           << " rows or more";
    if (!given)
      reason << ": " << chunks.functions.front().split << layer << " against " << chunks.gpu[1].milliseconds
             << " on the GPU alone";
    reason << ", " << 100.0 * prepared.error << "% from the GPU alone on the Neural Engine's part (set up in "
           << seconds() << ")";
    if (!requestedContextTokens && with != without)
      reason << "; context " << grouped(with) << " tokens (" << grouped(without) << " with --no-ane)";
    return {std::move(prepared.split), std::move(plan.plan), {Kind::Split, reason.str(), with, without}};
  } catch (const ane::Interrupted &) {
    throw;
  } catch (const std::exception &error) {
    if ((cancelled && cancelled()) || !model.healthy()) throw;
    return gpuAlone(Kind::Unavailable, error.what());
  }
}

} // namespace

std::string_view aneFfnOutcomeName(AneFfnOutcome::Kind kind) noexcept {
  switch (kind) {
  case Kind::Off:
    return "off";
  case Kind::Unsupported:
    return "unsupported";
  case Kind::Refused:
    return "refused";
  case Kind::NoGain:
    return "no_gain";
  case Kind::Unavailable:
    return "unavailable";
  case Kind::Split:
    return "split";
  }
  return "unknown";
}

AneFfnSetting AneFfnSetting::fromGiven(std::optional<double> share, std::optional<uint32_t> minimumRows) {
  if (!share) {
    if (minimumRows) throw std::invalid_argument("ANE FFN least chunk rows given without a share");
    return {};
  }
  if (*share == 0.0) return {.enabled = false};
  return {.given = Given{*share, minimumRows.value_or(ops::AneFfn::kMinimumRows)}};
}

AneFfnStart startAneFfn(const AneFfnModel &model, const AneFfnSetting &setting, uint32_t requestedContextTokens,
                        const std::function<EngineMemoryPlanResult(uint64_t aneFfnBytes)> &planMemory,
                        const std::function<bool()> &cancelled) {
  AneFfnStart start = decide(model, setting, requestedContextTokens, planMemory, cancelled);
  logOutcome(start.outcome, model, setting, requestedContextTokens);
  return start;
}

} // namespace splash::engine
