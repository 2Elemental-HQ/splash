#pragma once

#include "ops/DraftAttention.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"

#include <span>
#include <vector>

namespace splash::ops {

struct OperatorChoices final {
  std::vector<LinearChoice> linear;

  [[nodiscard]] bool empty() const noexcept { return linear.empty(); }
};

// One runtime owns this object; production models borrow it. Installation is
// a startup-only operation, before concurrent encoding. Arena sizing precedes
// confirmation; those trials may only toggle the preallocated baseline/selected
// pair. After Ready the owner and all borrowed plans remain immutable.
class ExecutionPlans final {
public:
  explicit ExecutionPlans(const DeviceCapabilities &device);
  [[nodiscard]] const Linear &linear() const noexcept { return linear_; }
  // Validate every table before replacing any installed choice. Missing keys
  // always use the operator's shipped baseline; an empty install resets all.
  void install(const OperatorChoices &choices);

  [[nodiscard]] PrefillAttentionPlan prefillAttention(
      uint32_t rows, uint32_t queryHeads, kv::Layout layout,
      uint32_t historyTokens) const;
  [[nodiscard]] VerifyAttentionPlan verifyAttention(
      uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
      std::span<const uint32_t> historyTokens) const;
  [[nodiscard]] DraftAttentionPlan draftAttention(
      DraftAttentionShape shape, uint32_t lanes) const;
  [[nodiscard]] MoePlan moePrefill(MoeShape shape, uint32_t rows) const;
  [[nodiscard]] MoePlan moeDecode(MoeShape shape, uint32_t lanes) const;

  // Bounds include baseline and every matching installed key, not just the
  // currently requested row count. Packed decode arenas use a per-lane stride
  // of max_B ceil(requiredBytes(B)/B), independently for each scratch field.
  [[nodiscard]] AttentionWorkspace prefillAttentionWorkspace(
      uint32_t maximumRows, uint32_t queryHeads, kv::Layout layout) const;
  [[nodiscard]] AttentionWorkspace verifyAttentionWorkspacePerLane(
      uint32_t queryHeads, kv::Layout layout) const;
  [[nodiscard]] DraftAttentionWorkspace draftAttentionWorkspacePerLane(
      DraftAttentionShape shape) const;
  [[nodiscard]] MoeWorkspace moePrefillWorkspace(
      MoeShape shape, uint32_t maximumRows) const;
  [[nodiscard]] MoeWorkspace moeDecodeWorkspacePerLane(MoeShape shape) const;
  // This scratch is one whole-command buffer, not a per-lane arena field.
  [[nodiscard]] uint64_t gateUpWorkspace(ProjectionShape shape) const;

private:
  // The device's configuration of a MoE plan of `rows` rows in `phase`.
  [[nodiscard]] MoeConfig moeConfig(MoeShape shape, uint32_t rows, MoePhase phase) const;

  Linear linear_;
  Linear baselineLinear_;
  uint32_t moeRouteWideRows_ = kMoeRouteWideRows;
  MoeExpertSimdgroups moeDecodeSimdgroups_ = MoeExpertSimdgroups::Eight;
  uint32_t appleGpuFamily_ = 0;
  OperatorChoices choices_;
};

} // namespace splash::ops
