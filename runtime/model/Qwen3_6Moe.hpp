#pragma once

#include "DFlashDraft.hpp"
#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <string_view>

namespace splash::model {

struct Qwen3_6MoeLayout final : QwenHybridLayout<8> {
  static constexpr std::string_view layerMagic = "MDFM0001";
  static constexpr std::string_view headMagic = "MDFM0002";

  constexpr Qwen3_6MoeLayout()
      : QwenHybridLayout{{.layers = 40,
                          .hiddenSize = 2048,
                          .vocabularySize = 248320,
                          .packedGdnWidth = 12544,
                          .packedFullWidth = 9216,
                          .convolutionDimension = 8192,
                          .gdnKeyHeads = 16,
                          .gdnValueHeads = 32,
                          .gdnHeadDimension = 128,
                          .attentionWidth = 4096,
                          .attentionQueryHeads = 16,
                          .attentionKvHeads = 2,
                          .attentionHeadDimension = 256,
                          .rotaryPairs = 32,
                          .rotaryTheta = 10'000'000.0F,
                          .fullAttentionPeriod = 4,
                          .maskToken = 248077,
                          .stopTokens = {248044, 248046},
                          .ffnKind = QwenFfnKind::SparseMoe,
                          .experts = 256,
                          .expertsPerToken = 8,
                          .expertIntermediateSize = 512},
                         /* hiddenCaptureLayers */ {1, 6, 11, 16, 22, 27, 32, 37}} {}
  bool operator==(const Qwen3_6MoeLayout &) const = default;
};

// The DFlash2 draft of Qwen3.6-35B-A3B.
inline constexpr DFlashDraftLayout kQwen3_6MoeDraftLayout{.layers = 6,
                                                          .hiddenSize = 2048,
                                                          .vocabularySize = 248320,
                                                          .dynamicSize = 512,
                                                          .qkvSize = 6144,
                                                          .attentionSize = 4096,
                                                          .intermediateSize = 6144,
                                                          .attentionHeadDimension = 128,
                                                          .rotaryTheta = 10'000'000.0F,
                                                          .targetHiddenSize = 16384,
                                                          .selectorRank = 256,
                                                          .kvHeads = 8};

struct Qwen3_6MoeLayerWeights final {
  ops::NormWeights inputNorm;
  QwenMixerWeights mixer;
  ops::NormWeights postAttentionNorm;
  ops::MoeWeights ffn;
};

using Qwen3_6MoeWeights = QwenTargetWeights<Qwen3_6MoeLayout, Qwen3_6MoeLayerWeights>;

[[nodiscard]] Qwen3_6MoeWeights
loadQwen3_6MoeWeights(metal::MetalBackend &backend, Qwen3_6MoeLayout layout,
                      const QwenTargetFiles<Qwen3_6MoeLayout> &files);

} // namespace splash::model
