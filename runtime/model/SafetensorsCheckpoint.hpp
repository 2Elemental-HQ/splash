#pragma once

#include "model/WeightSource.hpp"

#include <filesystem>
#include <memory>
#include <string_view>

namespace splash::model {

// A checkpoint's quantization configuration and safetensors index. Opening
// parses only metadata; the images read tensor data in bounded slices,
// without loading the MLX runtime or allocating tensors. The rest of its
// config.json is the model's, which inspectModelPackage checks.
class SafetensorsCheckpoint final {
public:
  explicit SafetensorsCheckpoint(const std::filesystem::path &directory);
  ~SafetensorsCheckpoint();
  [[nodiscard]] const SourceTensor *find(std::string_view name) const noexcept;
  [[nodiscard]] const SourceTensor &require(std::string_view name) const;
  void requireQuantization(std::string_view projection, uint32_t bits) const;
  void checkUnchanged() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
