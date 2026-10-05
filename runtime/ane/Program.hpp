#pragma once

#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::ane {

// Memory the GPU and the Neural Engine share without a copy: an IOSurface of
// `rows` rows of `width` fp16 or int8 elements, each row padded to 64 bytes,
// and the Metal buffer over it.
struct Surface final {
  enum class Element : uint8_t { Float16, Int8 };

  [[nodiscard]] static Surface create(metal::MetalBackend &backend, uint32_t rows, uint32_t width,
                                      Element element);
  // The bytes of a row, and the whole pages such a surface takes.
  [[nodiscard]] static uint32_t rowBytes(uint32_t width, Element element) noexcept;
  [[nodiscard]] static uint64_t bytes(uint32_t rows, uint32_t width, Element element) noexcept;

  // An element's MIL type.
  [[nodiscard]] static const char *milType(Element element) noexcept;
  // The MIL strides of this surface's leading `rows` rows, and the
  // tensor_buffer type of their leading `width` elements, as a program reads
  // or writes them.
  [[nodiscard]] std::string strides(uint32_t rows) const;
  [[nodiscard]] std::string bufferType(uint32_t rows, uint32_t width) const;

  std::shared_ptr<void> surface;
  metal::MetalBuffer buffer;
  Element element = Element::Float16;
  uint32_t strideBytes = 0;
};

// The weight blob of one fp16 constant tensor, which a program's MIL names as
// BLOBFILE(path = string("@model_path/weights.bin"), offset = kConstantOffset).
inline constexpr uint64_t kConstantOffset = 64;
[[nodiscard]] std::vector<uint8_t> constantBlob(std::span<const _Float16> values);

// A MIL program compiled and loaded on the Neural Engine through the private
// AppleNeuralEngine client, the service behind Core ML. Each of its functions
// is a procedure of the loaded model. The service holds memory for a loaded
// model's intermediate values beside the surfaces, which its procedures share:
// one program of several functions holds what its largest function needs,
// where a program per function would hold their sum. Evaluations run
// asynchronously, ordered against Metal work by a shared event.
class Program final {
public:
  // `weights` is the blob file the program's constants name. The service
  // keeps the compiled program under a key of the source's hash, whose files
  // stay in a directory of the user's temporary directory, so later processes
  // load it without compiling it again.
  Program(std::string_view mil, std::span<const uint8_t> weights);
  ~Program();
  Program(const Program &) = delete;
  Program &operator=(const Program &) = delete;

  // The procedure of the function named `function`.
  [[nodiscard]] uint32_t procedure(std::string_view function) const;
  // A procedure's input names in the order enqueue() takes their surfaces.
  [[nodiscard]] const std::vector<std::string> &inputs(uint32_t procedure) const;

  // Queues one evaluation of `procedure` that starts once `event` reaches
  // `wait` and raises it to `signal` when done. done(false) reports a failed
  // evaluation, after the event is raised anyway so that Metal work waiting
  // on it continues. Throws if the evaluation cannot be queued.
  void enqueue(uint32_t procedure, std::span<const Surface> inputs, const Surface &output,
               const metal::SharedEvent &event, uint64_t wait, uint64_t signal, std::function<void(bool)> done);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// A small value kept beside the compiled programs under `key`, such as the
// share a calibration on this Mac chose last, whose programs it compiled; none
// until remembered, or once macOS clears the temporary directory.
[[nodiscard]] std::optional<uint32_t> recall(std::string_view key);
void remember(std::string_view key, uint32_t value) noexcept;

} // namespace splash::ane
