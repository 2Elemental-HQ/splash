// The prefill FFN's Neural Engine split (ops/AneFfn.cpp, kernels/prefill/ane_ffn.metal):
// - inputs: ane_ffn_rotate and ane_ffn_pack turn bf16 rows into the ANE's channel-major int8 segments at their stride
//   and the rows' scales, as a CPU rotation of the rows does;
// - weights: ane_ffn_row_scale and ane_ffn_weights turn the rows of affine Q4 and of every GGUF format into int8 rows
//   at their stride and their scales at theirs, rotated in either block, as a CPU rotation of their dequantized values
//   does;
// - join: ane_ffn_join adds the ANE's channel-major partial rows to the chunk's output rows and to no others;
// - the split: which layers, shares and chunks it takes; for three layers of affine Q4 and of mixed GGUF formats,
//   Metal allocates what plannedBytes plans, and chunks of the fewest rows, of a count between two programs and of the
//   most rows compute what the GPU computes alone within int8's error; layers it is given out of order are refused,
//   and it keeps what it needs of the layers it was built from;
// - faults: each fault of the ANE's evaluations (ane/ProgramInstrumentation.hpp) at each layer of chunks of three
//   sizes stops the split without failing the Metal command, and the GPU alone then computes what it computes on its
//   own; an evaluation that signals the shared event below its value leaves the value as it is;
// - the program (runtime/ane/Program.mm): an invalid MIL, a function it lacks and a binding short of an input fail as
//   std::exceptions; a limit that runs out and an interrupted wait end the wait, after which a program still comes up;
//   unload() and load() round-trip, and load() does not compile; the cache writes a file of other bytes again and
//   leaves no partial file when it cannot write one.
// The kernels round where the CPU may not (fast-math rsqrt and division, fused multiply-adds), so an int8 value may
// differ by one and a scale by its last bit. The split runs on the Neural Engine, which every Apple Silicon Mac has.
// Metal's validation layer wraps the shared event that orders the split's GPU and ANE work in one the ANE cannot
// share, so `kernels` runs the first three checks, which test-engine-metal validates, and `split`, `faults` and
// `program` the others. The binary links the instrumented Program, whose faults only `faults` arms.
#include "AffineQ4Fixture.hpp"
#include "AneProgramFixture.hpp"
#include "AwakeClock.hpp"
#include "Checked.hpp"
#include "GgufFormatReference.hpp"
#include "TestBuffers.hpp"
#include "TestFiles.hpp"
#include "ane/ProgramInstrumentation.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/AneFfn.h"
#include "metal/abi/QuantFormat.h"
#include "ops/AneFfn.hpp"
#include "ops/Linear.hpp"
#include "tuning/LinearNumerics.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// What the program checks send the private AppleNeuralEngine client beside
// ane::Program.
@protocol AneTestModel
+ (id)modelAtURL:(NSURL *)url key:(NSString *)key;
@end
@protocol AneTestClient
+ (id)sharedConnection;
- (void)purgeCompiledModel:(id)model;
@end

using namespace splash;
using namespace splash::ops;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::tuning::bf16ToFloat;
using splash::ops::tuning::floatToBf16;

namespace {

// The smallest hidden size both the ANE programs (segments of 2560 channels) and ane_ffn_rotate (groups of 1024)
// take, and an intermediate size whose ANE share at kShare spans two segments of down's inputs.
constexpr uint32_t kHidden = 5120, kSegment = 2560, kIntermediate = 6144, kLayers = 3;
constexpr double kShare = 0.5;
// The split's output against the GPU's alone, as calibration bounds it (AneFfn.cpp).
constexpr double kMaximumError = 0.05;
// What a kernel must leave as it was.
constexpr uint8_t kUntouched = 0x5A;
constexpr uint16_t kUntouchedHalf = 0x5A5A;

std::mt19937 rng(20261003);
int failures = 0;
int sectionFailures = 0;

// Names one failure; the first few of a section are printed.
void fail(const std::string &what) {
  constexpr int kShown = 8;
  if (sectionFailures++ < kShown) std::cout << "  FAIL " << what << '\n';
  ++failures;
}
void section(const std::string &what) {
  std::cout << what << (sectionFailures ? " FAIL (" + std::to_string(sectionFailures) + " failures)" : " ok") << '\n';
  sectionFailures = 0;
}

MetalBuffer filled(MetalBackend &backend, uint64_t bytes, uint8_t value) {
  MetalBuffer buffer = test::sharedBuffer(backend, bytes);
  std::memset(buffer.contents(), value, bytes);
  return buffer;
}
template <class T>
MetalBuffer upload(MetalBackend &backend, const std::vector<T> &values) {
  MetalBuffer buffer = test::sharedBuffer(backend, values.size() * sizeof(T));
  std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(T));
  return buffer;
}
template <class T>
const T *contents(const MetalBuffer &buffer) {
  return static_cast<const T *>(buffer.contents());
}
void run(MetalBackend &backend, const CommandGraph &graph) {
  static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
}

float toHalf(float value) { return float(_Float16(value)); }
float fromHalf(uint16_t bits) { return float(std::bit_cast<_Float16>(bits)); }
int8_t toInt8(float value) {
  return int8_t(std::clamp(std::rint(value), -ANE_FFN_INT8_PEAK, ANE_FFN_INT8_PEAK));
}
bool near(int8_t got, int8_t expected) { return std::abs(int(got) - int(expected)) <= 1; }
bool near(float got, float expected) { return std::fabs(got - expected) <= 0x1p-9f * std::fabs(expected); }

// v -> sign (H v) / sqrt(n) on each block of n values, H the Sylvester Hadamard matrix: the kernels' fp32 butterflies
// in their order.
void rotate(std::span<float> values, uint32_t n, const std::vector<float> &sign) {
  const float norm = 1.0f / std::sqrt(float(n));
  for (size_t begin = 0; begin < values.size(); begin += n) {
    float *v = values.data() + begin;
    for (uint32_t span = 1; span < n; span <<= 1)
      for (uint32_t i = 0; i < n; ++i)
        if (!(i & span)) {
          const float a = v[i], b = v[i + span];
          v[i] = a + b;
          v[i + span] = a - b;
        }
    for (uint32_t i = 0; i < n; ++i) v[i] *= sign[i] * norm;
  }
}
float peak(std::span<const float> values) {
  float result = 0.0f;
  for (const float value : values) result = std::max(result, std::fabs(value));
  return result;
}

// ---------------------------------------------------------------- inputs
// Rows of normal values, one in 997 forty times larger, as the ANE's two int8 input segments, each channel's rows at
// a stride past them.
void inputs(MetalBackend &backend, const MetalBuffer &signs, const std::vector<float> &sign) {
  constexpr uint32_t kRows = 64, kStride = 96;
  std::normal_distribution<float> normal;
  std::vector<uint16_t> x(uint64_t{kRows} * kHidden);
  for (uint64_t i = 0; i < x.size(); ++i) x[i] = floatToBf16(normal(rng) * (i % 997 ? 1.0f : 40.0f));
  const MetalBuffer rotated = test::sharedBuffer(backend, x.size() * sizeof(uint16_t));
  const MetalBuffer scales = filled(backend, kRows * sizeof(uint16_t), kUntouched);
  CommandGraph graph;
  graph.add("ane_ffn_rotate", {upload(backend, x), signs, rotated, scales}, AneFfnRotateParams{kHidden},
            {kRows, 1, 1}, {ANE_FFN_ROTATE_THREADS, 1, 1});
  std::vector<MetalBuffer> packed;
  for (uint32_t segment = 0; segment < kHidden / kSegment; ++segment) {
    packed.push_back(filled(backend, uint64_t{kSegment} * kStride, kUntouched));
    graph.add("ane_ffn_pack", {rotated, packed.back()}, AneFfnPackParams{kHidden, segment * kSegment, kStride},
              {kRows / ANE_FFN_TILE, kSegment / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  }
  run(backend, graph);

  for (uint32_t row = 0; row < kRows; ++row) {
    std::vector<float> v(kHidden);
    for (uint32_t channel = 0; channel < kHidden; ++channel) v[channel] = bf16ToFloat(x[uint64_t{row} * kHidden + channel]);
    rotate(v, ANE_FFN_INPUT_BLOCK, sign);
    const float scale = std::max(peak(v) / ANE_FFN_INT8_PEAK, 1e-8f), inverse = 1.0f / scale;
    if (!near(fromHalf(contents<uint16_t>(scales)[row]), toHalf(scale * ANE_FFN_INT8_UNIT)))
      fail("row " + std::to_string(row) + ": scale " + std::to_string(fromHalf(contents<uint16_t>(scales)[row])) +
           ", not " + std::to_string(toHalf(scale * ANE_FFN_INT8_UNIT)));
    for (uint32_t channel = 0; channel < kHidden; ++channel) {
      const int8_t got = contents<int8_t>(packed[channel / kSegment])[uint64_t{channel % kSegment} * kStride + row];
      const int8_t expected = toInt8(toHalf(v[channel] * inverse));
      if (!near(got, expected))
        fail("row " + std::to_string(row) + " channel " + std::to_string(channel) + ": " + std::to_string(got) +
             ", not " + std::to_string(expected));
    }
  }
  for (const MetalBuffer &segment : packed)
    for (uint32_t channel = 0; channel < kSegment; ++channel)
      for (uint32_t byte = kRows; byte < kStride; ++byte)
        if (contents<uint8_t>(segment)[uint64_t{channel} * kStride + byte] != kUntouched)
          fail("channel " + std::to_string(channel) + ": packed past the rows");
  section("inputs: 64 rows of 5120 bf16 values as two int8 segments and the rows' scales");
}

// ---------------------------------------------------------------- weights
constexpr uint32_t kWeightRows = 512, kWeightInputs = 2048;
// A [kWeightRows, kWeightInputs] projection's planes as the weight kernels bind them, the parameters and kernel name
// suffix of its layout, and its values.
struct Source {
  std::string name;
  std::array<MetalBuffer, 3> planes;
  uint32_t groups = 0, format = 0;
  std::string suffix;
  std::vector<float> values;
};

Source affineSource(MetalBackend &backend) {
  const Projection projection = test::deterministicQ4Projection(backend, {kWeightRows, kWeightInputs}, 11);
  const AffineWeights &weights = projection.affine();
  Source source{"affine Q4", {weights.weights, weights.scales, weights.biases}, kWeightInputs / 64, 0, "", {}};
  const auto *nibbles = contents<uint8_t>(weights.weights);
  const auto *scales = contents<uint16_t>(weights.scales), *biases = contents<uint16_t>(weights.biases);
  source.values.resize(uint64_t{kWeightRows} * kWeightInputs);
  for (uint32_t row = 0; row < kWeightRows; ++row)
    for (uint32_t input = 0; input < kWeightInputs; ++input) {
      const uint64_t unit = quant_tile_index(row, input / 64, kWeightInputs / 64);
      const uint32_t code = (nibbles[unit * 32 + (input & 63) / 2] >> (input & 1 ? 4 : 0)) & 15;
      source.values[uint64_t{row} * kWeightInputs + input] =
          float(code) * bf16ToFloat(scales[unit]) + bf16ToFloat(biases[unit]);
    }
  return source;
}

Source ggufSource(MetalBackend &backend, gguf_reference::Fmt format) {
  using namespace gguf_reference;
  const std::vector<uint8_t> native = makeNative(format, kWeightRows, kWeightInputs, rng);
  const Packed packed = repack(format, native, kWeightRows, kWeightInputs, nullptr);
  const MetalBuffer meta = upload(backend, packed.meta);
  // A format without plane1 binds meta in its place (QuantizedSegment::plane1Slot).
  Source source{fmtName(format),
                {upload(backend, packed.w0), kQuantFormats[format].plane1_bytes ? upload(backend, packed.w1) : meta, meta},
                kWeightInputs / 32,
                uint32_t(format),
                "_gguf",
                {}};
  source.values.resize(uint64_t{kWeightRows} * kWeightInputs);
  for (uint32_t row = 0; row < kWeightRows; ++row)
    rowValues(format, native.data() + uint64_t{row} * rowBytes(format, kWeightInputs), kWeightInputs,
              source.values.data() + uint64_t{row} * kWeightInputs);
  return source;
}

// Rows [row, kWeightRows) of `source` under their scales over inputs [scaled, kWeightInputs), and their int8 values
// over inputs [input, kWeightInputs), rotated in blocks of `block` from `scaled`: as the split stages the rows of
// gate and up the ANE takes over a segment of the hidden inputs (block 128), or the rows of down over a segment of
// the ANE's inputs (block 512). Each row's values lie a stride apart, and its scales' copies another.
void weights(MetalBackend &backend, const MetalBuffer &signs, const std::vector<float> &sign, const Source &source,
             uint32_t block, uint32_t row, uint32_t scaled, uint32_t input) {
  const uint32_t rows = kWeightRows - row, width = kWeightInputs - input, stride = width + 64, scaleStride = 3;
  const std::string variant = block == ANE_FFN_INPUT_BLOCK ? "_inputs" : "_intermediate";
  const MetalBuffer rowScales = test::sharedBuffer(backend, rows * sizeof(uint16_t));
  const MetalBuffer output = filled(backend, uint64_t{rows} * stride, kUntouched);
  const MetalBuffer scales = filled(backend, uint64_t{rows} * scaleStride * sizeof(uint16_t), kUntouched);
  const auto &[a, b, c] = source.planes;
  CommandGraph graph;
  graph.add("ane_ffn_row_scale" + source.suffix + variant, {a, b, c, rowScales, signs},
            AneFfnWeightParams{source.groups, row, scaled, kWeightInputs - scaled, 0, 0, source.format},
            {rows / ANE_FFN_WEIGHT_ROWS, 1, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
  graph.add("ane_ffn_weights" + source.suffix + variant, {a, b, c, rowScales, output, scales, signs},
            AneFfnWeightParams{source.groups, row, input, width, stride, scaleStride, source.format},
            {rows / ANE_FFN_WEIGHT_ROWS, width / block, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
  run(backend, graph);

  const std::string label = source.name + variant;
  for (uint32_t r = 0; r < rows; ++r) {
    const auto first = source.values.begin() + uint64_t{row + r} * kWeightInputs;
    std::vector<float> v(first + scaled, first + kWeightInputs);
    rotate(v, block, sign);
    const uint16_t bits = contents<uint16_t>(rowScales)[r];
    const float rowScale = fromHalf(bits);
    const float expected = toHalf(std::max(peak(v), 1e-8f) / ANE_FFN_INT8_PEAK * ANE_FFN_INT8_UNIT);
    if (!near(rowScale, expected))
      fail(label + " row " + std::to_string(r) + ": scale " + std::to_string(rowScale) + ", not " +
           std::to_string(expected));
    const uint16_t *copy = contents<uint16_t>(scales) + uint64_t{r} * scaleStride;
    if (copy[0] != bits || copy[1] != kUntouchedHalf || copy[2] != kUntouchedHalf)
      fail(label + " row " + std::to_string(r) + ": scale copy");
    const float inverse = ANE_FFN_INT8_UNIT / rowScale;
    const int8_t *values = contents<int8_t>(output) + uint64_t{r} * stride;
    for (uint32_t i = 0; i < width; ++i) {
      const int8_t value = toInt8(v[input - scaled + i] * inverse);
      if (!near(values[i], value))
        fail(label + " row " + std::to_string(r) + " input " + std::to_string(input + i) + ": " +
             std::to_string(values[i]) + ", not " + std::to_string(value));
    }
    for (uint32_t byte = width; byte < stride; ++byte)
      if (uint8_t(values[byte]) != kUntouched) fail(label + " row " + std::to_string(r) + ": written past its values");
  }
}

void weights(MetalBackend &backend, const MetalBuffer &signs, const std::vector<float> &sign) {
  std::vector<Source> sources{affineSource(backend)};
  for (int format = 0; format < gguf_reference::FMT_COUNT; ++format)
    sources.push_back(ggufSource(backend, gguf_reference::Fmt(format)));
  for (const Source &source : sources) {
    weights(backend, signs, sign, source, ANE_FFN_INPUT_BLOCK, 256, 0, 512);
    weights(backend, signs, sign, source, ANE_FFN_INTERMEDIATE_BLOCK, 0, 512, 1024);
  }
  section("weights: affine Q4 and " + std::to_string(gguf_reference::FMT_COUNT) +
          " GGUF formats as int8 rows and scales, rotated in blocks of 128 and 512");
}

// ---------------------------------------------------------------- join
void join(MetalBackend &backend) {
  constexpr uint32_t kRows = 70, kTiles = (kRows + ANE_FFN_TILE - 1) / ANE_FFN_TILE, kStride = 128;
  std::normal_distribution<float> normal;
  std::vector<uint16_t> output(uint64_t{kTiles} * ANE_FFN_TILE * kHidden), partial(uint64_t{kHidden} * kStride);
  for (uint16_t &value : output) value = floatToBf16(normal(rng) * 64.0f);
  for (uint16_t &value : partial) value = std::bit_cast<uint16_t>(_Float16(normal(rng)));
  const MetalBuffer joined = upload(backend, output);
  CommandGraph graph;
  graph.add("ane_ffn_join", {joined, upload(backend, partial)}, AneFfnJoinParams{kHidden, kStride, kRows},
            {kTiles, kHidden / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  run(backend, graph);
  for (uint32_t row = 0; row < kTiles * ANE_FFN_TILE; ++row)
    for (uint32_t channel = 0; channel < kHidden; ++channel) {
      const uint64_t index = uint64_t{row} * kHidden + channel;
      const uint16_t expected =
          row < kRows ? floatToBf16(bf16ToFloat(output[index]) + fromHalf(partial[uint64_t{channel} * kStride + row]))
                      : output[index];
      if (contents<uint16_t>(joined)[index] != expected)
        fail("row " + std::to_string(row) + " channel " + std::to_string(channel) +
             (row < kRows ? ": not the sum" : ": past the chunk, changed"));
    }
  section("join: the partial rows of a 70-row chunk added to its output rows and to no others");
}

// ---------------------------------------------------------------- the split
// Three FFN layers: gate and up [kIntermediate, kHidden], down [kHidden, kIntermediate].
struct Model {
  std::string name;
  std::vector<Projection> projections;
  std::vector<SwiGluProjections> layers;
};
// The layers over the model's projections, which stay where they are.
Model withLayers(Model model) {
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    model.layers.push_back(
        {&model.projections[3 * layer], &model.projections[3 * layer + 1], &model.projections[3 * layer + 2]});
  return model;
}
constexpr std::array<LinearMatrix, 3> kMatrices{
    {{kIntermediate, kHidden}, {kIntermediate, kHidden}, {kHidden, kIntermediate}}};

Model affineModel(MetalBackend &backend) {
  Model model{"affine Q4", {}, {}};
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    for (const LinearMatrix matrix : kMatrices)
      model.projections.push_back(
          test::deterministicQ4Projection(backend, matrix, uint32_t(model.projections.size()) * 7919));
  return withLayers(std::move(model));
}

// Layers of mixed formats, gate's, up's and down's of each, with scales a 32nd of the fixture's: weights of a model's
// magnitude, whose FFN values the ANE's fp16 holds as it holds a model's. The fixture's random sub-block scales
// spread a row's values over a range no model's weights span, more so in other formats (Q6_K's split error is 4.8%
// on them, against 0.6% for Q5_K); `weights` checks every format's int8 rows.
Model ggufModel(MetalBackend &backend) {
  using namespace gguf_reference;
  constexpr std::array<std::array<Fmt, 3>, kLayers> kFormats{
      {{Q4K, Q4K, Q5K}, {Q5K, Q5K, Q4K}, {Q4K, Q5K, Q80}}};
  Model model{"GGUF", {}, {}};
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    for (uint32_t index = 0; index < 3; ++index) {
      const Fmt format = kFormats[layer][index];
      const auto [outputs, inputs] = kMatrices[index];
      std::uniform_real_distribution<float> range = scaleRange(format);
      const std::vector<uint8_t> native =
          makeNative(format, outputs, inputs, rng, [&] { return f2h(range(rng) / 32); });
      const Packed packed = repack(format, native, outputs, inputs, nullptr);
      model.projections.emplace_back(
          outputs, inputs,
          BlockWeights{{QuantizedSegment::planes(
              format, outputs, inputs, upload(backend, packed.w0),
              kQuantFormats[format].plane1_bytes ? upload(backend, packed.w1) : MetalBuffer{},
              upload(backend, packed.meta))}});
    }
  return withLayers(std::move(model));
}

// Which layers and shares the split takes.
void takes(MetalBackend &backend, const Model &affine, const Model &gguf) {
  if (!AneFfn::supports(affine.layers) || !AneFfn::supports(gguf.layers)) fail("the split does not take the models");
  if (AneFfn::supports({})) fail("the split takes no layers");
  // A hidden size the programs' segments do not take, and layers of two shapes.
  const Projection gate = test::deterministicQ4Projection(backend, {kIntermediate, 4096}, 1);
  const Projection down = test::deterministicQ4Projection(backend, {4096, kIntermediate}, 2);
  const std::vector<SwiGluProjections> narrow{{&gate, &gate, &down}}, mixed{affine.layers[0], narrow[0]};
  if (AneFfn::supports(narrow)) fail("the split takes a hidden size of 4096");
  if (AneFfn::supports(mixed)) fail("the split takes layers of two shapes");
  // Shares that leave the GPU or the ANE no channels.
  for (const double share : {0.0, 1.0})
    test::rejects([&] { static_cast<void>(AneFfn::plannedBytes(affine.layers, share)); }, "must lie in (0, 1)",
                  "share " + std::to_string(share) + " was planned");
  for (const double share : {0.02, 0.98})
    test::rejects([&] { static_cast<void>(AneFfn::plannedBytes(affine.layers, share)); }, "no channels",
                  "share " + std::to_string(share) + " leaves the GPU or the ANE no channels, yet plans");
  section("takes: models of one splittable shape, and shares that leave each part channels");
}

// The chunk's buffers of `rows` rows at most: the normalized rows and the FFN's scratch, and the hidden rows the
// layers' residuals and outputs alternate between.
struct Chunk {
  PrefillFfnBuffers ffn;
  std::array<MetalBuffer, 2> hidden;
};
Chunk chunk(MetalBackend &backend, uint32_t rows) {
  const auto bf16Rows = [&](uint32_t width) { return test::sharedBuffer(backend, uint64_t{rows} * width * 2); };
  const auto sumRows = [&](uint32_t width) { return test::sharedBuffer(backend, uint64_t{rows} * (width / 64) * 4); };
  // Chunks of at least kMinimumRows rows take no split scratch (Linear::prefillScratchSize).
  return {{bf16Rows(kHidden), sumRows(kHidden), bf16Rows(kIntermediate), bf16Rows(kIntermediate),
           sumRows(kIntermediate), {}},
          {bf16Rows(kHidden), bf16Rows(kHidden)}};
}
// Normal values scaled by row(r) into the leading rows of the chunk's normalized rows.
template <class Scale>
void fillRows(const Chunk &chunk, uint32_t rows, Scale row) {
  std::normal_distribution<float> normal;
  auto *values = static_cast<uint16_t *>(chunk.ffn.normalized.contents());
  for (uint64_t i = 0; i < uint64_t{rows} * kHidden; ++i) values[i] = floatToBf16(normal(rng) * row(i / kHidden));
}

// A forward of every layer's FFN of a chunk's `rows` normalized rows from a zero residual, split where `split` takes
// the chunk and on the GPU alone otherwise, as the target's prefill encodes it: its bf16 output, whether the split's
// results were usable, and the command's milliseconds.
struct Forward {
  std::vector<uint16_t> bits;
  bool usable = true;
  double milliseconds = 0.0;
  [[nodiscard]] float operator[](uint64_t index) const { return bf16ToFloat(bits[index]); }
};
Forward forward(MetalBackend &backend, const Linear &linear, const Model &model, AneFfn *split, const Chunk &chunk,
                uint32_t rows) {
  const auto &[ffn, hidden] = chunk;
  std::memset(hidden[0].contents(), 0, hidden[0].sizeBytes());
  CommandGraph graph;
  if (model.layers.front().gate->layout() == WeightLayout::Affine64)
    linear.addPrefillSums(graph, ffn.normalized, ffn.sums, *model.layers.front().gate, rows);
  const bool splits = split && split->splits(rows);
  if (split) static_cast<void>(split->begin());
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    const MetalBuffer &residual = hidden[layer & 1], &output = hidden[(layer & 1) ^ 1];
    if (splits)
      split->add(graph, layer, ffn, residual, output, rows);
    else
      linear.addPrefillSwiGlu(graph, model.layers[layer], ffn, residual, output, rows);
  }
  Forward result;
  const auto start = AwakeClock::now();
  if (split)
    static_cast<void>(split->commit(graph, {}).wait());
  else
    run(backend, graph);
  result.milliseconds = millisecondsSince(start);
  result.usable = !split || split->finish();
  const uint16_t *bits = contents<uint16_t>(hidden[kLayers & 1]);
  result.bits.assign(bits, bits + uint64_t{rows} * kHidden);
  return result;
}

// The RMS of the split's output's difference from the GPU's alone, relative to the GPU's, over the elements `counts`
// takes, and whether the split's are finite.
template <class Counts>
std::pair<double, bool> difference(const Forward &expected, const Forward &got, Counts counts) {
  double difference = 0.0, magnitude = 0.0;
  bool finite = true;
  for (uint64_t i = 0; i < got.bits.size(); ++i) {
    if (!counts(i / kHidden, i % kHidden)) continue;
    finite &= std::isfinite(got[i]);
    const double delta = double(got[i]) - expected[i];
    difference += delta * delta;
    magnitude += double(expected[i]) * expected[i];
  }
  return {std::sqrt(difference / magnitude), finite};
}
std::pair<double, bool> difference(const Forward &expected, const Forward &got) {
  return difference(expected, got, [](uint64_t, uint64_t) { return true; });
}

// The split's layers, built from copies of the model's projections that are gone before it runs: it keeps what it
// needs of them.
std::unique_ptr<AneFfn> splitOf(MetalBackend &backend, const Model &model) {
  const std::vector<Projection> copies = model.projections;
  std::vector<SwiGluProjections> layers;
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    layers.push_back({&copies[3 * layer], &copies[3 * layer + 1], &copies[3 * layer + 2]});
  return std::make_unique<AneFfn>(backend, layers, kShare);
}

void split(MetalBackend &backend, const Linear &linear, const Model &model, const Chunk &chunk) {
  const std::unique_ptr<AneFfn> split = splitOf(backend, model);
  if (split->share() != kShare) fail(model.name + ": runs share " + std::to_string(split->share()));
  if (split->splits(AneFfn::kMinimumRows - 1) || !split->splits(AneFfn::kMinimumRows) ||
      !split->splits(AneFfn::kMaximumRows) || split->splits(AneFfn::kMaximumRows + 1))
    fail(model.name + ": the split takes other chunks than those of 512 to 2048 rows");
  // The plan takes whole pages of each of its three buffers, of which Metal may report less.
  const uint64_t planned = AneFfn::plannedBytes(model.layers, kShare), allocated = split->allocatedBytes();
  if (allocated > planned || planned - allocated >= 3 * kHostPageBytes)
    fail(model.name + ": allocated " + std::to_string(allocated) + " bytes, planned " + std::to_string(planned));
  fillRows(chunk, AneFfn::kMaximumRows, [](uint64_t) { return 1.0f; });
  // The most rows first: later commands run on what earlier ones left in the split's surfaces.
  for (const uint32_t rows : {AneFfn::kMaximumRows, AneFfn::kMinimumRows, 700u}) {
    const Forward expected = forward(backend, linear, model, nullptr, chunk, rows);
    const Forward got = forward(backend, linear, model, split.get(), chunk, rows);
    const auto [error, finite] = difference(expected, got);
    std::cout << "  " << model.name << ", " << rows << " rows: " << 100.0 * error << "% RMS from the GPU alone, "
              << got.milliseconds << " ms\n";
    if (!got.usable || !finite || !(error <= kMaximumError))
      fail(model.name + ", " + std::to_string(rows) + " rows: " + std::to_string(100.0 * error) +
           "% RMS from the GPU alone" + (got.usable ? "" : ", unusable: " + split->reason()));
  }
  section("split: " + model.name + " layers at share 0.5 from layers gone before it runs, its planned memory, "
          "chunks of 2048, 512 and 700 rows");
}

// Layers added out of order and buffers short of a chunk's rows are refused before anything is encoded.
void refusals(MetalBackend &backend, const Model &model, const Chunk &chunk) {
  AneFfn split(backend, model.layers, kShare);
  constexpr uint32_t kRows = AneFfn::kMinimumRows;
  const auto &[ffn, hidden] = chunk;
  CommandGraph graph;
  if (!split.begin()) fail("a new split does not begin a command");
  test::rejects([&] { split.add(graph, 1, ffn, hidden[0], hidden[1], kRows); }, "not its next layer 0",
                "the split added layer 1 first");
  split.add(graph, 0, ffn, hidden[0], hidden[1], kRows);
  test::rejects([&] { split.add(graph, 0, ffn, hidden[1], hidden[0], kRows); }, "not its next layer 1",
                "the split added layer 0 twice");
  test::rejects([&] { split.add(graph, 2, ffn, hidden[1], hidden[0], kRows); }, "not its next layer 1",
                "the split skipped layer 1");
  const size_t encoded = graph.dispatches().size();
  const MetalBuffer shorter = backend.view(hidden[0], 0, uint64_t{kRows} * kHidden * 2 - 2);
  test::rejects([&] { split.add(graph, 1, ffn, hidden[1], shorter, kRows); }, "ANE FFN output buffer holds",
                "an output one element short was accepted");
  PrefillFfnBuffers narrow = ffn;
  narrow.normalized = backend.view(ffn.normalized, 0, uint64_t{kRows} * kHidden * 2 - 2);
  test::rejects([&] { split.add(graph, 1, narrow, hidden[1], hidden[0], kRows); }, "ANE FFN input buffer holds",
                "an input one element short was accepted");
  if (graph.dispatches().size() != encoded) fail("a refused layer encoded a dispatch");
  if (!split.begin()) fail("a split begins a command again");
  split.add(graph, 0, ffn, hidden[0], hidden[1], kRows);
  section("refusals: layers out of order and buffers short of the chunk's rows");
}

// The number of evaluations a split queues as it is built: one of each of its functions.
constexpr uint64_t kBuildEvaluations = (AneFfn::kMaximumRows - AneFfn::kMinimumRows) / AneFfn::kProgramStep + 1;

// A fault armed for the next split's program at the evaluation of `layer` in its first forward of `rows` rows: the
// forward's results are unusable and the split stops, without failing its command, within the handoff's bound of
// the evaluation's start, and the GPU alone then computes, bit for bit, what it computes on its own.
void stops(MetalBackend &backend, const Linear &linear, const Model &model, const Chunk &chunk,
           ane::ProgramInstrumentation::Faults faults, const std::string &what, uint32_t rows,
           std::string_view reason) {
  ane::ProgramInstrumentation::arm(faults);
  const auto split = std::make_unique<AneFfn>(backend, model.layers, kShare);
  const Forward faulted = forward(backend, linear, model, split.get(), chunk, rows);
  const std::string label = what + ", " + std::to_string(rows) + " rows";
  if (faulted.usable) fail(label + ": the results were usable");
  if (!split->retired() || split->splits(rows)) fail(label + ": the split did not stop");
  if (split->reason().find(reason) == std::string::npos) fail(label + ": stopped for " + split->reason());
  if (!backend.healthy()) fail(label + ": the backend is unhealthy: " + backend.unhealthyReason());
  if (!(faulted.milliseconds < 4000.0))
    fail(label + ": the command took " + std::to_string(faulted.milliseconds) + " ms");
  const Forward alone = forward(backend, linear, model, nullptr, chunk, rows);
  const Forward after = forward(backend, linear, model, split.get(), chunk, rows);
  if (!after.usable || after.bits != alone.bits)
    fail(label + ": the GPU's forward after the split stopped differs from the GPU's alone");
}

// The split of the affine Q4 and the GGUF model over normalized rows of normal values.
void split(MetalBackend &backend) {
  const Linear linear(backend.capabilities());
  const Model affine = affineModel(backend), gguf = ggufModel(backend);
  takes(backend, affine, gguf);
  const Chunk rows = chunk(backend, AneFfn::kMaximumRows);
  split(backend, linear, affine, rows);
  split(backend, linear, gguf, rows);
  refusals(backend, affine, rows);
}

// ---------------------------------------------------------------- faults
// Each fault of an evaluation at layer 0, 1 and 2 of chunks of 512, 700 and 2048 rows stops the split (stops()); an
// evaluation that signals the event below its value leaves it there.
void faults(MetalBackend &backend) {
  using Faults = ane::ProgramInstrumentation::Faults;
  const Linear linear(backend.capabilities());
  const Model model = affineModel(backend);
  const Chunk rows = chunk(backend, AneFfn::kMaximumRows);
  fillRows(rows, AneFfn::kMaximumRows, [](uint64_t) { return 1.0f; });
  struct Fault {
    const char *name;
    Faults (*arm)(uint64_t evaluation);
    const char *reason;
  };
  const Fault kinds[] = {
      {"a throwing enqueue", [](uint64_t at) { return Faults{.throwingEnqueue = at}; }, "could not start"},
      {"a failed evaluation", [](uint64_t at) { return Faults{.failingEvaluation = at}; }, "an evaluation failed"},
      {"a stalled evaluation", [](uint64_t at) { return Faults{.stalledEvaluation = at}; }, "did not complete"},
      {"a late evaluation",
       [](uint64_t at) { return Faults{.delayedEvaluation = at, .delay = std::chrono::milliseconds(3000)}; },
       "did not complete"},
  };
  for (const Fault &kind : kinds) {
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      for (const uint32_t count : {AneFfn::kMinimumRows, 700u, AneFfn::kMaximumRows})
        stops(backend, linear, model, rows, kind.arm(kBuildEvaluations + layer + 1),
              std::string(kind.name) + " at layer " + std::to_string(layer), count, kind.reason);
    section(std::string("faults: ") + kind.name + " at each layer of chunks of 512, 700 and 2048 rows");
  }

  // The CPU raises the event past an evaluation's signal before it runs.
  test::AneSum sum(backend);
  const test::TemporaryDirectory cache("splash-ane-faults");
  ane::Program program(sum.mil("below"), {}, {std::chrono::seconds(120), {}}, cache.path());
  const metal::SharedEvent event = backend.newSharedEvent();
  event.signal(100);
  struct Report final {
    std::mutex mutex;
    std::condition_variable changed;
    std::optional<bool> result;
  };
  const auto report = std::make_shared<Report>();
  sum.fill(rng);
  program.enqueue(sum.bind(program, "below"), event, 50, 60, [report](bool success) {
    std::lock_guard lock(report->mutex);
    report->result = success;
    report->changed.notify_all();
  });
  std::unique_lock lock(report->mutex);
  report->changed.wait_until(lock, AwakeClock::now() + std::chrono::seconds(10),
                             [&] { return report->result.has_value(); });
  const uint64_t value = [(__bridge id<MTLSharedEvent>)event.nativeHandle() signaledValue];
  if (report->result != true || !sum.summed()) fail("an evaluation waiting on a value passed did not run");
  if (value != 100) fail("an evaluation's signal of 60 moved the event from 100 to " + std::to_string(value));
  section("faults: an evaluation's signal below the event's value leaves it there");
}

// ---------------------------------------------------------------- the program
using ane::Program;
using test::AneSum;
using test::thrown;

const Program::Limits kLimits{std::chrono::seconds(120), {}};
constexpr auto kEvaluationTimeout = std::chrono::seconds(10);

// Purges the service's compilation of the program whose files are in
// `directory`, as the service may on its own: its name is the service's key.
void purge(const std::filesystem::path &directory) {
  @autoreleasepool {
    id model = [(Class<AneTestModel>)NSClassFromString(@"_ANEModel")
        modelAtURL:[NSURL fileURLWithPath:@(directory.c_str()) isDirectory:YES]
               key:@(directory.filename().c_str())];
    [[(Class<AneTestClient>)NSClassFromString(@"_ANEClient") sharedConnection] purgeCompiledModel:model];
  }
}

// One evaluation of `binding` that reports success and sums the inputs.
void evaluates(AneSum &sum, Program &program, const Program::Binding &binding, const std::string &what) {
  sum.fill(rng);
  const std::optional<bool> result = sum.evaluate(program, binding, kEvaluationTimeout);
  if (result != true)
    fail(what + (result ? ": the evaluation failed" : ": the evaluation did not complete"));
  else if (!sum.summed())
    fail(what + ": the output is not the inputs' sum");
}

// What fails as a std::exception, an evaluation, and unload() and load().
void programs(AneSum &sum, const std::filesystem::path &cache) {
  std::string message = thrown<std::runtime_error>([&] { Program broken("not a MIL program", {}, kLimits, cache); });
  if (message.find("ANE compilation failed") == std::string::npos) fail("an invalid MIL: " + message);
  Program program(sum.mil("sum"), {}, kLimits, cache);
  message = thrown<std::invalid_argument>([&] { static_cast<void>(program.procedure("none")); });
  if (message.find("no function none") == std::string::npos) fail("a function the program lacks: " + message);
  message = thrown<std::invalid_argument>(
      [&] { static_cast<void>(program.bind(program.procedure("sum"), sum.oneInput(), sum.output())); });
  if (message.find("input count") == std::string::npos) fail("a binding of one input short: " + message);
  const Program::Binding binding = sum.bind(program, "sum");
  evaluates(sum, program, binding, "an evaluation");
  section("program: an invalid MIL, a function it lacks and a binding of one input short throw; it evaluates");

  const std::vector<_Float16> loaded = sum.values();
  program.unload();
  program.unload();
  message = thrown<std::runtime_error>([&] { sum.enqueue(program, binding, [](bool) {}); });
  if (message.find("not loaded") == std::string::npos) fail("an evaluation of an unloaded program: " + message);
  program.load(kLimits);
  program.load(kLimits);
  const std::optional<bool> result = sum.evaluate(program, binding, kEvaluationTimeout);
  if (result != true)
    fail("an evaluation once loaded again did not succeed");
  else if (sum.values() != loaded)
    fail("an evaluation once loaded again computed other values");

  // A program whose compilation the service no longer holds.
  const test::TemporaryDirectory own("splash-ane-program");
  Program lost(sum.mil("sum_lost"), {}, kLimits, own.path());
  lost.unload();
  purge(test::programDirectory(own.path()));
  message = thrown<std::runtime_error>([&] { lost.load(kLimits); });
  if (message.find("not compiled") == std::string::npos) fail("load() of a program not compiled: " + message);
  section("unload() and load(): twice each, the same values once loaded again; load() does not compile");
}

// Limits that end the wait for the service.
void limits(AneSum &sum, const std::filesystem::path &cache) {
  std::string message = thrown<std::runtime_error>(
      [&] { Program late(sum.mil("sum_late"), {}, {std::chrono::nanoseconds(1), {}}, cache); });
  if (message.find("the Neural Engine did not answer within 1e-09 s") == std::string::npos)
    fail("a limit of 1 ns: " + message);
  // Queued behind the work the first stopped waiting for.
  Program next(sum.mil("sum_next"), {}, kLimits, cache);
  evaluates(sum, next, sum.bind(next, "sum_next"), "the program after a limit ran out");
  message = thrown<ane::Interrupted>([&] {
    Program interrupted(sum.mil("sum_interrupted"), {}, {std::chrono::seconds(120), [] { return true; }}, cache);
  });
  if (message.find("interrupted") == std::string::npos) fail("an interrupted wait: " + message);
  section("limits: 1 ns runs out, and a program comes up after it; an interrupted wait throws Interrupted");
}

// The cache's files: one of other bytes is written again, and one that cannot
// be written leaves no partial file.
void cacheFiles(AneSum &sum) {
  const test::TemporaryDirectory cache("splash-ane-program");
  const std::string mil = sum.mil("sum_cached");
  { Program first(mil, {}, kLimits, cache.path()); }
  const std::filesystem::path directory = test::programDirectory(cache.path()), source = directory / "model.mil";
  std::string other = mil;
  other[other.find("sum_cached")] = 'S';
  test::writeFile(source, other);
  {
    Program again(mil, {}, kLimits, cache.path());
    evaluates(sum, again, sum.bind(again, "sum_cached"), "a program whose cached source was written again");
  }
  const std::vector<uint8_t> written = test::readFile(source);
  if (std::string(written.begin(), written.end()) != mil) fail("a cached source of the same size kept other bytes");

  const auto partials = [&] {
    return std::ranges::count_if(std::filesystem::directory_iterator(directory),
                                 [](const auto &entry) { return entry.path().extension() == ".partial"; });
  };
  test::writeFile(source, other);
  std::filesystem::permissions(directory, std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);
  std::string message = thrown<std::runtime_error>([&] { Program unwritable(mil, {}, kLimits, cache.path()); });
  std::filesystem::permissions(directory, std::filesystem::perms::owner_all);
  if (message.find("unable to write") == std::string::npos) fail("an unwritable cache directory: " + message);
  if (partials()) fail("an unwritable cache directory kept a partial file");
  // A source that cannot be replaced: a directory in its place.
  std::filesystem::remove(source);
  std::filesystem::create_directories(source / "entry");
  message = thrown<std::filesystem::filesystem_error>([&] { Program unreplaced(mil, {}, kLimits, cache.path()); });
  if (message.find("rename") == std::string::npos) fail("a source that cannot be replaced: " + message);
  if (partials()) fail("a source that cannot be replaced left a partial file");
  section("cache: a source of the same size and other bytes is written again; a failed write leaves no partial file");
}

void program(MetalBackend &backend) {
  AneSum sum(backend);
  const test::TemporaryDirectory cache("splash-ane-program");
  programs(sum, cache.path());
  limits(sum, cache.path());
  cacheFiles(sum);
}

} // namespace

int main(int argc, char **argv) {
  const std::string_view mode = argc == 3 ? argv[2] : "";
  if (mode != "kernels" && mode != "split" && mode != "faults" && mode != "program") {
    std::cerr << "usage: ane-ffn METALLIB kernels|split|faults|program\n";
    return 2;
  }
  try {
    MetalBackend backend(argv[1]);
    if (mode == "kernels") {
      std::vector<float> sign(ANE_FFN_INTERMEDIATE_BLOCK);
      for (float &value : sign) value = rng() & 1 ? -1.0f : 1.0f;
      const MetalBuffer signs = upload(backend, sign);
      inputs(backend, signs, sign);
      weights(backend, signs, sign);
      join(backend);
    } else if (mode == "split") {
      split(backend);
    } else if (mode == "faults") {
      faults(backend);
    } else {
      program(backend);
    }
  } catch (const std::exception &error) {
    std::cout << "FAIL " << error.what() << '\n';
    return 1;
  }
  if (failures) {
    std::cout << failures << " failures\n";
    return 1;
  }
  std::cout << "ane-ffn " << mode << ": all checks passed\n";
  return 0;
}
