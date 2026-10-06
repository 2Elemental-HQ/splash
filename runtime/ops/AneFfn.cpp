#include "ops/AneFfn.hpp"

#include "AwakeClock.hpp"
#include "Checked.hpp"
#include "StderrLine.hpp"
#include "metal/abi/AneFfn.h"
#include "metal/abi/QuantFormat.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {
namespace {

using Element = ane::Surface::Element;

// The ANE program multiplies gate's and up's hidden inputs, and down's ANE
// inputs, in segments of kSegment channels, a matmul each: on an M5 Pro, two
// of 2560 run a 5120-channel FFN about 14% faster than one of 5120 (ane-ffn's
// FFN, 2048 rows). The weights of a segment are staged in whole blocks of
// either rotation, and its inputs packed in whole tiles.
constexpr uint32_t kSegment = 2560;
static_assert(kSegment % ANE_FFN_INPUT_BLOCK == 0 && kSegment % ANE_FFN_INTERMEDIATE_BLOCK == 0 &&
                  kSegment % ANE_FFN_TILE == 0,
              "segments hold whole rotation blocks and packing tiles");
constexpr uint32_t kQuantGroup = 64;
// The hidden channels ane_ffn_rotate's simdgroups rotate a block each of.
constexpr uint32_t kRotateGroup = ANE_FFN_INPUT_BLOCK * (ANE_FFN_ROTATE_THREADS / 32);
// The channels a split moves in: whole blocks of the intermediate rotation
// for the ANE, whole 256-row tiles of the weight planes for the GPU.
constexpr uint32_t kChannelUnit = std::max(QUANT_TILE_ROWS, ANE_FFN_INTERMEDIATE_BLOCK);
static_assert(kChannelUnit % ANE_FFN_WEIGHT_ROWS == 0 && kSegment % ANE_FFN_WEIGHT_ROWS == 0,
              "the weight kernels' threadgroups take whole rows of every matrix the split stages");
// How long an ANE evaluation may run once the GPU has raised its `ready`
// before the split stops (ane::Handoff). An evaluation takes 10-60 ms. Metal
// fails a command buffer whose event wait stays unmet for 5 s ("Caused GPU
// Timeout Error", macOS 27.0 on an M5 Max), and once two have failed so it
// ignores the process's later commands on that queue; the GPU waits on an
// evaluation only after its own part of the layer, so the release reaches a
// wait at most 2 s old, 3 s short of that.
constexpr auto kHandoffBound = std::chrono::seconds(2);
// How long the service may take to compile and load the program: four times
// the cold compile of the largest the split compiles, calibration's at share
// 0.8 (27 s on an M5 Max, about 30 s on an M6), so that a busy Mac still
// finishes and only a service that stopped answering runs out of it.
constexpr auto kProgramLimit = std::chrono::seconds(120);

uint32_t gpuChannels(uint32_t intermediate, double share) {
  if (!(share > 0.0 && share < 1.0)) throw std::invalid_argument("ANE FFN share must lie in (0, 1)");
  const auto units = static_cast<uint32_t>(std::lround((1.0 - share) * intermediate / kChannelUnit));
  if (!units || units >= intermediate / kChannelUnit)
    throw std::invalid_argument("ANE FFN share leaves the GPU or the ANE no channels");
  return units * kChannelUnit;
}

std::vector<uint32_t> segments(uint32_t channels) {
  std::vector<uint32_t> result;
  for (uint32_t begin = 0; begin < channels; begin += kSegment) result.push_back(std::min(kSegment, channels - begin));
  return result;
}

// A projection the split takes: affine Q4, or one quantized GGUF image tensor.
bool splittable(const Projection &projection) {
  if (projection.layout() == WeightLayout::Affine64) return true;
  const std::vector<QuantizedSegment> &segments = projection.blocks().segments;
  return segments.size() == 1 && !segments.front().isFloat() && !projection.rotation;
}

// Why the split does not take `layers`, or null when it does.
const char *unsupported(std::span<const SwiGluProjections> layers) {
  if (layers.empty()) return "ANE FFN split has no layers";
  const uint32_t hidden = layers.front().gate->inputSize, intermediate = layers.front().gate->outputSize;
  if (hidden % kSegment) return "ANE FFN split needs a hidden size of whole 2560-channel segments";
  if (hidden % kRotateGroup || hidden / kRotateGroup > ANE_FFN_ROTATE_BLOCKS)
    return "ANE FFN split needs a hidden size ane_ffn_rotate takes";
  if (intermediate % kChannelUnit) return "ANE FFN split needs whole rotation blocks of intermediate channels";
  for (const SwiGluProjections &layer : layers) {
    for (const Projection *projection : {layer.gate, layer.up, layer.down})
      if (!splittable(*projection)) return "ANE FFN split needs affine Q4 projections or quantized GGUF tensors";
    if (layer.gate->outputSize != intermediate || layer.gate->inputSize != hidden ||
        layer.up->outputSize != intermediate || layer.up->inputSize != hidden || layer.down->outputSize != hidden ||
        layer.down->inputSize != intermediate)
      return "ANE FFN split layers differ in shape";
  }
  return nullptr;
}

// The format of a projection's planes: its GGUF tensor's, or 0 for affine Q4.
uint32_t formatOf(const Projection &projection) {
  return projection.layout() == WeightLayout::Affine64 ? 0 : projection.blocks().segments.front().formatId;
}

// The rotation a weight kernel applies: over gate's and up's hidden inputs,
// or over down's intermediate inputs.
enum class Rotation : uint8_t { Inputs, Intermediate };
uint32_t blockOf(Rotation rotation) {
  return rotation == Rotation::Inputs ? ANE_FFN_INPUT_BLOCK : ANE_FFN_INTERMEDIATE_BLOCK;
}
std::string kernel(const char *name, const char *suffix, Rotation rotation) {
  return std::string(name) + suffix + (rotation == Rotation::Inputs ? "_inputs" : "_intermediate");
}

// The Hadamard signs D of the rotations R = D H / sqrt(n) of inputs, weights
// and the ANE's intermediate rows alike: a block of n values takes the first n.
std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> rotationSigns() {
  std::mt19937 generator(20260930);
  std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> signs{};
  for (float &sign : signs) sign = (generator() & 1) ? -1.0f : 1.0f;
  return signs;
}

// The program's constant: the rotation of each of the ANE's intermediate
// channels as the [channels, ANE_FFN_INTERMEDIATE_BLOCK, 1, 1] weight of a
// grouped 1x1 convolution.
std::vector<uint8_t> rotationBlob(uint32_t channels, const std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> &signs) {
  constexpr uint32_t block = ANE_FFN_INTERMEDIATE_BLOCK;
  std::vector<_Float16> values(uint64_t{channels} * block);
  const float norm = 1.0f / std::sqrt(float(block));
  for (uint32_t channel = 0; channel < channels; ++channel)
    for (uint32_t input = 0; input < block; ++input) {
      const uint32_t output = channel % block;
      const float hadamard = (std::popcount(output & input) & 1) ? -1.0f : 1.0f;
      values[uint64_t{channel} * block + input] = _Float16(signs[output] * hadamard * norm);
    }
  return ane::constantBlob(values);
}

// The name of the program's function of `rows` rows.
std::string functionName(uint32_t rows) { return "ffn" + std::to_string(rows); }

// The MIL dimensions of `rows` rows of `width` values.
std::string dimensions(uint64_t rows, uint64_t width) {
  return "[1, 1, " + std::to_string(rows) + ", " + std::to_string(width) + "]";
}
std::string tensor(const char *type, uint64_t rows, uint64_t width) {
  return std::string("tensor<") + type + ", " + dimensions(rows, width) + ">";
}
// An fp16 constant, exactly.
std::string fp16(double value) {
  char text[32];
  std::snprintf(text, sizeof text, "fp16(%a)", value);
  return text;
}

// Calibration. At share s, a layer of a full chunk takes
//   T(s) = max(G(s), A(s), uG G(s) + uA A(s)):
// G(s) is the GPU's part alone (its channels, the staging of the ANE's int8
// weights, and the packing and join of the ANE's channels) and A(s) the
// ANE's evaluation, each close to linear in s. The third term is the memory
// bandwidth both parts share when they run together, uG and uA each one's
// fraction of it alone. Timing each part alone at two shares gives G and A,
// and both together at those shares gives uG and uA where the bandwidth
// binds there. Neither part waits on the other while timed: a command buffer
// that hands off to the ANE after little GPU work can start the next one
// late, by tens of milliseconds at random on the M5 Max with Qwen3.8-27B
// resident, which would time Metal's scheduling instead.
// The split runs where G meets A: on Qwen3.8-27B's FFN the fastest share of a
// prefill of seconds, measured on the M5 Max (0.24), the M5 Pro (0.41) and the
// M6 (0.76). Short of it the ANE waits on the GPU, whose part the third term
// slows where the two share bandwidth, as on the M6; past it the GPU waits on
// the ANE, which T takes to run as fast as alone. It does not: on the M6 the
// two slow each other's memory accesses beyond the GPU's L2 (an ANE
// evaluation takes a third longer beside a GPU kernel that streams memory),
// and a minute of prefill warms the ANE a tenth slower. There these timings put the
// fastest share at 0.85-0.88, which a 14K-token prompt runs 7% slower than
// 0.76. The split takes the least share within kTolerance of it, for the ANE's
// memory and error, if T(s) predicts a gain of kMinimumGain there on the GPU
// alone, timed as well.
constexpr std::array<double, 2> kTimedShares{0.4, 0.8};
// Per-layer milliseconds come from commands of one and of kTimedLayers
// layers, which cancels each command's fixed costs, of layers spread over the
// model, whose formats can differ with depth; one more layer gives the last
// one's next weights to stage.
constexpr uint32_t kTimedLayers = 4;
// The split's gain below which the GPU runs the FFN alone, the excess over
// max(G, A) that shows the bandwidth binding, the slack of the optimum, and
// the most the split's output may differ from the GPU's (Calibration).
constexpr double kMinimumGain = 0.05, kBindingMargin = 0.03, kTolerance = 0.01, kMaximumError = 0.05;

// A share's per-layer milliseconds: the GPU's part alone, the ANE's
// evaluations alone, and both together.
struct Timing final {
  double share, gpu, ane, both;
};
struct Line final {
  double at0, slope;
  [[nodiscard]] double operator()(double share) const noexcept { return at0 + slope * share; }
};
struct Model final {
  Line gpu, ane;
  double uG = 0.0, uA = 0.0;
  [[nodiscard]] double operator()(double share) const noexcept {
    const double g = gpu(share), a = ane(share);
    return std::max({g, a, uG * g + uA * a});
  }
};

Model fit(const Timing &p, const Timing &q) {
  const auto line = [&](double Timing::*part) {
    const double slope = (q.*part - p.*part) / (q.share - p.share);
    return Line{p.*part - slope * p.share, slope};
  };
  Model model{line(&Timing::gpu), line(&Timing::ane)};
  const auto excess = [](const Timing &x) { return x.both / std::max(x.gpu, x.ane) - 1.0; };
  bool solved = false;
  if (excess(p) > kBindingMargin && excess(q) > kBindingMargin) {
    const double det = p.gpu * q.ane - p.ane * q.gpu;
    if (std::abs(det) > 1e-9) {
      model.uG = (p.both * q.ane - p.ane * q.both) / det;
      model.uA = (p.gpu * q.both - p.both * q.gpu) / det;
      solved = model.uG >= 0.0 && model.uG <= 1.05 && model.uA >= 0.0 && model.uA <= 1.05;
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

// The GPU's channel units, of `units`, at which full chunks split: where the
// GPU's part and the ANE's alone take most nearly as long, the most units
// within kTolerance of the least max(G, A), or the `previous` choice while it
// still is, as each share compiles a program of its own, which takes half a
// minute on an M6, whose ANE service keeps few; none if `model` predicts no
// gain of kMinimumGain there on `gpuAlone` milliseconds.
std::optional<uint32_t> choose(const Model &model, double gpuAlone, uint32_t units,
                               std::optional<uint32_t> previous) {
  const auto share = [&](uint32_t gpuUnits) { return 1.0 - static_cast<double>(gpuUnits) / units; };
  const auto longer = [&](uint32_t gpuUnits) {
    return std::max(model.gpu(share(gpuUnits)), model.ane(share(gpuUnits)));
  };
  double best = std::numeric_limits<double>::infinity();
  for (uint32_t gpuUnits = 1; gpuUnits < units; ++gpuUnits) best = std::min(best, longer(gpuUnits));
  uint32_t gpuUnits = units - 1;
  if (previous && *previous && *previous < units && longer(*previous) <= (1.0 + kTolerance) * best)
    gpuUnits = *previous;
  else
    while (longer(gpuUnits) > (1.0 + kTolerance) * best) --gpuUnits;
  if (model(share(gpuUnits)) > (1.0 - kMinimumGain) * gpuAlone) return std::nullopt;
  return gpuUnits;
}

// What calibration's remembered choice is kept under: the layers' shapes and
// formats.
std::string choiceKey(std::span<const SwiGluProjections> layers) {
  std::string key = "ane-ffn";
  for (const SwiGluProjections &layer : layers)
    for (const Projection *projection : {layer.gate, layer.up, layer.down})
      key += " " + std::to_string(projection->outputSize) + "x" + std::to_string(projection->inputSize) + ":" +
             std::to_string(formatOf(*projection)) + (projection->layout() == WeightLayout::Affine64 ? "a" : "g");
  return key;
}

// The rows of the program's functions, ascending.
std::vector<uint32_t> functionRows() {
  std::vector<uint32_t> rows;
  for (uint32_t row = AneFfn::kMinimumRows; row <= AneFfn::kMaximumRows; row += AneFfn::kProgramStep)
    rows.push_back(row);
  return rows;
}

} // namespace

AneFfn::Planes::Planes(const Projection &projection) {
  if (projection.layout() == WeightLayout::Affine64) {
    requireAffineProjection(projection, {projection.outputSize, projection.inputSize});
    const AffineWeights &weights = projection.affine();
    groups = projection.inputSize / kQuantGroup;
    buffers = {weights.weights, weights.scales, weights.biases};
    return;
  }
  const QuantizedSegment &segment = projection.blocks().segments.front();
  requireSegmentPlanes(segment, "ANE FFN projection");
  groups = projection.inputSize / 32;
  format = segment.formatId;
  suffix = "_gguf";
  buffers = {segment.plane0, segment.plane1Slot(), segment.meta};
}

AneFfn::Shape AneFfn::shapeOf(std::span<const SwiGluProjections> layers, double share) {
  if (const char *reason = unsupported(layers)) throw std::invalid_argument(reason);
  Shape shape;
  shape.hidden = layers.front().gate->inputSize;
  shape.intermediate = layers.front().gate->outputSize;
  shape.layers = static_cast<uint32_t>(layers.size());
  shape.gpu = gpuChannels(shape.intermediate, share);
  shape.ane = shape.intermediate - shape.gpu;
  shape.downSegments = segments(shape.ane);
  return shape;
}

template <class MakeBuffer, class MakeSurface>
AneFfn::Memory AneFfn::allocate(const Shape &shape, MakeBuffer &&buffer, MakeSurface &&surface) {
  const uint32_t inputSegments = shape.hidden / kSegment;
  Memory memory;
  memory.signs = buffer(ANE_FFN_INTERMEDIATE_BLOCK * sizeof(float), "ane ffn signs");
  memory.rowScales = buffer(uint64_t{shape.layers} * (2 * shape.ane + shape.hidden) * sizeof(_Float16),
                            "ane ffn row scales");
  memory.rotated = buffer(uint64_t{kMaximumRows} * shape.hidden * sizeof(_Float16), "ane ffn rotated input");
  for (uint32_t k = 0; k < inputSegments; ++k) memory.inputs.push_back(surface(kSegment, kMaximumRows, Element::Int8));
  memory.tokenScale = surface(1, kMaximumRows, Element::Float16);
  memory.partial = surface(shape.hidden, kMaximumRows, Element::Float16);
  for (Weights &set : memory.sets) {
    for (uint32_t k = 0; k < inputSegments; ++k) {
      set.gate.push_back(surface(shape.ane, kSegment, Element::Int8));
      set.up.push_back(surface(shape.ane, kSegment, Element::Int8));
    }
    for (const uint32_t width : shape.downSegments) set.down.push_back(surface(shape.hidden, width, Element::Int8));
    set.gateScale = surface(shape.ane, 1, Element::Float16);
    set.upScale = surface(shape.ane, 1, Element::Float16);
    set.downScale = surface(shape.hidden, 1, Element::Float16);
  }
  return memory;
}

uint64_t AneFfn::plannedBytes(std::span<const SwiGluProjections> layers, double share) {
  uint64_t bytes = 0;
  static_cast<void>(allocate(
      shapeOf(layers, share),
      [&](uint64_t size, const char *) {
        bytes += alignUp(size);
        return metal::MetalBuffer{};
      },
      [&](uint32_t rows, uint32_t width, Element element) {
        bytes += ane::Surface::bytes(rows, width, element);
        return ane::Surface{};
      }));
  return bytes;
}

// x<k>: the rotated input rows of segment k, channel-major in int8, and tx
// their per-token scales; wg<k> and wu<k>: gate's and up's rotated int8 rows
// over segment k, and sg and su their per-row scales; wd<i>: down's rotated
// int8 rows over segment i of its ANE inputs, and sd their per-row scales.
std::vector<AneFfn::Input> AneFfn::programInputs(const Shape &shape, const Memory &memory) {
  const auto both = [&](auto member) {
    return std::array{&(memory.sets[0].*member), &(memory.sets[1].*member)};
  };
  std::vector<Input> inputs;
  for (uint32_t k = 0; k < memory.inputs.size(); ++k)
    inputs.push_back({"x" + std::to_string(k), {&memory.inputs[k], &memory.inputs[k]}, kSegment, 0});
  inputs.push_back({"tx", {&memory.tokenScale, &memory.tokenScale}, 1, 0});
  for (uint32_t k = 0; k < memory.inputs.size(); ++k) {
    inputs.push_back({"wg" + std::to_string(k), {&memory.sets[0].gate[k], &memory.sets[1].gate[k]}, shape.ane, kSegment});
    inputs.push_back({"wu" + std::to_string(k), {&memory.sets[0].up[k], &memory.sets[1].up[k]}, shape.ane, kSegment});
  }
  inputs.push_back({"sg", both(&Weights::gateScale), shape.ane, 1});
  inputs.push_back({"su", both(&Weights::upScale), shape.ane, 1});
  for (uint32_t i = 0; i < shape.downSegments.size(); ++i)
    inputs.push_back({"wd" + std::to_string(i), {&memory.sets[0].down[i], &memory.sets[1].down[i]}, shape.hidden,
                      shape.downSegments[i]});
  inputs.push_back({"sd", both(&Weights::downScale), shape.hidden, 1});
  return inputs;
}

// The ANE's share of the FFN over `rows` rows. Every int8 value is
// dequantized by 1 / ANE_FFN_INT8_UNIT against fp16 overflow, which the
// scales carry back. The intermediate rows are rotated and quantized per
// token here. The function reads and writes the leading `rows` of each row of
// the chunk's surfaces.
std::string AneFfn::function(const Shape &shape, std::span<const Input> inputs, const ane::Surface &output,
                             uint32_t rows) {
  const uint32_t inputSegments = shape.hidden / kSegment, channels = shape.ane;
  const std::string unit = fp16(1.0 / ANE_FFN_INT8_UNIT);
  std::string parameters, body;
  const auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
  const auto f16 = [&](const std::string &name, uint64_t height, uint64_t width, const std::string &expression) {
    line(tensor("fp16", height, width) + " " + name + " = " + expression);
  };
  for (const Input &input : inputs) {
    const uint32_t width = input.width ? input.width : rows;
    parameters += (parameters.empty() ? "" : ", ") + input.surfaces[0]->bufferType(input.rows, width) + " " + input.name;
    line(tensor(ane::Surface::milType(input.surfaces[0]->element), input.rows, width) + " " + input.name +
         "_t = tensor_buffer_to_tensor<ios17>(input = " + input.name + ")");
  }
  const auto matmul = [&](const std::string &name, const std::string &weights, const std::string &values,
                          uint64_t height, uint64_t width) {
    f16(weights + "_d", height, width, "dequantize(input = " + weights + "_t, scale = " + unit + ")");
    line(tensor("fp16", height, rows) + " " + name +
         " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + weights + "_d, y = " + values + ")");
  };
  const auto sum = [&](const std::string &prefix, size_t terms, uint64_t height) {
    std::string total = prefix + "0";
    for (size_t term = 1; term < terms; ++term) {
      const std::string next = prefix + "_sum" + std::to_string(term);
      f16(next, height, rows, "add(x = " + total + ", y = " + prefix + std::to_string(term) + ")");
      total = next;
    }
    return total;
  };

  for (uint32_t k = 0; k < inputSegments; ++k) {
    const std::string x = "x" + std::to_string(k);
    f16(x + "_d", kSegment, rows, "dequantize(input = " + x + "_t, scale = " + unit + ")");
  }
  for (const char *projection : {"g", "u"}) {
    const std::string p = projection;
    for (uint32_t k = 0; k < inputSegments; ++k)
      matmul(p + "m" + std::to_string(k), "w" + p + std::to_string(k), "x" + std::to_string(k) + "_d", channels,
             kSegment);
    f16(p + "s", channels, rows, "mul(x = " + sum(p + "m", inputSegments, channels) + ", y = s" + p + "_t)");
  }
  const std::string c = std::to_string(channels), r = std::to_string(rows),
                    block = std::to_string(ANE_FFN_INTERMEDIATE_BLOCK);
  f16("gt", channels, rows, "mul(x = gs, y = tx_t)");
  f16("sig", channels, rows, "sigmoid(x = gt)");
  f16("silu", channels, rows, "mul(x = gt, y = sig)");
  f16("h", channels, rows, "mul(x = silu, y = us)");
  line("tensor<fp16, [1, " + c + ", 1, " + r + "]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, " + c + ", 1, " +
       r + "]))");
  line("tensor<fp16, [" + c + ", " + block + ", 1, 1]> rotation = const()[name = string(\"rotation\"), val = tensor<fp16, [" +
       c + ", " + block + ", 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(" +
       std::to_string(ane::kConstantOffset) + ")))]");
  line("tensor<fp16, [1, " + c + ", 1, " + r + "]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32(" +
       std::to_string(channels / ANE_FFN_INTERMEDIATE_BLOCK) +
       "), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, "
       "1]), weight = rotation, x = h4)");
  f16("hr", channels, rows, "reshape(x = hr4, shape = tensor<int32, [4]>(" + dimensions(channels, rows) + "))");
  f16("habs", channels, rows, "abs(x = hr)");
  f16("peak", 1, rows, "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
  f16("floor", 1, rows, "maximum(x = peak, y = " + fp16(0x1p-12) + ")");
  f16("inverse", 1, rows, "real_div(x = " + fp16(ANE_FFN_INT8_PEAK) + ", y = floor)");
  f16("hs", channels, rows, "mul(x = hr, y = inverse)");
  line(tensor("int8", channels, rows) + " hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))");
  f16("hd", channels, rows, "dequantize(input = hq, scale = " + unit + ")");
  f16("hscale", 1, rows, "mul(x = floor, y = " + fp16(double(ANE_FFN_INT8_UNIT) / ANE_FFN_INT8_PEAK) + ")");
  uint32_t begin = 0;
  for (size_t i = 0; i < shape.downSegments.size(); ++i) {
    const uint32_t width = shape.downSegments[i];
    const std::string index = std::to_string(i), slice = "hd" + index;
    f16(slice, width, rows,
        "slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, " + std::to_string(begin) +
            ", 0]), size = tensor<int32, [4]>(" + dimensions(width, rows) + "))");
    matmul("dm" + index, "wd" + index, slice, shape.hidden, width);
    begin += width;
  }
  f16("ds", shape.hidden, rows, "mul(x = " + sum("dm", shape.downSegments.size(), shape.hidden) + ", y = sd_t)");
  f16("ys", 1, rows, "mul(x = hscale, y = tx_t)");
  f16("yt", shape.hidden, rows, "mul(x = ds, y = ys)");
  line(output.bufferType(shape.hidden, rows) +
       " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
       "strides = tensor<int64, [4]>(" + output.strides(shape.hidden) + "))");
  return "    func " + functionName(rows) + "<ios18>(" + parameters + ") {\n" + body + "    } -> (y);\n";
}

std::string AneFfn::program(const Shape &shape, std::span<const Input> inputs, const ane::Surface &output,
                            std::span<const uint32_t> rows) {
  std::string text = "program(1.3)\n{\n";
  for (const uint32_t count : rows) text += function(shape, inputs, output, count);
  return text + "}\n";
}

AneFfn::AneFfn(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers, double share)
    : AneFfn(backend, layers, share, functionRows()) {}

AneFfn::AneFfn(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers, double share,
               std::span<const uint32_t> functionRows)
    : backend_(backend), linear_(backend.capabilities()), share_(share), shape_(shapeOf(layers, share)),
      handoff_(backend, kHandoffBound) {
  // What Metal allocates, which the memory audit compares with the plan.
  memory_ = allocate(
      shape_,
      [&](uint64_t bytes, const char *label) {
        metal::MetalBuffer buffer = backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
        allocatedBytes_ += buffer.allocatedBytes();
        return buffer;
      },
      [&](uint32_t rows, uint32_t width, Element element) {
        ane::Surface surface = ane::Surface::create(backend_, rows, width, element);
        allocatedBytes_ += surface.buffer.allocatedBytes();
        return surface;
      });
  const std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> signs = rotationSigns();
  std::memcpy(memory_.signs.contents(), signs.data(), sizeof signs);

  for (const SwiGluProjections &source : layers)
    layers_.push_back({{Planes(*source.gate), Planes(*source.up), Planes(*source.down)},
                       source.gate->leadingRows(backend_, shape_.gpu),
                       source.up->leadingRows(backend_, shape_.gpu),
                       source.down->leadingInputs(shape_.gpu)});

  const std::vector<Input> inputs = programInputs(shape_, memory_);
  program_ = std::make_unique<ane::Program>(program(shape_, inputs, memory_.partial, functionRows),
                                            rotationBlob(shape_.ane, signs),
                                            ane::Program::Limits{kProgramLimit, {}});
  for (const uint32_t rows : functionRows) {
    Evaluation &evaluation = evaluations_.emplace_back();
    evaluation.rows = rows;
    const uint32_t procedure = program_->procedure(functionName(rows));
    for (uint32_t set = 0; set < 2; ++set) {
      std::vector<ane::Surface> surfaces;
      for (const std::string &name : program_->inputs(procedure)) {
        const auto input = std::ranges::find(inputs, name, &Input::name);
        if (input == inputs.end()) throw std::logic_error("ANE FFN program has an unknown input " + name);
        surfaces.push_back(*input->surfaces[set]);
      }
      evaluation.bindings.push_back(program_->bind(procedure, surfaces, memory_.partial));
    }
  }

  // Each row's shared int8 scale over the ANE's share of its inputs.
  metal::CommandGraph graph;
  for (uint32_t layer = 0; layer < layers_.size(); ++layer) {
    const auto add = [&](Matrix matrix, uint32_t row, uint32_t input, uint32_t width, uint32_t rows,
                         Rotation rotation) {
      const Planes &planes = layers_[layer].planes[static_cast<size_t>(matrix)];
      graph.add(kernel("ane_ffn_row_scale", planes.suffix, rotation),
                {planes.buffers[0], planes.buffers[1], planes.buffers[2], rowScales(layer, matrix), memory_.signs},
                AneFfnWeightParams{planes.groups, row, input, width, 0, 0, planes.format},
                {rows / ANE_FFN_WEIGHT_ROWS, 1, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
    };
    add(Matrix::Gate, shape_.gpu, 0, shape_.hidden, shape_.ane, Rotation::Inputs);
    add(Matrix::Up, shape_.gpu, 0, shape_.hidden, shape_.ane, Rotation::Inputs);
    add(Matrix::Down, 0, shape_.gpu, shape_.ane, shape_.hidden, Rotation::Intermediate);
  }
  static_cast<void>(backend_.submitCommandAsync(graph.dispatches()).wait());

  // Each function evaluates once now, so that one the ANE loads but cannot run
  // fails where startup falls back to the GPU, not in a request.
  std::vector<std::pair<uint32_t, uint32_t>> functions;
  for (uint32_t index = 0; index < evaluations_.size(); ++index) functions.emplace_back(index, 0);
  queueNow(functions);
  if (!handoff_.finish()) throw std::runtime_error("ANE FFN program does not run (" + handoff_.reason() + ")");
}

bool AneFfn::supports(std::span<const SwiGluProjections> layers) { return !unsupported(layers); }

bool AneFfn::splits(uint32_t rows) const {
  return rows >= kMinimumRows && rows <= kMaximumRows && !handoff_.retired();
}

metal::MetalBuffer AneFfn::rowScales(uint32_t layer, Matrix matrix) const {
  const uint64_t perLayer = 2 * shape_.ane + shape_.hidden, part = static_cast<uint64_t>(matrix);
  return backend_.view(memory_.rowScales, (layer * perLayer + part * shape_.ane) * sizeof(_Float16),
                       uint64_t{matrix == Matrix::Down ? shape_.hidden : shape_.ane} * sizeof(_Float16));
}

// Layer `layer`'s int8 weights and row scales into staging set `set`.
void AneFfn::addWeights(metal::CommandGraph &graph, uint32_t layer, uint32_t set) const {
  const Layer &source = layers_.at(layer);
  const Weights &target = memory_.sets[set];
  const auto add = [&](Matrix matrix, const ane::Surface &output, const ane::Surface &scale, uint32_t row,
                       uint32_t input, uint32_t width, uint32_t rows, Rotation rotation) {
    const Planes &planes = source.planes[static_cast<size_t>(matrix)];
    const uint32_t scaleStride = scale.strideBytes / uint32_t{sizeof(_Float16)};
    requireBytes(output.buffer, rowBytes(rows, output.strideBytes, width, 1), "ANE FFN weight surface");
    requireBytes(scale.buffer, rowBytes(rows, scaleStride, 1, sizeof(_Float16)), "ANE FFN weight scale surface");
    graph.add(kernel("ane_ffn_weights", planes.suffix, rotation),
              {planes.buffers[0], planes.buffers[1], planes.buffers[2], rowScales(layer, matrix), output.buffer,
               scale.buffer, memory_.signs},
              AneFfnWeightParams{planes.groups, row, input, width, output.strideBytes, scaleStride, planes.format},
              {rows / ANE_FFN_WEIGHT_ROWS, width / blockOf(rotation), 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
  };
  for (uint32_t k = 0; k < target.gate.size(); ++k) {
    add(Matrix::Gate, target.gate[k], target.gateScale, shape_.gpu, k * kSegment, kSegment, shape_.ane,
        Rotation::Inputs);
    add(Matrix::Up, target.up[k], target.upScale, shape_.gpu, k * kSegment, kSegment, shape_.ane, Rotation::Inputs);
  }
  uint32_t begin = shape_.gpu;
  for (size_t i = 0; i < shape_.downSegments.size(); ++i) {
    add(Matrix::Down, target.down[i], target.downScale, 0, begin, shape_.downSegments[i], shape_.hidden,
        Rotation::Intermediate);
    begin += shape_.downSegments[i];
  }
}

bool AneFfn::begin() {
  // A command committed but never finished failed on the GPU, and the
  // evaluations it queued with it.
  if (std::exchange(unfinished_, false)) {
    handoff_.cancel("a command it split did not finish");
    stopped();
  }
  jobs_.clear();
  nextLayer_ = 0;
  return !handoff_.retired();
}

void AneFfn::add(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn,
                 metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows) {
  encode(graph, layer, ffn, residual, output, rows, Parts::Both);
}

void AneFfn::encode(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn,
                    metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows, Parts parts) {
  if (rows < kMinimumRows || rows > kMaximumRows)
    throw std::invalid_argument("ANE FFN split does not take a chunk of these rows");
  if (layer != nextLayer_ || layer >= layers_.size())
    throw std::logic_error("ANE FFN split adds layer " + std::to_string(layer) + ", not its next layer " +
                           std::to_string(nextLayer_) + " (begin() starts each command at layer 0)");
  const uint64_t hiddenBytes = uint64_t{rows} * shape_.hidden * sizeof(uint16_t);
  requireBytes(ffn.normalized, hiddenBytes, "ANE FFN input");
  requireBytes(output, hiddenBytes, "ANE FFN output");
  const Layer &current = layers_[layer];
  const uint32_t set = layer & 1, tiles = (rows + ANE_FFN_TILE - 1) / ANE_FFN_TILE;
  const bool ane = parts == Parts::Both;
  const auto found =
      std::ranges::find_if(evaluations_, [&](const Evaluation &evaluation) { return evaluation.rows >= rows; });
  if (found == evaluations_.end()) throw std::invalid_argument("ANE FFN has no function of the chunk's rows");
  const auto index = static_cast<uint32_t>(found - evaluations_.begin());
  // Each command stages layer 0's weights, then each layer the next one's.
  if (!layer) addWeights(graph, 0, 0);
  graph.add("ane_ffn_rotate", {ffn.normalized, memory_.signs, memory_.rotated, memory_.tokenScale.buffer},
            AneFfnRotateParams{shape_.hidden}, {rows, 1, 1}, {ANE_FFN_ROTATE_THREADS, 1, 1});
  for (uint32_t k = 0; k < memory_.inputs.size(); ++k)
    graph.add("ane_ffn_pack", {memory_.rotated, memory_.inputs[k].buffer},
              AneFfnPackParams{shape_.hidden, k * kSegment, memory_.inputs[k].strideBytes},
              {tiles, kSegment / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  const uint64_t ready = ane ? handoff_.next() : 0;
  if (ane) graph.signal(handoff_.event(), ready);
  linear_.addPrefillSwiGlu(graph, {&current.gate, &current.up, &current.down}, ffn, residual, output, rows);
  if (layer + 1 < layers_.size()) addWeights(graph, layer + 1, set ^ 1);
  const uint64_t done = ane ? handoff_.next() : 0;
  if (ane) graph.wait(handoff_.event(), done);
  graph.add("ane_ffn_join", {output, memory_.partial.buffer},
            AneFfnJoinParams{shape_.hidden, memory_.partial.strideBytes / uint32_t{sizeof(_Float16)}, rows},
            {tiles, shape_.hidden / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  if (ane) jobs_.push_back({index, set, ready, done});
  ++nextLayer_;
}

metal::CommandTicket AneFfn::commit(const metal::CommandGraph &graph, metal::CommandCompletion completion) {
  const std::vector<Job> jobs = std::exchange(jobs_, {});
  for (const Job &job : jobs) queue(job);
  unfinished_ = !jobs.empty();
  const auto cancel = [&](const std::string &reason) {
    if (!std::exchange(unfinished_, false)) return;
    handoff_.cancel(reason);
    stopped();
  };
  try {
    return backend_.submitCommandAsync(graph.command(), std::move(completion));
  } catch (const std::exception &error) {
    cancel(std::string("its Metal command was not submitted: ") + error.what());
    throw;
  } catch (...) {
    cancel("its Metal command was not submitted");
    throw;
  }
}

bool AneFfn::finish() {
  if (completed()) return true;
  stopped();
  return false;
}

bool AneFfn::completed() {
  unfinished_ = false;
  return handoff_.finish();
}

void AneFfn::stopped() {
  if (std::exchange(warned_, true)) return;
  logWarning("Neural Engine FFN split stopped (", handoff_.reason(),
             "); the GPU runs the prefill FFN alone until the engine restarts.");
}

void AneFfn::queue(const Job &job) {
  const ane::Program::Binding &binding = evaluations_.at(job.evaluation).bindings.at(job.set);
  handoff_.queue(job.ready, job.done,
                 [&](const metal::SharedEvent &event, uint64_t wait, uint64_t signal, ane::Handoff::Report report) {
                   program_->enqueue(binding, event, wait, signal, std::move(report));
                 });
}

void AneFfn::queueNow(std::span<const std::pair<uint32_t, uint32_t>> evaluations) {
  uint64_t ready = handoff_.met();
  for (const auto &[evaluation, set] : evaluations) {
    const uint64_t done = handoff_.next();
    queue({evaluation, set, ready, done});
    ready = done;
  }
}

AneFfn::Calibration AneFfn::calibrate(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers,
                                      const PrefillFfnBuffers &ffn, const std::array<metal::MetalBuffer, 2> &hidden) {
  if (const char *reason = unsupported(layers)) throw std::invalid_argument(reason);
  const Linear linear(backend.capabilities());
  // A model of fewer layers times some twice.
  std::vector<SwiGluProjections> sampled;
  for (uint32_t index = 0; index <= kTimedLayers; ++index)
    sampled.push_back(layers[index * (layers.size() - 1) / kTimedLayers]);

  // Inputs of the magnitude of normalized rows; timing does not depend on
  // their values.
  if (auto *data = static_cast<uint16_t *>(ffn.normalized.contents())) {
    std::mt19937 random(20261001);
    std::uniform_real_distribution<float> uniform(-2.0f, 2.0f);
    for (uint64_t i = 0; i < ffn.normalized.sizeBytes() / 2; ++i)
      data[i] = static_cast<uint16_t>(std::bit_cast<uint32_t>(uniform(random)) >> 16);
  }
  for (const metal::MetalBuffer &buffer : {ffn.sums, hidden[0], hidden[1]})
    if (auto *data = static_cast<uint8_t *>(buffer.contents())) std::memset(data, 0, buffer.sizeBytes());

  // The per-layer milliseconds of `run`, which runs a number of layers and
  // returns its milliseconds: the least of two runs of one and of
  // kTimedLayers layers.
  const auto perLayer = [](const auto &run) {
    const auto least = [&](uint32_t count) { return std::min(run(count), run(count)); };
    return (least(kTimedLayers) - least(1)) / (kTimedLayers - 1);
  };
  // The split at `share` timed: the GPU's part of a number of layers in one
  // command buffer, and the ANE's evaluations back to back, the first
  // starting at once.
  const auto measure = [&](double share) {
    AneFfn split(backend, sampled, share, std::array{kMaximumRows});
    const auto parts = [&](bool gpu, bool ane) {
      return [&, gpu, ane](uint32_t count) {
        metal::CommandGraph graph;
        split.begin();
        if (gpu)
          for (uint32_t layer = 0; layer < count; ++layer)
            split.encode(graph, layer, ffn, hidden[layer & 1], hidden[(layer & 1) ^ 1], kMaximumRows, Parts::Gpu);
        std::vector<std::pair<uint32_t, uint32_t>> evaluations;
        for (uint32_t layer = 0; layer < count; ++layer) evaluations.emplace_back(0, layer & 1);
        const auto start = AwakeClock::now();
        if (ane) split.queueNow(evaluations);
        if (gpu) static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
        if (!split.handoff_.finish())
          throw std::runtime_error("ANE FFN evaluations failed (" + split.reason() + ")");
        return millisecondsSince(start);
      };
    };
    // The first command compiles and wires what the others run.
    static_cast<void>(parts(true, true)(kTimedLayers));
    return Timing{share, perLayer(parts(true, false)), perLayer(parts(false, true)), perLayer(parts(true, true))};
  };
  const Model model = fit(measure(kTimedShares[0]), measure(kTimedShares[1]));
  const double gpuMilliseconds = perLayer([&](uint32_t count) {
    metal::CommandGraph graph;
    for (uint32_t layer = 0; layer < count; ++layer)
      linear.addPrefillSwiGlu(graph, sampled[layer], ffn, hidden[layer & 1], hidden[(layer & 1) ^ 1], kMaximumRows);
    const auto start = AwakeClock::now();
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    return millisecondsSince(start);
  });

  const uint32_t units = layers.front().gate->outputSize / kChannelUnit;
  const std::string key = choiceKey(layers);
  const std::optional<uint32_t> gpuUnits = choose(model, gpuMilliseconds, units, ane::recall(key));
  if (!gpuUnits) return {0.0, gpuMilliseconds, gpuMilliseconds};
  const double share = 1.0 - static_cast<double>(*gpuUnits) / units;
  // The split must compute what the GPU computes alone: an ANE that runs but
  // returns other values, as a change to its private client could make it,
  // would otherwise serve wrong logits.
  const double error = splitError(backend, linear, sampled, share, ffn, hidden);
  if (!(error <= kMaximumError))
    throw std::runtime_error("the split's FFN differs from the GPU's by " + std::to_string(error));
  ane::remember(key, *gpuUnits);
  return {share, gpuMilliseconds, model(share), error};
}

double AneFfn::splitError(metal::MetalBackend &backend, const Linear &linear,
                          std::span<const SwiGluProjections> layers, double share, const PrefillFfnBuffers &ffn,
                          const std::array<metal::MetalBuffer, 2> &hidden) {
  // Layer l adds its FFN of the normalized rows to hidden[l & 1] into
  // hidden[(l & 1) ^ 1], from zeros, each way. The affine Q4 kernels read the
  // rows' sums, which the GPU's way computes first.
  const metal::MetalBuffer &result = hidden[layers.size() & 1];
  const auto output = [&] {
    const auto *bits = static_cast<const uint16_t *>(result.contents());
    std::vector<float> values(uint64_t{kMaximumRows} * layers.front().gate->inputSize);
    for (size_t i = 0; i < values.size(); ++i) values[i] = std::bit_cast<float>(uint32_t{bits[i]} << 16);
    return values;
  };
  const auto zero = [&] { std::memset(hidden[0].contents(), 0, hidden[0].sizeBytes()); };
  zero();
  metal::CommandGraph gpu;
  if (layers.front().gate->layout() == WeightLayout::Affine64)
    linear.addPrefillSums(gpu, ffn.normalized, ffn.sums, *layers.front().gate, kMaximumRows);
  for (uint32_t layer = 0; layer < layers.size(); ++layer)
    linear.addPrefillSwiGlu(gpu, layers[layer], ffn, hidden[layer & 1], hidden[(layer & 1) ^ 1], kMaximumRows);
  static_cast<void>(backend.submitCommandAsync(gpu.dispatches()).wait());
  const std::vector<float> expected = output();

  zero();
  AneFfn split(backend, layers, share, std::array{kMaximumRows});
  metal::CommandGraph graph;
  split.begin();
  for (uint32_t layer = 0; layer < layers.size(); ++layer)
    split.add(graph, layer, ffn, hidden[layer & 1], hidden[(layer & 1) ^ 1], kMaximumRows);
  static_cast<void>(split.commit(graph, {}).wait());
  if (!split.completed()) throw std::runtime_error("ANE FFN evaluations failed (" + split.reason() + ")");
  const std::vector<float> got = output();

  double difference = 0.0, magnitude = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const double value = expected[i], delta = got[i] - value;
    difference += delta * delta;
    magnitude += value * value;
  }
  return std::sqrt(difference / magnitude);
}

} // namespace splash::ops
