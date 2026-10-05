#import "ane/Program.hpp"

#include "Checked.hpp"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <dlfcn.h>
#include <sys/qos.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

// The private AppleNeuralEngine interface this file uses.
@protocol SplashAneModel
+ (id)modelAtURL:(NSURL *)url key:(NSString *)key;
- (NSDictionary *)modelAttributes;
@end
@protocol SplashAneClient
+ (id)sharedConnection;
- (BOOL)compileModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)compiledModelExistsFor:(id)model;
- (void)purgeCompiledModel:(id)model;
- (BOOL)loadModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)unloadModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)evaluateWithModel:(id)model options:(NSDictionary *)options request:(id)request qos:(unsigned)qos
                    error:(NSError **)error;
@end
@protocol SplashAneSurface
+ (id)objectWithIOSurface:(IOSurfaceRef)surface;
@end
@protocol SplashAneRequest
+ (id)requestWithInputs:(NSArray *)inputs inputIndices:(NSArray *)inputIndices outputs:(NSArray *)outputs
          outputIndices:(NSArray *)outputIndices weightsBuffer:(id)weights perfStats:(id)stats
         procedureIndex:(NSNumber *)procedure sharedEvents:(id)events transactionHandle:(NSNumber *)transaction;
- (void)setCompletionHandler:(void (^)(BOOL success, NSError *error))handler;
@end
@protocol SplashAneEvents
+ (id)waitEventWithValue:(uint64_t)value sharedEvent:(id)event eventType:(uint64_t)type;
+ (id)signalEventWithValue:(uint64_t)value symbolIndex:(unsigned)symbol eventType:(int64_t)type sharedEvent:(id)event;
+ (id)sharedEventsWithSignalEvents:(NSArray *)signals waitEvents:(NSArray *)waits;
@end

namespace splash::ane {
namespace {

constexpr unsigned kQos = QOS_CLASS_DEFAULT;

Class requireClass(const char *name) {
  static const bool loaded =
      dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
  Class result = loaded ? NSClassFromString(@(name)) : nil;
  if (!result) throw std::runtime_error(std::string("AppleNeuralEngine does not provide ") + name);
  return result;
}

[[noreturn]] void fail(const char *what, NSError *error) {
  throw std::runtime_error(std::string("ANE ") + what + " failed" +
                           (error ? std::string(": ") + error.description.UTF8String : std::string()));
}

id<SplashAneClient> client() { return [(Class<SplashAneClient>)requireClass("_ANEClient") sharedConnection]; }

uint32_t elementBytes(Surface::Element element) noexcept { return element == Surface::Element::Int8 ? 1 : 2; }

// Where compiled programs keep their sources, and the choices beside them.
std::filesystem::path cacheDirectory() { return std::filesystem::temp_directory_path() / "splash-ane-programs"; }

// FNV-1a of `bytes`, continuing `hash`, in hexadecimal: the names of the
// cache's files.
constexpr uint64_t kFnvBasis = 14695981039346656037ULL;
uint64_t fnv1a(std::span<const uint8_t> bytes, uint64_t hash = kFnvBasis) {
  for (const uint8_t byte : bytes) hash = (hash ^ byte) * 1099511628211ULL;
  return hash;
}
std::string hex(uint64_t value) {
  char text[17];
  std::snprintf(text, sizeof text, "%016llx", static_cast<unsigned long long>(value));
  return text;
}
std::span<const uint8_t> bytesOf(std::string_view text) {
  return {reinterpret_cast<const uint8_t *>(text.data()), text.size()};
}

// Writes `file` whole: another process may read it at the same time.
void writeWhole(const std::filesystem::path &file, std::span<const uint8_t> bytes) {
  const std::filesystem::path partial = file.string() + "." + std::to_string(getpid()) + ".partial";
  {
    std::ofstream out(partial, std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("unable to write " + file.string());
  }
  std::filesystem::rename(partial, file);
}

} // namespace

uint32_t Surface::rowBytes(uint32_t width, Element element) noexcept {
  return (width * elementBytes(element) + 63) / 64 * 64;
}

uint64_t Surface::bytes(uint32_t rows, uint32_t width, Element element) noexcept {
  return alignUp(uint64_t{rowBytes(width, element)} * rows);
}

const char *Surface::milType(Element element) noexcept { return element == Element::Int8 ? "int8" : "fp16"; }

std::string Surface::strides(uint32_t rows) const {
  const uint64_t stride = strideBytes / elementBytes(element), plane = uint64_t{rows} * stride;
  return "[" + std::to_string(plane) + ", " + std::to_string(plane) + ", " + std::to_string(stride) + ", 1]";
}

std::string Surface::bufferType(uint32_t rows, uint32_t width) const {
  return std::string("tensor_buffer<") + milType(element) + ", shape=[1, 1, " + std::to_string(rows) + ", " +
         std::to_string(width) + "], strides=" + strides(rows) + ", interleave_factors=[1, 1, 1, 1]>";
}

Surface Surface::create(metal::MetalBackend &backend, uint32_t rows, uint32_t width, Element element) {
  const uint32_t stride = rowBytes(width, element);
  const uint64_t size = bytes(rows, width, element);
  NSDictionary *properties = @{
    (id)kIOSurfaceWidth : @(width),
    (id)kIOSurfaceHeight : @(rows),
    (id)kIOSurfaceBytesPerElement : @(elementBytes(element)),
    (id)kIOSurfaceBytesPerRow : @(stride),
    (id)kIOSurfaceAllocSize : @(size),
    (id)kIOSurfacePixelFormat : @(element == Element::Int8 ? 0x4c303038 : 0x4c303068), // 'L008', 'L00h'
  };
  IOSurfaceRef created = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
  if (!created) throw std::runtime_error("IOSurface creation failed");
  std::shared_ptr<void> owner(created, [](void *surface) { CFRelease(surface); });
  // Programs and kernels address the rows at the stride asked for.
  if (IOSurfaceGetBytesPerRow(created) != stride || IOSurfaceGetAllocSize(created) != size)
    throw std::runtime_error("IOSurface took another row stride or size");
  Surface result;
  result.buffer = backend.wrapSharedMemory(IOSurfaceGetBaseAddress(created), size, owner, "ane surface");
  result.surface = std::move(owner);
  result.element = element;
  result.strideBytes = stride;
  return result;
}

std::vector<uint8_t> constantBlob(std::span<const _Float16> values) {
  constexpr uint64_t kData = 128;
  std::vector<uint8_t> blob(kData + values.size_bytes());
  const auto put = [&](uint64_t offset, auto value) { std::memcpy(blob.data() + offset, &value, sizeof value); };
  put(0, uint32_t{1}); // blobs
  put(4, uint32_t{2}); // storage version
  put(kConstantOffset, uint32_t{0xdeadbeef}); // the record's sentinel
  put(kConstantOffset + 4, uint32_t{1});      // fp16
  put(kConstantOffset + 8, uint64_t{values.size_bytes()});
  put(kConstantOffset + 16, kData);
  std::memcpy(blob.data() + kData, values.data(), values.size_bytes());
  return blob;
}

// A procedure's function, and its inputs and output as the model's symbols:
// their names, and their indices among all procedures', which a request
// takes.
struct Procedure final {
  std::string function;
  std::vector<std::string> inputs;
  std::vector<uint32_t> inputSymbols;
  uint32_t outputSymbol = 0;
};

struct Program::Impl {
  __strong id model = nil;
  // By procedure.
  std::vector<Procedure> procedures;
  bool loaded = false;

  ~Impl() {
    if (loaded) [client() unloadModel:model options:@{} qos:kQos error:nil];
  }
};

Program::Program(std::string_view mil, std::span<const uint8_t> weights) : impl_(std::make_unique<Impl>()) {
  @autoreleasepool {
    // The source's hash names its directory and the service's key; a file of
    // the source's size there already holds it.
    const std::string key = hex(fnv1a(weights, fnv1a(bytesOf(mil))));
    const std::filesystem::path directory = cacheDirectory() / key;
    std::filesystem::create_directories(directory);
    for (const auto &[name, bytes] : {std::pair{"model.mil", bytesOf(mil)}, std::pair{"weights.bin", weights}}) {
      std::error_code missing;
      if (std::filesystem::file_size(directory / name, missing) != bytes.size()) writeWhole(directory / name, bytes);
    }

    impl_->model = [(Class<SplashAneModel>)requireClass("_ANEModel")
        modelAtURL:[NSURL fileURLWithPath:@(directory.c_str()) isDirectory:YES]
               key:@(key.c_str())];
    if (!impl_->model) fail("model creation", nil);
    const auto compile = [&] {
      NSError *error = nil;
      if (![client() compileModel:impl_->model
                          options:@{@"kANEFModelType" : @"kANEFModelMIL", @"kANEFNetPlistFilenameKey" : @"model.mil"}
                              qos:kQos
                            error:&error])
        fail("compilation", error);
    };
    const bool compiled = [client() compiledModelExistsFor:impl_->model];
    if (!compiled) compile();
    NSError *error = nil;
    if (![client() loadModel:impl_->model options:@{} qos:kQos error:&error]) {
      // A compilation the service kept from another compiler, say, compiles
      // again once.
      if (!compiled) fail("load", error);
      [client() purgeCompiledModel:impl_->model];
      compile();
      error = nil;
      if (![client() loadModel:impl_->model options:@{} qos:kQos error:&error]) fail("load", error);
    }
    impl_->loaded = true;
    NSDictionary *description = [impl_->model modelAttributes][@"ANEFModelDescription"];
    NSArray *symbols = description[@"kANEFModelInputSymbolsArrayKey"];
    NSDictionary *functions = description[@"kANEFModelProcedureNameToIDMapKey"];
    NSArray *procedures = description[@"ANEFModelProcedures"];
    if (![symbols isKindOfClass:NSArray.class] || ![functions isKindOfClass:NSDictionary.class] ||
        ![procedures isKindOfClass:NSArray.class] || procedures.count != functions.count)
      throw std::runtime_error("ANE program does not describe its procedures");
    impl_->procedures.resize(procedures.count);
    for (NSString *function in functions) {
      const NSUInteger index = [functions[function] unsignedIntegerValue];
      if (index >= procedures.count) throw std::runtime_error("ANE program names an unknown procedure");
      impl_->procedures[index].function = function.UTF8String;
    }
    for (NSDictionary *entry in procedures) {
      const NSUInteger index = [entry[@"ANEFModelProcedureID"] unsignedIntegerValue];
      NSArray *outputs = entry[@"ANEFModelOutputSymbolIndexArray"];
      if (index >= procedures.count || outputs.count != 1)
        throw std::runtime_error("ANE program has a procedure of other than one output");
      Procedure &procedure = impl_->procedures[index];
      procedure.outputSymbol = [outputs[0] unsignedIntValue];
      for (NSNumber *symbol in entry[@"ANEFModelInputSymbolIndexArray"]) {
        if (symbol.unsignedIntegerValue >= symbols.count)
          throw std::runtime_error("ANE program names an unknown input");
        procedure.inputSymbols.push_back(symbol.unsignedIntValue);
        procedure.inputs.emplace_back([symbols[symbol.unsignedIntegerValue] UTF8String]);
      }
    }
  }
}

Program::~Program() = default;

uint32_t Program::procedure(std::string_view function) const {
  const auto found = std::ranges::find(impl_->procedures, function, &Procedure::function);
  if (found == impl_->procedures.end())
    throw std::invalid_argument("ANE program has no function " + std::string(function));
  return static_cast<uint32_t>(found - impl_->procedures.begin());
}

const std::vector<std::string> &Program::inputs(uint32_t procedure) const {
  return impl_->procedures.at(procedure).inputs;
}

void Program::enqueue(uint32_t procedure, std::span<const Surface> inputs, const Surface &output,
                      const metal::SharedEvent &event, uint64_t wait, uint64_t signal,
                      std::function<void(bool)> done) {
  const Procedure &called = impl_->procedures.at(procedure);
  if (inputs.size() != called.inputs.size()) throw std::invalid_argument("ANE program input count mismatch");
  @autoreleasepool {
    Class<SplashAneSurface> surfaces = (Class<SplashAneSurface>)requireClass("_ANEIOSurfaceObject");
    NSMutableArray *objects = [NSMutableArray arrayWithCapacity:inputs.size()];
    NSMutableArray *indices = [NSMutableArray arrayWithCapacity:inputs.size()];
    for (size_t index = 0; index < inputs.size(); ++index) {
      [objects addObject:[surfaces objectWithIOSurface:(IOSurfaceRef)inputs[index].surface.get()]];
      [indices addObject:@(called.inputSymbols[index])];
    }
    id native = (__bridge id)event.nativeHandle();
    // The service shares the event by its Mach port, which an event Metal's
    // validation layer wraps lacks: it would raise an Objective-C exception.
    if (![native respondsToSelector:NSSelectorFromString(@"eventPort")])
      throw std::runtime_error("the Neural Engine cannot share a Metal event the validation layer wraps");
    id events = [(Class<SplashAneEvents>)requireClass("_ANESharedEvents")
        sharedEventsWithSignalEvents:@[ [(Class<SplashAneEvents>)requireClass("_ANESharedSignalEvent")
                                         signalEventWithValue:signal
                                                  symbolIndex:0
                                                    eventType:0
                                                  sharedEvent:native] ]
                          waitEvents:@[ [(Class<SplashAneEvents>)requireClass("_ANESharedWaitEvent")
                                         waitEventWithValue:wait
                                                sharedEvent:native
                                                  eventType:0] ]];
    id request = [(Class<SplashAneRequest>)requireClass("_ANERequest")
        requestWithInputs:objects
             inputIndices:indices
                  outputs:@[ [surfaces objectWithIOSurface:(IOSurfaceRef)output.surface.get()] ]
            outputIndices:@[ @(called.outputSymbol) ]
            weightsBuffer:nil
                perfStats:nil
           procedureIndex:@(procedure)
             sharedEvents:events
        transactionHandle:nil];
    if (!request) fail("request creation", nil);
    // An evaluation with shared events runs asynchronously and requires a
    // completion handler.
    metal::SharedEvent retained = event;
    [request setCompletionHandler:^(BOOL success, NSError *) {
      if (!success) retained.signal(signal);
      done(success);
    }];
    NSError *error = nil;
    if (![client() evaluateWithModel:impl_->model options:@{} request:request qos:kQos error:&error])
      fail("evaluation", error);
  }
}

namespace {
std::filesystem::path choiceFile(std::string_view key) { return cacheDirectory() / ("choice-" + hex(fnv1a(bytesOf(key)))); }
} // namespace

std::optional<uint32_t> recall(std::string_view key) {
  std::ifstream file(choiceFile(key));
  uint32_t value = 0;
  if (file >> value) return value;
  return std::nullopt;
}

void remember(std::string_view key, uint32_t value) noexcept {
  try {
    std::filesystem::create_directories(cacheDirectory());
    writeWhole(choiceFile(key), bytesOf(std::to_string(value)));
  } catch (const std::exception &) {
    // The next start chooses afresh.
  }
}

} // namespace splash::ane
