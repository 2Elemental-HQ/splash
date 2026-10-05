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
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <variant>

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

[[noreturn]] void fail(const char *what, NSError *error) {
  throw std::runtime_error(std::string("ANE ") + what + " failed" +
                           (error ? std::string(": ") + error.description.UTF8String : std::string()));
}

// Runs `body`, which messages the private interface, and throws an
// Objective-C exception it raises as a std::runtime_error naming `step`:
// std::exception handlers do not catch an NSException.
template <class F> decltype(auto) guarded(const char *step, F &&body) {
  @try {
    return std::forward<F>(body)();
  } @catch (NSException *exception) {
    throw std::runtime_error(
        [NSString stringWithFormat:@"ANE %s raised %@: %@", step, exception.name, exception.reason].UTF8String);
  }
}

// The private interface's classes and the service's client.
struct Api final {
  __strong id<SplashAneClient> client = nil;
  Class<SplashAneModel> model = nil;
  Class<SplashAneSurface> surface = nil;
  Class<SplashAneEvents> events = nil, signalEvent = nil, waitEvent = nil;
  Class<SplashAneRequest> request = nil;
};

// Loads the framework and finds each class this file uses and the shared
// client, checking that each responds to every selector this file sends it.
// Throws at the first failure.
Api resolve() {
  if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
    const char *reason = dlerror();
    throw std::runtime_error(std::string("AppleNeuralEngine does not load: ") + (reason ? reason : "no reason"));
  }
  const auto lacks = [](const std::string &what) { return std::runtime_error("AppleNeuralEngine lacks " + what); };
  // +[class selector] for a class method, -[class selector] for an instance
  // method.
  const auto method = [](char kind, const char *name, SEL selector) {
    return std::string(1, kind) + "[" + name + " " + sel_getName(selector) + "]";
  };
  const auto require = [&](const char *name, std::initializer_list<SEL> classMethods,
                           std::initializer_list<SEL> instanceMethods) {
    Class found = NSClassFromString(@(name));
    if (!found) throw lacks(name);
    for (const SEL selector : classMethods)
      if (![found respondsToSelector:selector]) throw lacks(method('+', name, selector));
    for (const SEL selector : instanceMethods)
      if (![found instancesRespondToSelector:selector]) throw lacks(method('-', name, selector));
    return found;
  };
  Api api;
  api.model = require("_ANEModel", {@selector(modelAtURL:key:)}, {@selector(modelAttributes)});
  api.surface = require("_ANEIOSurfaceObject", {@selector(objectWithIOSurface:)}, {});
  api.events = require("_ANESharedEvents", {@selector(sharedEventsWithSignalEvents:waitEvents:)}, {});
  api.signalEvent =
      require("_ANESharedSignalEvent", {@selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:)}, {});
  api.waitEvent = require("_ANESharedWaitEvent", {@selector(waitEventWithValue:sharedEvent:eventType:)}, {});
  api.request = require("_ANERequest",
                        {@selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:
                                   procedureIndex:sharedEvents:transactionHandle:)},
                        {@selector(setCompletionHandler:)});
  // The client's instance methods, on the client the service shares.
  id client = [(Class<SplashAneClient>)require("_ANEClient", {@selector(sharedConnection)}, {}) sharedConnection];
  if (!client) throw lacks("a shared connection");
  for (const SEL selector : {@selector(compileModel:options:qos:error:), @selector(compiledModelExistsFor:),
                             @selector(purgeCompiledModel:), @selector(loadModel:options:qos:error:),
                             @selector(unloadModel:options:qos:error:),
                             @selector(evaluateWithModel:options:request:qos:error:)})
    if (![client respondsToSelector:selector]) throw lacks(method('-', "_ANEClient", selector));
  api.client = client;
  return api;
}

// The private interface, resolved on first use. A resolution that fails is
// not retried: each use throws its message.
const Api &api() {
  static const std::variant<Api, std::string> resolved = []() -> std::variant<Api, std::string> {
    @autoreleasepool {
      try {
        return guarded("interface lookup", resolve);
      } catch (const std::exception &error) {
        return std::string(error.what());
      }
    }
  }();
  if (const std::string *failure = std::get_if<std::string>(&resolved)) throw std::runtime_error(*failure);
  return std::get<Api>(resolved);
}

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

// Whether `file` holds `bytes` and nothing else.
bool holds(const std::filesystem::path &file, std::span<const uint8_t> bytes) {
  std::ifstream in(file, std::ios::binary | std::ios::ate);
  if (!in || static_cast<uint64_t>(in.tellg()) != bytes.size()) return false;
  std::vector<uint8_t> contents(bytes.size());
  in.seekg(0);
  return in.read(reinterpret_cast<char *>(contents.data()), static_cast<std::streamsize>(contents.size())) &&
         std::ranges::equal(contents, bytes);
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
  explicit Impl(const Api &api) : api(api) {}

  const Api &api;
  __strong id model = nil;
  // By procedure.
  std::vector<Procedure> procedures;
  bool loaded = false;

  ~Impl() {
    if (!loaded) return;
    // A destructor does not throw: an unload that fails is ignored.
    try {
      guarded("unload", [&] { [api.client unloadModel:model options:@{} qos:kQos error:nil]; });
    } catch (...) {
    }
  }
};

Program::Program(std::string_view mil, std::span<const uint8_t> weights) : impl_(std::make_unique<Impl>(api())) {
  const Api &api = impl_->api;
  @autoreleasepool {
    // The source's hash names its directory and the service's key; a file
    // there that does not hold the source's bytes is written again.
    const std::string key = hex(fnv1a(weights, fnv1a(bytesOf(mil))));
    const std::filesystem::path directory = cacheDirectory() / key;
    std::filesystem::create_directories(directory);
    for (const auto &[name, bytes] : {std::pair{"model.mil", bytesOf(mil)}, std::pair{"weights.bin", weights}})
      if (!holds(directory / name, bytes)) writeWhole(directory / name, bytes);

    impl_->model = guarded("model creation", [&] {
      return [api.model modelAtURL:[NSURL fileURLWithPath:@(directory.c_str()) isDirectory:YES] key:@(key.c_str())];
    });
    if (!impl_->model) fail("model creation", nil);
    const auto compile = [&] {
      guarded("compilation", [&] {
        NSDictionary *options = @{@"kANEFModelType" : @"kANEFModelMIL", @"kANEFNetPlistFilenameKey" : @"model.mil"};
        NSError *error = nil;
        if (![api.client compileModel:impl_->model options:options qos:kQos error:&error]) fail("compilation", error);
      });
    };
    const bool compiled =
        guarded("compiled model lookup", [&] { return [api.client compiledModelExistsFor:impl_->model]; });
    if (!compiled) compile();
    guarded("load", [&] {
      NSError *error = nil;
      if ([api.client loadModel:impl_->model options:@{} qos:kQos error:&error]) return;
      // A compilation the service kept from another compiler, say, compiles
      // again once.
      if (!compiled) fail("load", error);
      guarded("purge", [&] { [api.client purgeCompiledModel:impl_->model]; });
      compile();
      error = nil;
      if (![api.client loadModel:impl_->model options:@{} qos:kQos error:&error]) fail("load", error);
    });
    impl_->loaded = true;
    // Each value of the attributes is checked for its kind before it is
    // messaged.
    guarded("model attributes", [&] {
      const auto undescribed = [] { return std::runtime_error("ANE program does not describe its procedures"); };
      NSDictionary *attributes = [impl_->model modelAttributes];
      if (![attributes isKindOfClass:NSDictionary.class]) throw undescribed();
      NSDictionary *description = attributes[@"ANEFModelDescription"];
      if (![description isKindOfClass:NSDictionary.class]) throw undescribed();
      NSArray *symbols = description[@"kANEFModelInputSymbolsArrayKey"];
      NSDictionary *functions = description[@"kANEFModelProcedureNameToIDMapKey"];
      NSArray *procedures = description[@"ANEFModelProcedures"];
      if (![symbols isKindOfClass:NSArray.class] || ![functions isKindOfClass:NSDictionary.class] ||
          ![procedures isKindOfClass:NSArray.class] || procedures.count != functions.count)
        throw undescribed();
      impl_->procedures.resize(procedures.count);
      for (NSString *function in functions) {
        NSNumber *index = functions[function];
        if (![function isKindOfClass:NSString.class] || ![index isKindOfClass:NSNumber.class]) throw undescribed();
        if (index.unsignedIntegerValue >= procedures.count)
          throw std::runtime_error("ANE program names an unknown procedure");
        impl_->procedures[index.unsignedIntegerValue].function = function.UTF8String;
      }
      for (NSDictionary *entry in procedures) {
        if (![entry isKindOfClass:NSDictionary.class]) throw undescribed();
        NSNumber *index = entry[@"ANEFModelProcedureID"];
        NSArray *outputs = entry[@"ANEFModelOutputSymbolIndexArray"];
        NSArray *inputs = entry[@"ANEFModelInputSymbolIndexArray"];
        if (![index isKindOfClass:NSNumber.class] || ![outputs isKindOfClass:NSArray.class] ||
            ![inputs isKindOfClass:NSArray.class])
          throw undescribed();
        if (index.unsignedIntegerValue >= procedures.count || outputs.count != 1)
          throw std::runtime_error("ANE program has a procedure of other than one output");
        NSNumber *output = outputs[0];
        if (![output isKindOfClass:NSNumber.class]) throw undescribed();
        Procedure &procedure = impl_->procedures[index.unsignedIntegerValue];
        procedure.outputSymbol = output.unsignedIntValue;
        for (NSNumber *symbol in inputs) {
          if (![symbol isKindOfClass:NSNumber.class]) throw undescribed();
          if (symbol.unsignedIntegerValue >= symbols.count)
            throw std::runtime_error("ANE program names an unknown input");
          NSString *input = symbols[symbol.unsignedIntegerValue];
          if (![input isKindOfClass:NSString.class]) throw undescribed();
          procedure.inputSymbols.push_back(symbol.unsignedIntValue);
          procedure.inputs.emplace_back(input.UTF8String);
        }
      }
    });
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
  const Api &api = impl_->api;
  @autoreleasepool {
    const auto object = [&](const Surface &surface) {
      return guarded("surface object creation", [&] {
        id result = [api.surface objectWithIOSurface:(IOSurfaceRef)surface.surface.get()];
        if (!result) fail("surface object creation", nil);
        return result;
      });
    };
    NSMutableArray *objects = [NSMutableArray arrayWithCapacity:inputs.size()];
    NSMutableArray *indices = [NSMutableArray arrayWithCapacity:inputs.size()];
    for (size_t index = 0; index < inputs.size(); ++index) {
      [objects addObject:object(inputs[index])];
      [indices addObject:@(called.inputSymbols[index])];
    }
    id native = (__bridge id)event.nativeHandle();
    // The service shares the event by its Mach port, which an event Metal's
    // validation layer wraps lacks: it would raise an Objective-C exception.
    if (![native respondsToSelector:NSSelectorFromString(@"eventPort")])
      throw std::runtime_error("the Neural Engine cannot share a Metal event the validation layer wraps");
    id events = guarded("event creation", [&] {
      id signalEvent = [api.signalEvent signalEventWithValue:signal symbolIndex:0 eventType:0 sharedEvent:native];
      id waitEvent = [api.waitEvent waitEventWithValue:wait sharedEvent:native eventType:0];
      if (!signalEvent || !waitEvent) fail("event creation", nil);
      id result = [api.events sharedEventsWithSignalEvents:@[ signalEvent ] waitEvents:@[ waitEvent ]];
      if (!result) fail("event creation", nil);
      return result;
    });
    id outputObject = object(output);
    id request = guarded("request creation", [&] {
      id result = [api.request requestWithInputs:objects
                                    inputIndices:indices
                                         outputs:@[ outputObject ]
                                   outputIndices:@[ @(called.outputSymbol) ]
                                   weightsBuffer:nil
                                       perfStats:nil
                                  procedureIndex:@(procedure)
                                    sharedEvents:events
                               transactionHandle:nil];
      if (!result) fail("request creation", nil);
      return result;
    });
    // An evaluation with shared events runs asynchronously and requires a
    // completion handler, which only reports the evaluation's result. It is
    // built outside the lambda below so that it holds a copy of `done`: a
    // block in a lambda holds the lambda's reference instead.
    void (^completion)(BOOL, NSError *) = ^(BOOL success, NSError *) {
      if (done) done(success);
    };
    guarded("completion handler", [&] { [request setCompletionHandler:completion]; });
    guarded("evaluation", [&] {
      NSError *error = nil;
      if (![api.client evaluateWithModel:impl_->model options:@{} request:request qos:kQos error:&error])
        fail("evaluation", error);
    });
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
