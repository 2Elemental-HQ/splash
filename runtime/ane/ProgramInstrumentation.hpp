#pragma once

#include "ane/Program.hpp"

#include <chrono>
#include <cstdint>
#include <string>

namespace splash::ane {

// Faults tests inject into Programs, to check how their users meet the
// service's failures. Only the instrumented build of Program.mm
// (SPLASH_ANE_INSTRUMENTATION) defines this; production binaries never link
// it.
class ProgramInstrumentation final {
public:
  // A program's faults. Its enqueue() calls count from 1, and an evaluation
  // takes the count of the call that queues it; 0 names none.
  struct Faults final {
    // Construction throws std::runtime_error from the service's queue, as a
    // failed compilation does.
    bool construction = false;
    // Construction checks the framework again with the expected type encoding
    // of this method, as "-[_ANEClient loadModel:options:qos:error:]", made
    // one no method has.
    std::string changedMethod;
    // guarded() raises an NSException at the step of this name, as "load" or
    // "evaluation" (Program.mm names its steps).
    std::string raiseAt;
    // The enqueue() call that throws.
    uint64_t throwingEnqueue = 0;
    // The evaluation that reports failure without running, the one that
    // never runs nor reports (`done` is never called and its event never
    // raised), and the one that runs `delay` late.
    uint64_t failingEvaluation = 0, stalledEvaluation = 0, delayedEvaluation = 0;
    std::chrono::milliseconds delay{0};
    // The evaluation that does not run but, once its event reaches its wait,
    // writes an fp16 infinity as its output's first value, raises the event
    // and reports success, as one whose output overflowed would.
    uint64_t poisonedEvaluation = 0;
  };
  // The faults of the next Program this process constructs, for its life.
  // Programs constructed after it take none until faults are armed again.
  static void arm(Faults faults);
};

} // namespace splash::ane
