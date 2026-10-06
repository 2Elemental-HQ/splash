// How the prefill FFN's Neural Engine split is calibrated (ops/AneFfnCalibration.cpp), from timings it is given: the
// model's fit, the share it chooses, the timings it takes and the least chunk it splits.

#include "TestChecks.hpp"
#include "ops/AneFfnCalibration.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace splash::ops::ane_ffn;
using splash::test::rejects;
using splash::test::require;

bool near(double got, double expected) { return std::abs(got - expected) <= 1e-9 * std::max(1.0, std::abs(expected)); }

// Timings of parts that do not bind each other's bandwidth: both together take the longer alone.
Timing apart(double share, double gpu, double ane) { return {share, gpu, ane, std::max(gpu, ane)}; }

// G(s) = 20 - 20 s and A(s) = 40 s cross at s = 1/3, where max(G, A) is least: of 10 units, 3 (G 14, A 12) beat 4
// (G 12, A 16).
void testCrossingLines() {
  const Model model = fit(apart(0.3, 14, 12), apart(0.8, 4, 32));
  require(near(model.gpu(0.0), 20) && near(model.gpu.slope, -20) && near(model.ane(0.0), 0) &&
              near(model.ane.slope, 40) && model.uG == 0.0 && model.uA == 0.0,
          "lines through the timings of parts apart");
  require(choose(model, 20, 10, 9, std::nullopt) == 3u, "the crossing's units were not chosen");
  // The least units within 1% of the best: of 100 units, 33 (max 13.4) beat 34 (13.6, 1.5% more).
  require(choose(model, 20, 100, 99, std::nullopt) == 33u, "the least units near the crossing were not chosen");
  rejects([] { static_cast<void>(fit(apart(0.5, 10, 10), apart(0.5, 8, 12))); }, "needs two shares",
          "a fit of one share");
  for (const uint32_t most : {0u, 10u})
    rejects([&] { static_cast<void>(choose(model, 20, 10, most, std::nullopt)); }, "no feasible units",
            "a choice of " + std::to_string(most) + " units of 10");
}

// Where both parts together take longer than either alone at both shares, the fit solves for each part's fraction
// of the bandwidth: here uG 0.8 and uA 0.5, and T(s) the bandwidth's term where it binds.
void testSolvedFit() {
  const Timing p{0.3, 10, 8, 12}, q{0.6, 8, 10, 11.4};
  const Model model = fit(p, q);
  require(near(model.uG, 0.8) && near(model.uA, 0.5), "the bandwidth's fractions were not solved");
  require(near(model(0.3), 12) && near(model(0.6), 11.4), "T is not the bandwidth's term where it binds");
}

// Where both together take longer at one share only, the GPU's part takes all the bandwidth and the ANE's the excess
// over it, at most all.
void testOneShareBinds() {
  Model model = fit({0.3, 10, 6, 13}, apart(0.6, 5, 12));
  require(model.uG == 1.0 && near(model.uA, 0.5), "one share's excess was not the ANE's fraction");
  model = fit({0.3, 10, 2, 13}, apart(0.6, 5, 12));
  require(model.uG == 1.0 && model.uA == 1.0, "the ANE's fraction was not clamped to all the bandwidth");
}

// Timings whose lines fall below zero, or are not numbers, choose nothing, and the choice ends.
void testUnusableLines() {
  const Model falling = fit(apart(0.5, 1, 1), apart(0.6, 0.1, 0.1));
  require(falling.gpu(0.9) < 0.0, "the lines do not fall below zero");
  require(!choose(falling, 20, 34, 33, std::nullopt), "lines below zero chose units");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  require(!choose(fit(apart(0.3, nan, 12), apart(0.8, 4, 32)), 20, 10, 9, std::nullopt), "lines of NaN chose units");
}

// Lines of one sign: both parts slower with more ANE units choose the fewest, both faster the most.
void testSlopesOfOneSign() {
  require(choose(fit(apart(0.2, 7, 6), apart(0.6, 11, 14)), 20, 10, 9, std::nullopt) == 1u,
          "rising lines did not choose one unit");
  require(choose(fit(apart(0.2, 9, 7.2), apart(0.6, 7, 5.6)), 20, 10, 6, std::nullopt) == 6u,
          "falling lines did not choose the most units the plan holds");
}

// The last start's units stay while their max(G, A) is within 3% of the best, among the units the plan holds.
void testPrevious() {
  const Model model = fit(apart(0.3, 14, 12), apart(0.8, 4, 32));
  require(choose(model, 20, 100, 99, 34u) == 34u, "units 1.5% from the best did not stay");
  require(choose(model, 20, 100, 99, 35u) == 33u, "units 4.5% from the best stayed");
  require(choose(model, 20, 100, 99, 32u) == 32u && choose(model, 20, 100, 99, 30u) == 33u,
          "units below the best stayed within 3% of it, or further");
  for (const uint32_t previous : {0u, 100u, 250u})
    require(choose(model, 20, 100, 99, previous) == 33u, std::to_string(previous) + " units of 100 stayed");
  require(choose(model, 20, 100, 30, 34u) == 30u, "units the plan does not hold stayed");
  // G(s) = 16 - 10 s and A(s) = 4 + 30 s put the best at 10 of 34 units (13.06 ms), which gains 5% on 14 ms; the
  // last start's 9 (13.35, 2.3% slower) gains less and gives way to 10.
  const Model lines = fit(apart(14.0 / 34, 16 - 10 * 14.0 / 34, 4 + 30 * 14.0 / 34),
                          apart(27.0 / 34, 16 - 10 * 27.0 / 34, 4 + 30 * 27.0 / 34));
  require(choose(lines, 14, 34, 33, 9u) == 10u, "units within 3% of the best that gain too little stayed");
  require(choose(lines, 20, 34, 33, 9u) == 9u, "units within 3% of the best that gain did not stay");
}

// The split runs only where T is predicted at least 5% below the GPU alone: 14 against 14.5 is not, and the GPU's
// timing not a number proves nothing.
void testNoGain() {
  const Model model = fit(apart(0.3, 14, 12), apart(0.8, 4, 32));
  require(choose(model, 14.8, 10, 9, std::nullopt) == 3u, "a 5.4% gain was refused");
  require(!choose(model, 14.5, 10, 9, std::nullopt), "a 3.4% gain was taken");
  require(!choose(model, std::numeric_limits<double>::quiet_NaN(), 10, 9, std::nullopt),
          "a gain over a timing not a number was taken");
}

// A measured duration is usable only finite and positive.
void testUsable() {
  require(usable(1.5) == 1.5, "a duration was changed");
  for (const double milliseconds : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::infinity()})
    rejects([&] { static_cast<void>(usable(milliseconds)); }, "calibration timings are not usable",
            std::to_string(milliseconds) + " ms was usable");
}

// The units nearest a share, from one to all but one, and no share outside (0, 1) or of fewer than two units.
void testNearestUnits() {
  require(nearestUnits(0.4, 34) == 14 && nearestUnits(0.8, 34) == 27 && nearestUnits(0.01, 34) == 1 &&
              nearestUnits(0.99, 34) == 33,
          "the units nearest a share");
  for (const auto &[share, units] : std::vector<std::pair<double, uint32_t>>{
           {0.0, 34}, {1.0, 34}, {std::numeric_limits<double>::quiet_NaN(), 34}, {0.5, 1}})
    rejects([&] { static_cast<void>(nearestUnits(share, units)); }, "must lie in (0, 1)",
            "a share of " + std::to_string(share) + " of " + std::to_string(units) + " units");
}

// The timings' shares are the split's units over all: on an M5 Max's timings of Qwen3.8-27B's 34 units, at 14 and 27
// units, 9 units, where G meets A; taken at the nominal 0.4 and 0.8 they would put it a unit lower.
void testRealizedShares() {
  const Timing low{14.0 / 34, 11.725, 22.287, 22.173}, high{27.0 / 34, 5.094, 43.802, 43.794};
  require(choose(fit(low, high), 18.951, 34, 33, std::nullopt) == 9u, "the realized shares did not choose 9 units");
  Timing nominalLow = low, nominalHigh = high;
  nominalLow.share = 0.4;
  nominalHigh.share = 0.8;
  require(choose(fit(nominalLow, nominalHigh), 18.951, 34, 33, std::nullopt) == 8u,
          "the nominal shares did not choose 8 units");
}

// The GPU alone's layer on the line through 5 ms at 512 rows and 20 at 2048.
double gpuAt(uint32_t rows) { return 5.0 + (rows - 512) * 15.0 / 1536; }

// Functions every 128 rows from 2048 down to 512, each running the chunks above the one below it (the least, those
// of 512 rows): each function's split layer at `ratio` of the GPU's at its own rows, but where `split` says
// otherwise.
ChunkTimings chunks(double ratio, const std::vector<std::pair<uint32_t, double>> &split = {}) {
  ChunkTimings timings;
  timings.gpu = {{{512, gpuAt(512)}, {2048, gpuAt(2048)}}};
  for (uint32_t rows = 2048; rows >= 512; rows -= 128) {
    double milliseconds = ratio * gpuAt(rows);
    for (const auto &[function, given] : split)
      if (function == rows) milliseconds = given;
    timings.functions.push_back({rows == 512 ? 512 : rows - 127, rows, milliseconds});
  }
  return timings;
}

// A chunk splits while its function's split layer takes at most 2% longer than the GPU alone at the chunk's rows,
// each row count checked; the least chunk is the one from which every larger one splits.
void testMinimumRows() {
  require(minimumRows(chunks(0.5)) == 512u, "a split of every function twice as fast held back a chunk");
  require(!minimumRows(chunks(0.8, {{2048, 21.0}})), "a split slower at the most rows split a chunk");
  // Function 640 at 5.5 ms: a batch of a 512-row chunk and another lane's row, 513 rows, runs it slower than the GPU
  // alone (5.01 ms); its chunks split from 553 rows (1.02 x 5.40 ms), and not from 552 (1.02 x 5.39).
  require(minimumRows(chunks(0.8, {{640, 5.5}})) == 553u, "the least chunk of a function was not found by its rows");
  // Function 896 slower than the GPU at each of its chunks holds back every chunk below it.
  require(minimumRows(chunks(0.8, {{896, 9.7}})) == 897u, "a function that does not split split");
  // Within 2%: function 2048 as slow as the GPU at 2048 rows splits chunks of 2008 rows, whose GPU layer takes 1.95%
  // less, and not of 2007; 2.5% slower than the GPU it splits none.
  require(minimumRows(chunks(0.8, {{2048, 20.0}})) == 2008u && !minimumRows(chunks(0.8, {{2048, 20.5}})),
          "the margin is not 2%");
  // The functions timed from the most rows down to the first that does not split all its chunks give the least.
  ChunkTimings timed = chunks(0.8, {{896, 9.7}});
  timed.functions.resize(10);
  require(timed.functions.back().rows == 896 && minimumRows(timed) == 897u, "the functions timed did not suffice");
  timed.functions.resize(3);
  require(minimumRows(timed) == timed.functions.back().least, "functions that split all their chunks held some back");
  ChunkTimings ascending = chunks(0.5), gap = chunks(0.5), reversed = chunks(0.5), empty = chunks(0.5);
  std::ranges::reverse(ascending.functions);
  gap.functions.erase(gap.functions.begin() + 1);
  std::ranges::reverse(reversed.gpu);
  empty.functions.clear();
  ChunkTimings none = chunks(0.5), over = chunks(0.5);
  none.functions.back().least = 0;
  over.functions.back().least = over.functions.back().rows + 1;
  for (const ChunkTimings &timings : {ascending, gap, reversed, empty, none, over})
    rejects([&] { static_cast<void>(minimumRows(timings)); }, "not those of its functions",
            "chunks of no functions were ruled");
}

} // namespace

int main() {
  try {
    testCrossingLines();
    testSolvedFit();
    testOneShareBinds();
    testUnusableLines();
    testSlopesOfOneSign();
    testPrevious();
    testNoGain();
    testUsable();
    testNearestUnits();
    testRealizedShares();
    testMinimumRows();
    std::cout << "ane ffn calibration tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "ane ffn calibration tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
