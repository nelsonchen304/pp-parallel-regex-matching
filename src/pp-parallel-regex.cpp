#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include "advanced_cuda.hpp"
#include "basic_cuda.hpp"
#include "enumeration_omp.hpp"
#include "framework.hpp"
#include "pararegex_omp.hpp"
#include "serial.hpp"

DFA dfa;

std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
    end_time;

void start_timer() { start_time = std::chrono::high_resolution_clock::now(); }

double stop_timer() {
  end_time = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(end_time - start_time)
      .count();
}

void usage(const char *const name) {
  std::cout
      << "Usage: " << name
      << " [-t <thread count>] [-c <pcap packet count>] [-l <pcap "
         "length>] [-r <num runs>] [-o true] [-h] <dfa file> <pcap folder>\n";
  std::cout << "Options:\n";
  std::cout << "  -t <thread count>: Set number of threads (default: 4)\n";
  std::cout << "  -c <pcap packet count>: Set maximum number of packets to "
               "parse; conflict with -l (default: 3)\n";
  std::cout << "  -l <pcap length>: Set minimum number of bytes to parse; "
               "conflict with -c\n";
  std::cout << "  -r <num runs>: Set number of times to run each matcher; "
               "minimum time is used to evaluate (default: 3)\n";
  std::cout << "  -o: Run only the OpenMP matcher, else run only CUDA "
               "matcher(default: false)\n";
  std::cout << "  -v: Verbose output (default: false)\n";
  std::cout << "  -h: Print this help message and exit\n";
}

void set_arg(size_t &arg, const char *const optarg, const char *const name) {
  try {
    arg = std::stoll(optarg);
  } catch (std::invalid_argument &) {
    std::cerr << "invalid argument for option: " << optarg << "\n";
    usage(name);
    exit(1);
  }
}

int main(int argc, char **argv) {
  size_t thread_count = 4;
  InputLimit limit(InputLimit::Type::count, 3);
  size_t num_runs = 3;
  int opt;
  bool c_present = false, l_present = false, omp_only = false, verbose = false;

  while ((opt = getopt(argc, argv, "t:c:r:l:ovh")) != -1) {
    switch (opt) {
    case 't':
      set_arg(thread_count, optarg, argv[0]);
      break;
    case 'c':
      if (l_present) {
        std::cerr << "cannot specify -c and -l at the same time\n";
        usage(argv[0]);
        exit(1);
      }
      c_present = true;
      set_arg(limit.value, optarg, argv[0]);
      break;
    case 'r':
      set_arg(num_runs, optarg, argv[0]);
      break;
    case 'l':
      if (c_present) {
        std::cerr << "cannot specify -c and -l at the same time\n";
        usage(argv[0]);
        exit(1);
      }
      l_present = true;
      limit.type = InputLimit::Type::length;
      set_arg(limit.value, optarg, argv[0]);
      break;
    case 'o':
      omp_only = true;
      break;
    case 'v':
      verbose = true;
      break;
    case 'h':
      usage(argv[0]);
      exit(0);
      break;
    default:
      usage(argv[0]);
      exit(1);
    }
  }
  if (optind > argc - 2) {
    std::cerr << "Must specify DFA and PCAP files\n";
    usage(argv[0]);
    exit(1);
  }

  const std::string dfa_file = argv[optind];
  const std::string traffic_folder = argv[optind + 1];

  dfa_from_file(dfa_file, dfa);
  if (verbose) {
    std::cout << "DFA loaded with " << dfa.size << " states \n";
  }
  Testcase testcase;
  size_t actual_count =
      testcase_from_pcap_folder(traffic_folder, testcase, limit);
  if (verbose) {
    std::cout << "Testcase loaded with length " << testcase.input.size() << "("
              << actual_count << " packets)\n";
  }

  SerialMatcher serial_matcher(std::make_shared<const DFA>(dfa));
  if (verbose) {
    std::cout << "Running serial Matcher . . . ";
  }
  double baseline_time = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < num_runs; ++i) {
    start_timer();
    bool result = serial_matcher.match(testcase.input);
    testcase.expected = result;
    baseline_time = std::min(baseline_time, stop_timer());
  }
  if (verbose) {
    std::cout << baseline_time << "ms\n";
  }

  /* initialize matchers */
  std::vector<std::unique_ptr<Matcher>> matchers;
  if (omp_only) {
    matchers.emplace_back(std::make_unique<EnumerationMatcher>(
        std::make_shared<const DFA>(dfa), thread_count));
    matchers.emplace_back(std::make_unique<OmpParaMatcher>(
        std::make_shared<const DFA>(dfa), thread_count));
  } else {
    matchers.emplace_back(std::make_unique<EnumerationCUDAMatcher>(
        std::make_shared<const DFA>(dfa)));
    matchers.emplace_back(std::make_unique<AdvancedCudaMatcher>(
        std::make_shared<const DFA>(dfa), false));
    matchers.emplace_back(std::make_unique<AdvancedCudaMatcher>(
        std::make_shared<const DFA>(dfa), true));
  }
  if (verbose) {
    std::cout << "Running " << matchers.size() << " matchers with "
              << thread_count << " threads\n";
  }

  for (auto &matcher : matchers) {
    if (verbose) {
      std::cout << "====== Matcher " << matcher->name() << " ======\n";
    }
    bool passed = true;

    double matcher_time = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < num_runs; ++i) {
      start_timer();
      bool result = matcher->match(testcase.input);
      if (result != testcase.expected) {
        std::cerr << matcher->name() << " failed to match!\n";
        passed = false;
        break;
      }
      matcher_time = std::min(matcher_time, stop_timer());
    }

    if (passed) {
      double speedup = baseline_time / matcher_time;
      if (verbose) {
        std::cout << matcher->name() << " passed with time " << matcher_time
                  << "ms (" << speedup << "x)!\n";
      } else {
        std::cout << matcher->name() << ":::" << speedup << "\n";
      }
    } else {
      std::cout << matcher->name() << " failed (skip timing)!\n";
    }
  }
  return 0;
}
