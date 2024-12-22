#include <filesystem>
#include <fstream>
#include <iostream>
#include <pcap.h>
#include <pcap/pcap.h>
#include <set>

#include "framework.hpp"

namespace fs = std::filesystem;

void dfa_from_file(const std::string &filename, DFA &dfa) {
  std::ifstream infile(filename);
  std::string line;
  std::string currentSection;

  while (getline(infile, line)) {
    if (line.empty()) {
      continue;
    }

    if (line[0] == '#') {
      currentSection = line.substr(1);
      DEBUG_MSG("Section: " << currentSection << "\n");
      continue;
    }

    if (currentSection == "states") {
      dfa.size++;
    } else if (currentSection == "initial") {
      if (line.compare("s0") != 0) {
        std::cerr << "Unexpected initial state; should be s0\n";
      }
      continue;
    } else if (currentSection == "accepting") {
      DEBUG_MSG(line.substr(1) << "\n");
      int stateNum;
      try {
        stateNum = std::stoi(line.substr(1));
        dfa.accept_state.set(stateNum);
      } catch (std::exception &e) {
        std::cerr << "Failed to parse state number in accepting: "
                  << line.substr(1) << std::endl;
      }
    } else if (currentSection == "alphabet") {
      if (line.substr(0, 2) == "0x") {
        try {
          int hex = std::stoi(line.substr(2), nullptr, 16);
          dfa.used_alphabet.set(hex);
        } catch (std::exception &e) {
          std::cerr << "Failed to parse hex number in alphabet: "
                    << line.substr(2) << std::endl;
        }
      } else {
        dfa.used_alphabet.set(line[0]);
      }
    } else if (currentSection == "transitions") {
      size_t colon = line.find(':');
      size_t arrow = line.rfind('>');
      std::string fromState = line.substr(0, colon);
      std::string alphabet = line.substr(colon + 1, arrow - colon - 1);
      std::string toState = line.substr(arrow + 1);
      // DEBUG_MSG(fromState << " " << alphabet << " " << toState << "\n");
      int from = -1, to = -1, c = -1;
      bool ok = true;
      try {
        from = std::stoi(fromState.substr(1));
      } catch (std::exception &e) {
        std::cerr << "Failed to parse source state number in transitions: "
                  << line << "(" << fromState.substr(1) << ")\n";
        ok = false;
      }
      try {
        to = std::stoi(toState.substr(1));
      } catch (std::exception &e) {
        std::cerr << "Failed to parse destination state number in transitions: "
                  << line << "(" << toState.substr(1) << ")\n";
        ok = false;
      }
      if (alphabet.substr(0, 2) == "0x") {
        try {
          c = std::stoi(alphabet.substr(2), nullptr, 16);
        } catch (std::exception &e) {
          std::cerr << "Failed to parse alphabet hex number in alphabet: "
                    << line << "(" << alphabet.substr(2) << ")\n";
          ok = false;
        }
      } else {
        c = alphabet[0];
      }
      if (ok) {
        dfa.transitions[from][c] = to;
      }
    }
  }
}

struct HandlerArg {
  Testcase &testcase;
  size_t limit;
  pcap_t *handle;
  size_t &actual_count;
};

void packet_handler(unsigned char *p_arg,
                    const struct pcap_pkthdr *packet_header,
                    const unsigned char *packet_data) {
  HandlerArg *arg = (HandlerArg *)p_arg;
  if (arg->limit > 0 && arg->testcase.input.size() >= arg->limit) {
    pcap_breakloop(arg->handle);
    return;
  }
  DEBUG_MSG("#" << std::dec << std::setw(4) << std::setfill('0')
                << arg->actual_count << " " << packet_header->ts.tv_sec << "."
                << packet_header->ts.tv_usec << " (" << packet_header->caplen
                << ")");
  ustring packet(packet_data, packet_header->caplen);

  for (size_t i = 0; i < packet.size(); ++i) {
    if (i % 16 == 0) {
      DEBUG_MSG("\n\t0x" << std::hex << std::setw(4) << std::setfill('0') << i
                         << ":");
    }
    if (i % 2 == 0) {
      DEBUG_MSG(" ");
    }
    DEBUG_MSG(std::hex << std::setw(2) << std::setfill('0') << (int)packet[i]);
  }
  DEBUG_MSG(std::dec << std::endl);
  arg->testcase.input += packet;
  arg->actual_count++;
}

size_t testcase_from_pcap(const std::string &filename, Testcase &testcase,
                          InputLimit limit) {
  pcap_t *handle;
  char errbuf[PCAP_ERRBUF_SIZE];
  size_t input_length_limit =
      limit.type == InputLimit::Type::length ? limit.value : -1;
  size_t count = limit.type == InputLimit::Type::count ? limit.value : 0;
  size_t actual_count = 0;

  handle = pcap_open_offline(filename.data(), errbuf);
  if (handle == nullptr) {
    std::cerr << "Error opening pcap file: " << errbuf << std::endl;
    return 0;
  }

  HandlerArg handler_arg = {testcase, input_length_limit, handle, actual_count};
  int ret =
      pcap_loop(handle, count, reinterpret_cast<pcap_handler>(packet_handler),
                (unsigned char *)&handler_arg);
  if (ret < 0 && ret != PCAP_ERROR_BREAK) {
    std::cerr << "Error reading packets: " << pcap_geterr(handle) << std::endl;
  }

  pcap_close(handle);
  return actual_count;
}

size_t testcase_from_pcap_folder(const std::string &foldername,
                                 Testcase &testcase, InputLimit limit) {
  size_t input_length_limit =
      (limit.type == InputLimit::Type::length) ? limit.value : -1;
  size_t count_limit =
      (limit.type == InputLimit::Type::count) ? limit.value : 0;
  size_t actual_length = 0;
  size_t actual_count = 0;
  size_t file_cnt = 0;

  std::set<fs::path> files;
  for (const auto &entry : fs::directory_iterator(foldername)) {
    if (entry.is_regular_file() && entry.path().extension() == ".tcpdump") {
      files.insert(entry.path());
    }
  }
  for (auto &filename : files) {
    std::cout << "Reading " << filename << std::endl;
    pcap_t *handle;
    char errbuf[PCAP_ERRBUF_SIZE];

    handle = pcap_open_offline(filename.c_str(), errbuf);
    if (handle == nullptr) {
      std::cerr << "Error opening pcap file: " << errbuf << std::endl;
      continue;
    }

    HandlerArg handler_arg = {testcase, input_length_limit, handle,
                              actual_count};
    int ret = pcap_loop(handle, count_limit,
                        reinterpret_cast<pcap_handler>(packet_handler),
                        (unsigned char *)&handler_arg);
    if (ret < 0 && ret != PCAP_ERROR_BREAK) {
      std::cerr << "Error reading packets from file " << filename << ": "
                << pcap_geterr(handle) << std::endl;
    }

    pcap_close(handle);
    actual_length = testcase.input.size();
    file_cnt++;

    if (limit.type == InputLimit::Type::length) {
      if (actual_length >= input_length_limit) {
        break;
      }
    } else {
      if (actual_count >= count_limit) {
        break;
      } else {
        count_limit = limit.value - actual_count;
      }
    }
  }

  std::cout << file_cnt << " files loaded\n";
  return actual_count;
}
