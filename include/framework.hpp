#pragma once
#include "regex.hpp"
#include <memory>

typedef std::basic_string<unsigned char> ustring;

struct Testcase {
  ustring input;
  bool expected;
};

struct InputLimit {
  enum class Type { count, length };
  Type type;
  size_t value;
  InputLimit(Type type, size_t value) : type(type), value(value) {}
};

class Matcher {
public:
  explicit Matcher(std::shared_ptr<const DFA> dfa) : dfa(dfa) {}
  virtual bool match(const ustring &input) = 0;
  virtual const std::string name() = 0;

protected:
  std::shared_ptr<const DFA> dfa;
};

void dfa_from_file(const std::string &filename, DFA &dfa);
size_t testcase_from_pcap(const std::string &filename, Testcase &testcases,
                          InputLimit limit);
size_t testcase_from_pcap_folder(const std::string &foldername,
                                 Testcase &testcases, InputLimit limit);
