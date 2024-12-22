#pragma once
#include <string>

#include "framework.hpp"

class SerialMatcher final : public Matcher {
public:
  using Matcher::Matcher;
  bool match(const ustring &text) override;
  const std::string name() override { return "Serial"; }
};
