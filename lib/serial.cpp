#include "serial.hpp"

bool SerialMatcher::match(const ustring &text) {
  int current_state = 0;
  bool matched = true;

  for (unsigned char c : text) {
    DEBUG_MSG(current_state << " ");
    unsigned int char_code = c;

    if (dfa->transitions[current_state][char_code] != -1) {
      current_state = dfa->transitions[current_state][char_code];
    } else {
      matched = false;
      DEBUG_MSG("\nunmatched at " << std::hex << std::setw(2)
                                  << std::setfill('0') << char_code << std::dec
                                  << "\n");
      DEBUG_MSG("  current state: " << current_state << "\n");
      DEBUG_MSG("  transitions: " << dfa->transitions[current_state][char_code]
                                  << "\n");
      break;
    }
  }

  DEBUG_MSG("\nfinal state: " << current_state << "\n");

  return matched && dfa->accept_state.test(current_state);
}
