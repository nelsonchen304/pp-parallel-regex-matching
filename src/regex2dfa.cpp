#include <iostream>

#include "regex.hpp"

NFA nfa;
DFA dfa;

int main(int argc, char **argv) {
  std::string input_regex;
  std::string regex;
  std::vector<alphabet_set> sets;
  bool pad_around = false;
  if (argc > 1 && std::string(argv[1]) == "-p") {
    pad_around = true;
  }

#ifndef DEBUG
  std::vector<std::string> regex_set;
#endif

  while (std::getline(std::cin, input_regex)) {
    if (input_regex.empty()) {
      break;
    }

#ifdef DEBUG
    int flags = parse_regex_string(input_regex, regex);
    DEBUG_MSG("\n========= REGEX  INFO =========\n");
    DEBUG_MSG("/" << regex << "/ : " << (flags & CASE_INSENSITIVE ? 'i' : '-')
                  << (flags & NEWLINE_IN_DOT ? 'm' : '-') << "\n");
    std::vector<int> pre = preprocess(regex, sets, flags);
    DEBUG_MSG("\n===== PREPROCESSED  INFIX =====\n");
    for (auto &i : pre) {
      switch (i) {
      case LPARAN:
        DEBUG_MSG("[(]");
        break;
      case RPARAN:
        DEBUG_MSG("[)]");
        break;
      case ALTERN:
        DEBUG_MSG("[|]");
        break;
      case KLEENE:
        DEBUG_MSG("[*]");
        break;
      case CONCAT:
        DEBUG_MSG("[##]");
        break;
      case KLEENE_PLUS:
        DEBUG_MSG("[+]");
        break;
      case OPTIONAL:
        DEBUG_MSG("[?]");
        break;
      default:
        DEBUG_MSG(set_repr(sets.at(-i)));
      }
      DEBUG_MSG(" ");
    }
    DEBUG_MSG("\n");
    DEBUG_MSG("\n=========== POSTFIX ===========\n");
    std::vector<int> post = postfix(pre);
    for (auto &i : post) {
      switch (i) {
      case LPARAN:
        DEBUG_MSG("[(]");
        break;
      case RPARAN:
        DEBUG_MSG("[)]");
        break;
      case ALTERN:
        DEBUG_MSG("[|]");
        break;
      case KLEENE:
        DEBUG_MSG("[*]");
        break;
      case CONCAT:
        DEBUG_MSG("[##]");
        break;
      case KLEENE_PLUS:
        DEBUG_MSG("[+]");
        break;
      case OPTIONAL:
        DEBUG_MSG("[?]");
        break;
      default:
        DEBUG_MSG(set_repr(sets.at(-i)));
      }
      DEBUG_MSG(" ");
    }
    DEBUG_MSG("\n");
    DEBUG_MSG("\n======== CONSTRUCT NFA ========\n");
    construct_nfa(post, nfa, sets);
    DEBUG_MSG("\n========= NFA  STATES =========\n");
    nfa.print();
    DEBUG_MSG("\n===============================\n");
#else
    regex_set.emplace_back(input_regex);
#endif
  }

#ifdef DEBUG
  if (pad_around) {
    int new_init = nfa.new_state();
    int new_accept = nfa.new_state();

    for (int c = 0; c < ALPHABET_SIZE; ++c) {
      nfa.add_transition(new_init, c, new_init);
      nfa.add_transition(new_accept, c, new_accept);
    }

    nfa.add_transition(new_init, EPS_MOVE, nfa.start);
    nfa.add_transition(nfa.accept, EPS_MOVE, new_accept);

    nfa.start = new_init;
    nfa.accept = new_accept;
  }
  DEBUG_MSG("\n======== END NFA INPUT ========\n");
  DEBUG_MSG("\n======== CONSTRUCT DFA ========\n");
  construct_dfa(nfa, dfa);
#else
  regex_set_to_min_dfa(regex_set, dfa, pad_around);
#endif
  DEBUG_MSG("\n========= DFA  STATES =========\n");
  for (int i = 0; i < dfa.size; ++i) {
    DEBUG_MSG("State " << i << (i == 0 ? " (start)" : "")
                       << (dfa.accept_state.test(i) ? " (accept)" : "")
                       << "\n");
    for (int j = 0; j < ALPHABET_SIZE; ++j) {
      if (dfa.transitions[i][j] != -1) {
        DEBUG_MSG(" [" << std::setw(3) << j << "]-> " << dfa.transitions[i][j]
                       << "\n");
      }
    }
  }
  DEBUG_MSG("\n======== MINIMIZE  DFA ========\n");
  minimize_dfa(dfa);
  DEBUG_MSG("\n===== MINIMIZE DFA STATES =====\n");
  for (int i = 0; i < dfa.size; ++i) {
    DEBUG_MSG("State " << i << (i == 0 ? " (start)" : "")
                       << (dfa.accept_state.test(i) ? " (accept)" : "")
                       << "\n");
    for (int j = 0; j < ALPHABET_SIZE; ++j) {
      if (dfa.transitions[i][j] != -1) {
        DEBUG_MSG(" [" << std::setw(3) << j << "]-> " << dfa.transitions[i][j]
                       << "\n");
      }
    }
  }
  DEBUG_MSG("\n===============================\n");

  dfa.print();
}
