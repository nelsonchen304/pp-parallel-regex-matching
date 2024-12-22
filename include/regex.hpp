#pragma once

#include <bitset>
#include <string>
#include <vector>

#ifdef DEBUG
#include <iomanip>
#include <iostream>
#define DEBUG_MSG(str)                                                         \
  do {                                                                         \
    std::cout << str;                                                          \
  } while (0)
#else
#define DEBUG_MSG(str)                                                         \
  do {                                                                         \
  } while (0)
#endif

const int ALPHABET_SIZE = 256;
const int MAX_NFA_STATES = 10000;
const int EPS_MOVE = ALPHABET_SIZE;

enum operators {
  KLEENE = 1,
  KLEENE_PLUS,
  OPTIONAL,
  CONCAT,
  ALTERN,
  LPARAN,
  RPARAN,
};

enum flag {
  CASE_INSENSITIVE = 1 << 0, /* i */
  NEWLINE_IN_DOT = 1 << 1,   /* s */
};

typedef std::bitset<MAX_NFA_STATES> state_set;
typedef std::bitset<ALPHABET_SIZE> alphabet_set;

inline alphabet_set make_nums_set() {
  alphabet_set ret;
  for (int i = '0'; i <= '9'; ++i) {
    ret.set(i);
  }
  return ret;
}

inline alphabet_set make_word_set() {
  alphabet_set ret;
  for (int i = 'A'; i <= 'Z'; ++i) {
    ret.set(i);
  }
  for (int i = 'a'; i <= 'z'; ++i) {
    ret.set(i);
  }
  for (int i = '0'; i <= '9'; ++i) {
    ret.set(i);
  }
  ret.set('_');
  return ret;
}

inline alphabet_set make_all_set(bool exclude_newline = true) {
  alphabet_set ret;
  ret = ~ret;
  if (exclude_newline) {
    ret.reset('\n');
  }
  return ret;
}

inline alphabet_set make_white_set() {
  alphabet_set ret;
  ret.set(' ');
  ret.set('\t');
  return ret;
}

const alphabet_set nums = make_nums_set();
const alphabet_set word = make_word_set();
const alphabet_set all_characters_no_newline = make_all_set();
const alphabet_set all_characters = make_all_set(false);
const alphabet_set white = make_white_set();

class NFA {
public:
  NFA();
  int size() const;
  int new_state();
  void delete_state(int s);
  void merge_state(int s, int t);
  void add_transition(int from, int c, int to);
  bool has_transition(int from, int c, int to) const;
  void print() const;
  state_set eps_closure(const state_set &s) const;
  state_set mov(const state_set &t, int a) const;

private:
  void eps_closure(int state, state_set &mask) const;

public:
  int start;
  int accept;

private:
  int transitions[MAX_NFA_STATES + 2][ALPHABET_SIZE + 1]
                 [2]; /* atmost 2 transitions per state */
  std::bitset<MAX_NFA_STATES> used;
  int state_count;
  int min_unused_state;
};

class DFA {
public:
  DFA();
  void print();

public:
  int transitions[MAX_NFA_STATES][ALPHABET_SIZE];
  state_set accept_state;
  alphabet_set used_alphabet;
  int size;
};

std::string set_repr(state_set s);
std::string set_repr(alphabet_set s);
int parse_regex_string(const std::string &input_regex, std::string &regex);
std::vector<int> preprocess(const std::string &regex,
                            std::vector<alphabet_set> &sets, int flags);
std::vector<int> postfix(const std::vector<int> &pre);
void construct_nfa(const std::vector<int> &post, NFA &nfa,
                   std::vector<alphabet_set> &sets);
void construct_nfa(const std::string &regex, NFA &nfa);
void construct_dfa(const NFA &nfa, DFA &dfa);
void minimize_dfa(DFA &dfa);

void regex_set_to_min_dfa(const std::vector<std::string> &regex_set, DFA &dfa,
                          bool pad_around);
