#include <algorithm>
#include <bitset>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <memory>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "regex.hpp"

std::string set_repr(state_set s) {
  std::stringstream ss;
  bool first = true;
  ss << "{";
  for (int j = 0; j < MAX_NFA_STATES; ++j) {
    if (s.test(j)) {
      if (first) {
        first = false;
      } else {
        ss << ", ";
      }
      ss << j;
    }
  }
  ss << "}";
  return ss.str();
}

std::string set_repr(alphabet_set s) {
  std::stringstream ss;
  bool first = true;
  ss << "{";
  for (int j = 0; j < ALPHABET_SIZE; ++j) {
    if (s.test(j)) {
      if (first) {
        first = false;
      } else {
        ss << ", ";
      }
      ss << j;
    }
  }
  ss << "}";
  return ss.str();
}

/* NFA implementation */
NFA::NFA() : start(-1), accept(-1), state_count(0), min_unused_state(0) {
  for (int i = 0; i < MAX_NFA_STATES; ++i) {
    for (int j = 0; j < ALPHABET_SIZE + 1; ++j) {
      transitions[i][j][0] = -1;
      transitions[i][j][1] = -1;
    }
  }
}

int NFA::size() const { return state_count; }

int NFA::new_state() {
  if (min_unused_state == -1) {
    std::cerr << "NFA: too many states\n";
    exit(1);
  }
  int r = min_unused_state;
  used.set(r);
  for (int i = 0; i < MAX_NFA_STATES; ++i) {
    if (!used.test(i)) {
      min_unused_state = i;
      break;
    }
  }
  state_count++;
  if (state_count >= MAX_NFA_STATES) {
    min_unused_state = -1;
  }
  return r;
}

void NFA::delete_state(int s) {
  used.reset(s);
  for (int i = 0; i < ALPHABET_SIZE + 1; ++i) {
    transitions[s][i][0] = -1;
    transitions[s][i][1] = -1;
  }
  state_count--;
}

void NFA::merge_state(int s, int t) {
  for (int i = 0; i < ALPHABET_SIZE + 1; ++i) {
    transitions[s][i][0] = transitions[t][i][0];
    transitions[s][i][1] = transitions[t][i][1];
  }
  delete_state(t);
}

void NFA::add_transition(int from, int c, int to) {
  if (transitions[from][c][0] == -1) {
    transitions[from][c][0] = to;
  } else if (transitions[from][c][1] == -1) {
    transitions[from][c][1] = to;
  } else {
  }
}

bool NFA::has_transition(int from, int c, int to) const {
  return transitions[from][c][0] == to || transitions[from][c][1] == to;
}

void NFA::print() const {
  for (int i = 0; i < MAX_NFA_STATES; ++i) {
    if (!used.test(i))
      continue;
    DEBUG_MSG("State " << i
                       << (i == start    ? " (start)"
                           : i == accept ? " (accept)"
                                         : "")
                       << "\n");
    for (int j = 0; j < ALPHABET_SIZE + 1; ++j) {
      int a = transitions[i][j][0];
      int b = transitions[i][j][1];
      if (a != -1) {
        DEBUG_MSG("  [" << std::setw(3) << j << "]-> {" << a);
        if (b != -1) {
          DEBUG_MSG(", " << b);
        }
        DEBUG_MSG("}\n");
      }
    }
  }
}

state_set NFA::eps_closure(const state_set &s) const {
  state_set mask;
  for (int i = 0; i < MAX_NFA_STATES; ++i) {
    if (s.test(i)) {
      eps_closure(i, mask);
    }
  }

  if (mask.any()) {
    DEBUG_MSG("eps-closure(" << set_repr(s) << ") = " << set_repr(mask)
                             << "\n");
  }

  return mask;
}

state_set NFA::mov(const state_set &t, int a) const {
  state_set ret;
  for (int i = 0; i < MAX_NFA_STATES; ++i) {
    if (t.test(i)) {
      for (int j = 0; j < MAX_NFA_STATES; ++j) {
        if (has_transition(i, a, j)) {
          ret.set(j);
        }
      }
    }
  }

  if (ret.any()) {
    DEBUG_MSG("move(" << set_repr(t) << ", " << a << ") = " << set_repr(ret)
                      << "\n");
  }

  return ret;
}

void NFA::eps_closure(int state, state_set &mask) const {
  mask.set(state);
  for (int i = 0; i < MAX_NFA_STATES; ++i) {
    if (!mask.test(i) && has_transition(state, EPS_MOVE, i)) {
      eps_closure(i, mask);
    }
  }
}

DFA::DFA() : size(0) {
  for (int i = 0; i < MAX_NFA_STATES; ++i) {
    for (int j = 0; j < ALPHABET_SIZE; ++j) {
      transitions[i][j] = -1;
    }
  }
}

void print_alpha(int c) {
  if (32 <= c && c <= 126) {
    std::cout << static_cast<char>(c);
  } else {
    std::cout << "0x" << std::hex << std::uppercase << std::setw(2)
              << std::setfill('0') << c << std::dec << std::nouppercase;
  }
}

void DFA::print() {
  std::cout << "#states\n";
  for (int i = 0; i < size; ++i) {
    std::cout << "s" << i << "\n";
  }
  std::cout << "#initial\n";
  std::cout << "s0\n";
  std::cout << "#accepting\n";
  for (int i = 0; i < size; ++i) {
    if (accept_state.test(i)) {
      std::cout << "s" << i << "\n";
    }
  }
  std::cout << "#alphabet\n";
  for (int i = 0; i < ALPHABET_SIZE; ++i) {
    if (used_alphabet.test(i)) {
      print_alpha(i);
      std::cout << "\n";
    }
  }
  std::cout << "#transitions\n";
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < ALPHABET_SIZE; ++j) {
      if (transitions[i][j] != -1) {
        std::cout << "s" << i << ":";
        print_alpha(j);
        std::cout << ">s" << transitions[i][j] << "\n";
      }
    }
  }
}

int parse_regex_string(const std::string &input_regex, std::string &regex) {
  if (input_regex[0] != '/') {
    std::cerr << "Regex must start with '/': " << input_regex << "\n";
    return -1;
  }

  std::string process_regex = input_regex.substr(1);

  std::string::size_type end = input_regex.rfind("/");
  if (end == std::string::npos) {
    std::cerr << "Failed to find end of regex: " << input_regex << "\n";
    return -1;
  }
  regex = input_regex.substr(1, end - 1);
  std::string flag_string = input_regex.substr(end + 1);

  int flags = 0;
  for (auto &c : flag_string) {
    switch (c) {
    case 'i':
      flags |= CASE_INSENSITIVE;
      break;
    case 's':
      flags |= NEWLINE_IN_DOT;
      break;
    default:
      std::cerr << "Skipping flag:" << c << "\n";
    }
  }
  return flags;
}

std::vector<int> preprocess(const std::string &regex,
                            std::vector<alphabet_set> &sets, int flags) {
  std::vector<int> ret;

  for (size_t i = 0; i < regex.size(); ++i) {
    switch (regex[i]) {
    case '\\':
      if (i + 1 < regex.size()) {
        if (i > 0 && ret.back() < CONCAT) {
          ret.emplace_back(CONCAT);
        }
        char next = regex[i + 1];
        if (next == 'x' && i + 3 < regex.size() &&
            std::isxdigit(regex[i + 2]) && std::isxdigit(regex[i + 3])) {
          std::string hex = regex.substr(i + 2, 2);
          int value = std::stoi(hex, nullptr, 16);
          sets.emplace_back(alphabet_set());
          sets.back().set(value);
          ret.emplace_back(-(sets.size() - 1));
          i += 3;
        } else if (next == 'd') {
          sets.emplace_back(nums);
          ret.emplace_back(-(sets.size() - 1));
          i += 1;
        } else if (next == 'D') {
          sets.emplace_back(~nums);
          ret.emplace_back(-(sets.size() - 1));
          i += 1;
        } else if (next == 's') {
          sets.emplace_back(white);
          ret.emplace_back(-(sets.size() - 1));
          i += 1;
        } else if (next == 'S') {
          sets.emplace_back(~white);
          ret.emplace_back(-(sets.size() - 1));
          i += 1;
        } else if (next == 'w') {
          sets.emplace_back(word);
          ret.emplace_back(-(sets.size() - 1));
          i += 1;
        } else if (next == 'W') {
          sets.emplace_back(~word);
          ret.emplace_back(-(sets.size() - 1));
          i += 1;
        } else if (next == 'n') {
          sets.emplace_back(alphabet_set());
          sets.back().set('\n');
          ret.emplace_back(-(sets.size() - 1));
          i += 1;
        } else if (next == 'r') {
          sets.emplace_back(alphabet_set());
          sets.back().set('\r');
          ret.emplace_back(-(sets.size() - 1));
          i += 1;
        } else if (next == 't') {
          sets.emplace_back(alphabet_set());
          sets.back().set('\t');
          ret.emplace_back(-(sets.size() - 1));
          i += 1;
        } else {
          sets.emplace_back(alphabet_set());
          sets.back().set(next);
          ret.emplace_back(-(sets.size() - 1));
          i += 1;
        }
      } else {
        std::cerr << "stray '\\' at end of regex: " << regex << "\n";
      }
      break;
    case '(':
      if (i > 0 && ret.back() < CONCAT) {
        ret.emplace_back(CONCAT);
      }
      ret.emplace_back(LPARAN);
      break;
    case ')':
      ret.emplace_back(RPARAN);
      break;
    case '|':
      ret.emplace_back(ALTERN);
      break;
    case '.':
      if (i > 0 && ret.back() < CONCAT) {
        ret.emplace_back(CONCAT);
      }
      if (flags & NEWLINE_IN_DOT) {
        sets.emplace_back(all_characters);
      } else {
        sets.emplace_back(all_characters_no_newline);
      }
      ret.emplace_back(-(sets.size() - 1));
      break;
    case '*':
      ret.emplace_back(KLEENE);
      break;
    case '+':
      ret.emplace_back(KLEENE_PLUS);
      break;
    case '?':
      ret.emplace_back(OPTIONAL);
      break;
    case '[': {
      int to = regex.substr(i).find(']');
      std::string group = regex.substr(i + 1, to - 1);
      DEBUG_MSG("group=" << group << "\n");
      bool neg = false;
      alphabet_set alpha_group;
      if (group[0] == '^') {
        neg = true;
        group = group.substr(1);
      }
      for (size_t j = 0; j < group.size(); ++j) {
        if (group[j] == '\\') {
          if (j + 1 < group.size()) {
            char next = group[j + 1];
            if (next == 'x' && j + 3 < regex.size() &&
                std::isxdigit(group[j + 2]) && std::isxdigit(group[j + 3])) {
              std::string hex = group.substr(j + 2, 2);
              int value = std::stoi(hex, nullptr, 16);
              alpha_group.set(value);
              j += 3;
            } else if (next == 'd') {
              alpha_group |= nums;
              j += 1;
            } else if (next == 'D') {
              alpha_group |= ~nums;
              j += 1;
            } else if (next == 's') {
              alpha_group |= white;
              j += 1;
            } else if (next == 'S') {
              alpha_group |= ~white;
              j += 1;
            } else if (next == 'w') {
              alpha_group |= word;
              j += 1;
            } else if (next == 'W') {
              alpha_group |= ~word;
              j += 1;
            } else if (next == 'n') {
              alpha_group.set('\n');
              j += 1;
            } else if (next == 'r') {
              alpha_group.set('\r');
              j += 1;
            } else if (next == 't') {
              alpha_group.set('\t');
              j += 1;
            } else {
              alpha_group.set(next);
              j += 1;
            }
          }
        } else {
          if (std::isalpha(group[j]) && (flags & CASE_INSENSITIVE)) {
            alpha_group.set(std::toupper(group[j]));
            alpha_group.set(std::tolower(group[j]));
          } else {
            alpha_group.set(group[j]);
          }
        }
      }
      if (neg) {
        alpha_group.flip();
      }
      sets.emplace_back(alpha_group);
      DEBUG_MSG("alpha_group=" << set_repr(alpha_group) << "\n");
      ret.emplace_back(-(sets.size() - 1));
      i += to + 1;
      break;
    }
    default:
      if (i > 0 && ret.back() < CONCAT) {
        ret.emplace_back(CONCAT);
      }
      sets.emplace_back(alphabet_set());
      if (std::isalpha(regex[i]) && (flags & CASE_INSENSITIVE)) {
        sets.back().set(std::toupper(regex[i]));
        sets.back().set(std::tolower(regex[i]));
      } else {
        sets.back().set(regex[i]);
      }
      ret.emplace_back(-(sets.size() - 1));
    }
  }
  return ret;
}

int prec(int op) { return op < CONCAT ? -KLEENE : -op; }

std::vector<int> postfix(const std::vector<int> &pre) {
  std::vector<int> ret;
  std::stack<int> s;
  int paran = 0;
  for (auto &i : pre) {
    if (i < CONCAT) {
      ret.emplace_back(i);
    } else if (i == LPARAN) {
      s.push(i);
      paran++;
    } else if (i == RPARAN) {
      while (s.top() != LPARAN) {
        ret.emplace_back(s.top());
        s.pop();
      }
      s.pop();
      paran--;
    } else {
      // if (s.empty() || i < s.top() || paran > 0) {
      //   s.push(i);
      // } else {
      while (!s.empty() && prec(i) <= prec(s.top())) {
        ret.emplace_back(s.top());
        s.pop();
      }
      s.push(i);
      // }
    }
  }
  while (!s.empty()) {
    ret.emplace_back(s.top());
    s.pop();
  }
  return ret;
}

void construct_nfa(const std::vector<int> &post, NFA &nfa,
                   std::vector<alphabet_set> &sets) {
  std::stack<std::pair<int, int>> stk;
  if (nfa.start != -1) {
    stk.push(std::make_pair(nfa.start, nfa.accept));
  }
  for (const int &i : post) {
    if (i <= 0) {
      int q = nfa.new_state();
      int f = nfa.new_state();
      for (int a = 0; a < ALPHABET_SIZE; ++a) {
        if (sets.at(-i).test(a)) {
          nfa.add_transition(q, a, f);
        }
      }
      stk.push(std::make_pair(q, f));
    } else if (i == ALTERN) {
      int q = nfa.new_state();
      int f = nfa.new_state();
      std::pair<int, int> s = stk.top();
      stk.pop();
      std::pair<int, int> t = stk.top();
      stk.pop();
      nfa.add_transition(q, EPS_MOVE, s.first);
      nfa.add_transition(q, EPS_MOVE, t.first);
      nfa.add_transition(s.second, EPS_MOVE, f);
      nfa.add_transition(t.second, EPS_MOVE, f);
      stk.push(std::make_pair(q, f));
    } else if (i == KLEENE) {
      int q = nfa.new_state();
      int f = nfa.new_state();
      std::pair<int, int> s = stk.top();
      stk.pop();
      nfa.add_transition(q, EPS_MOVE, f);
      nfa.add_transition(q, EPS_MOVE, s.first);
      nfa.add_transition(s.second, EPS_MOVE, f);
      nfa.add_transition(s.second, EPS_MOVE, s.first);
      stk.push(std::make_pair(q, f));
    } else if (i == KLEENE_PLUS) {
      int q = nfa.new_state();
      int f = nfa.new_state();
      std::pair<int, int> s = stk.top();
      stk.pop();
      nfa.add_transition(q, EPS_MOVE, s.first);
      nfa.add_transition(s.second, EPS_MOVE, f);
      nfa.add_transition(s.second, EPS_MOVE, s.first);
      stk.push(std::make_pair(q, f));
    } else if (i == OPTIONAL) {
      int q = nfa.new_state();
      int f = nfa.new_state();
      std::pair<int, int> s = stk.top();
      stk.pop();
      nfa.add_transition(q, EPS_MOVE, f);
      nfa.add_transition(q, EPS_MOVE, s.first);
      nfa.add_transition(s.second, EPS_MOVE, f);
      stk.push(std::make_pair(q, f));
    } else if (i == CONCAT) {
      std::pair<int, int> s = stk.top();
      stk.pop();
      std::pair<int, int> t = stk.top();
      stk.pop();
      nfa.merge_state(t.second, s.first);
      stk.push(std::make_pair(t.first, s.second));
    } else {
      std::cerr << "Unknown token: " << i << "\n";
    }
  }
  if (stk.size() != 1) {
    int q = nfa.new_state();
    int f = nfa.new_state();
    std::pair<int, int> s = stk.top();
    stk.pop();
    std::pair<int, int> t = stk.top();
    stk.pop();
    nfa.add_transition(q, EPS_MOVE, s.first);
    nfa.add_transition(q, EPS_MOVE, t.first);
    nfa.add_transition(s.second, EPS_MOVE, f);
    nfa.add_transition(t.second, EPS_MOVE, f);
    stk.push(std::make_pair(q, f));
  }
  std::pair<int, int> s = stk.top();
  nfa.start = s.first;
  nfa.accept = s.second;
}

void construct_nfa(const std::string &input_regex, NFA &nfa) {
  std::vector<alphabet_set> sets;
  std::string regex;
  int flags = parse_regex_string(input_regex, regex);
  if (flags < 0) {
    return;
  }
  std::vector<int> pre = preprocess(regex, sets, flags);
  std::vector<int> post = postfix(pre);
  construct_nfa(post, nfa, sets);
}

void construct_dfa(const NFA &nfa, DFA &dfa) {
  for (int i = 0; i < MAX_NFA_STATES; ++i) {
    for (int j = 0; j < ALPHABET_SIZE; ++j) {
      dfa.transitions[i][j] = -1;
    }
  }
  std::vector<state_set> dstates;
  state_set s;
  s.set(nfa.start);
  dstates.emplace_back(nfa.eps_closure(s));
  int last_marked = -1;
  while (last_marked < (int)dstates.size() - 1) {
    last_marked++;
    const state_set t = dstates.at(last_marked);
    DEBUG_MSG("marked: " << set_repr(t) << " (" << last_marked << ")\n");
    for (int a = 0; a < ALPHABET_SIZE; ++a) {
      state_set u = nfa.eps_closure(nfa.mov(t, a));
      if (u.none()) {
        continue;
      }
      auto it = std::find_if(dstates.begin(), dstates.end(),
                             [u](const state_set &s) { return s == u; });
      int n = std::distance(dstates.begin(), it);
      if (it == dstates.end()) {
        dstates.emplace_back(u);
        DEBUG_MSG("> add: " << set_repr(u) << " (" << n
                            << "), size(dstates) = " << dstates.size() << "\n");
        if (u.test(nfa.accept)) {
          dfa.accept_state.set(dstates.size() - 1);
        }
        n = dstates.size() - 1;
      }
      dfa.transitions[last_marked][a] = n;
    }
  }
  dfa.size = dstates.size();
  for (int i = 0; i < dfa.size; ++i) {
    for (int c = 0; c < ALPHABET_SIZE; ++c) {
      if (dfa.transitions[i][c] != -1) {
        dfa.used_alphabet.set(c);
      }
    }
  }
}

int lowest(const state_set &s) { return s.none() ? -1 : s._Find_first(); }

bool cmp_stateset(const state_set &a, const state_set &b) {
  return lowest(a) < lowest(b);
}

void minimize_dfa(DFA &dfa) {
  std::list<state_set> p;
  std::list<state_set> w;
  state_set all = ~state_set() >> (MAX_NFA_STATES - dfa.size);
  state_set f = dfa.accept_state, nf = all & ~f;
  p.emplace_back(nf);
  p.emplace_back(f);
  w.emplace_back(nf);
  w.emplace_back(f);

  while (!w.empty()) {
    state_set a = w.front();
    w.pop_front();
    DEBUG_MSG("distinguisher: " << set_repr(a) << "\n");
    for (int c = 0; c < ALPHABET_SIZE; ++c) {
      state_set x; // X := set of states for which a transition on c leads to a
                   // state in A
      for (int i = 0; i < dfa.size; ++i) {
        int to = dfa.transitions[i][c];
        if (to != -1 && a.test(to)) {
          x.set(i);
        }
      }
      // for each set Y in P for which X \union Y is nonempty and Y \\ X is
      // nonempty ( Y_1 <- X \union Y, Y_2 <- Y \\ X )
      for (auto it = p.begin(); it != p.end();) {
        state_set y = *it;
        state_set y1 = x & y;
        state_set y2 = y & ~x;
        if (y1.none() || y2.none()) {
          it++;
          continue;
        }
        DEBUG_MSG("> " << set_repr(y) << "=>" << set_repr(y1) << " U "
                       << set_repr(y2) << " by " << c << "\n");
        // replace Y in P by two sets X \union Y and Y \\ X
        it = p.erase(it);
        p.emplace_back(y1);
        p.emplace_back(y2);
        auto ne = std::remove(w.begin(), w.end(), y);
        if (ne != w.end()) { // Y \in W
          w.erase(ne);
          w.emplace_back(y1);
          w.emplace_back(y2);
        } else {
          if (y1.count() < y2.count()) {
            w.emplace_back(y1);
          } else {
            w.emplace_back(y2);
          }
        }
      }
    }
  }
  p.sort(cmp_stateset);
#ifdef DEBUG
  for (auto &s : p) {
    std::cout << set_repr(s) << "\n";
  }
#endif
  int new_size = p.size();

  int state_to[dfa.size];
  int state_from[new_size];
  state_set new_accept_state;
  std::fill_n(state_from, new_size, -1);
  DEBUG_MSG("convert to state: ");
  for (int i = 0; i < dfa.size; ++i) {
    int t = 0;
    for (auto &s : p) {
      if (s.test(i)) {
        state_to[i] = t;
        if (state_from[t] == -1) {
          state_from[t] = i;
        }
        if (dfa.accept_state.test(i)) {
          new_accept_state.set(t);
        }
        break;
      }
      ++t;
    }
    DEBUG_MSG(t << " ");
  }
  DEBUG_MSG("\n");
  dfa.accept_state = new_accept_state;

  DEBUG_MSG("convert from state: ");
  for (int i = 0; i < new_size; ++i) {
    DEBUG_MSG(state_from[i] << " ");
  }
  DEBUG_MSG("\n");

  for (int i = 0; i < new_size; ++i) {
    for (int c = 0; c < ALPHABET_SIZE; ++c) {
      int orig = dfa.transitions[state_from[i]][c];
      dfa.transitions[i][c] = orig == -1 ? -1 : state_to[orig];
      if (orig != -1) {
        dfa.used_alphabet.set(c);
      }
    }
  }
  dfa.size = new_size;
}

void regex_set_to_min_dfa(const std::vector<std::string> &regex_set, DFA &dfa,
                          bool pad_around) {
  std::unique_ptr<NFA> nfa = std::make_unique<NFA>();
  for (auto &regex : regex_set) {
    construct_nfa(regex, *nfa);
  }
  if (pad_around) {
    int new_init = nfa->new_state();
    int new_accept = nfa->new_state();

    for (int c = 0; c < ALPHABET_SIZE; ++c) {
      nfa->add_transition(new_init, c, new_init);
      nfa->add_transition(new_accept, c, new_accept);
    }

    nfa->add_transition(new_init, EPS_MOVE, nfa->start);
    nfa->add_transition(nfa->accept, EPS_MOVE, new_accept);

    nfa->start = new_init;
    nfa->accept = new_accept;
  }
  construct_dfa(*nfa, dfa);
  minimize_dfa(dfa);
}
