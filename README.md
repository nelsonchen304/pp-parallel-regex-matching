# Parallelization of Regular Expression Matching

Main framework to test and compare parallel regex implementations.

## Prerequisite
The traffic dataset is not included in the repository and is needed to run the program.
You may get it from [DARPA Dataset's Webpage](https://archive.ll.mit.edu/ideval/data/1999/testing/week4/index.html).

The result shown in our report is based on `outside.tcpdump.data` from Monday~Thursday of week 4, 1999. We provide [scripts/get_traffic.sh](scripts/get_traffic.sh) for downloading the data required to reproduce the experiment. `gunzip` and `wget` are required to run the script. The downloaded files can be found under `dataset/traffic/`.

## Usage

You can reproduce the benchmark presented in Figure 2 of the report or execute the program directly.

### Benchmark

Python >= 3.9 is needed to run the benchmarks.

> <details>
> 
> <summary> Our experimental environment: </summary>
> 
> - OS:  Ubuntu 22.04 (6.8.0-49-generic #49~22.04.1-Ubuntu)  
> - CPU: Intel(R) Core(TM) i7-8700K @ 3.70GHz (6C12T)  
> - RAM: 64GB DDR4 (4 * 16GB)  
> - GPU: NVIDIA RTX 4090 GPU 24GB
> 
> </datails>

If `uv` is available, simply run `make benchmark` to build and run all the benchmarks. 

To select a specific benchmark, manually run the script. The script can be run by `uv run scripts/benchmark.py` without creating an environment. If run by `python`, matplotlib and click are needed. For available options, pass `--help` as an argument.

```
$ uv run scripts/benchmark.py --help
Reading inline script metadata from `scripts/benchmark.py`
Usage: benchmark.py [OPTIONS]

 Runs benchmark for "Parallelization of Regular Expression Matching".

Options:
 -r, --runs TEXT  Comma-separated list of runs (len-omp, len-cuda, state-omp, state-cuda, thread),
 or one of the following keywords.
 'all': run all benchmarks.
 'omp': shorthand of 'len-omp,state-amp'.
 'cuda': shorthand of 'len-cuda,state-cuda,thread'.  [default: all]
 --help           Show this message and exit.
```

> [!NOTE]
> The benchmark takes about 5~10 minutes to complete in our environment. 
> Execution time may vary depending on the hardware.

The result plots will be saved in `benchmark_results`. 
Our experiment result is included in this repository in [report](report).

### Run Manually
Use `make` to build the project. You can run `./bin/pp-parallel-regex -h` to see the usage.

```
$ ./bin/pp-parallel-regex -h
Usage: ./bin/pp-parallel-regex [-t <thread count>] [-c <pcap packet count>] [-l <pcap length>] [-r <num runs>] [-ovh] <dfa file> <pcap folder>
Options:
 -t <thread count>: Set number of threads (default: 4)
 -c <pcap packet count>: Set maximum number of packets to parse; conflict with -l (default: 3)
 -l <pcap length>: Set minimum number of bytes to parse; conflict with -c
 -r <num runs>: Set number of times to run each matcher; minimum time is used to evaluate (default: 3)
 -o: Run only the OpenMP matcher, else run only CUDA matcher(default: false)
 -v: Verbose output (default: false)
 -h: Print this help message and exit
```

### Create DFA files
`./bin/regex2dfa` is used to generate DFA files from regexes.

Lines of regular expressions are read from stdin. 
The input is terminated by an empty line.

The regular expression is in the format `/{regular expression}/{flags}`. For example:  
- `/(a|b)*abb/`
- `/a+(b|c)*/i`

> <details>
> 
> <summary> supported regular expression syntax: </summary>
> 
> ### alphabet
> - ASCII printable characters
> - hex byte specified as `\x[00-ff]`
> - all alphabets specified as `.` (see [flags](###flags))
> - `\d`, `\D`, `\s` `\S`, `\w`, `\W`, `\r`, `\n`, `\t` as defined in pcre
> - use `\` to escape special characters
> - `[{alphabets}]`/`[^{alphabets}]` specify a group of included/excluded alphabet.
> 
> ### operators (order by high to low precedence)
> - `(`, `)`: parentheses; use to change precedence
> - `*`, `+`: zero or more / one or more
> - concatenation (does not need to specify explicitly)
> - `|`: alternation (or)
>
> ### flags
> - `i`: case insensitive
> - `s`: `.` includes the newline character
> 
> </details>

Minimized DFA is written to stdout. Some examples:
- direct input (use [rlwrap](https://github.com/hanslub42/rlwrap) to make your life easier):
```
$ rlwrap ./bin/regex2dfa
/c\x87\x63/

#states
s0
s1
s2
s3
#initial
s0
#accepting
s3
#alphabet
c
0x87
#transitions
s0:c>s1
s1:0x87>s2
s2:c>s3
```
- pipe from stdin:
```
$ echo '/(a|b)*abb/\n' | ./bin/regex2dfa
#states
s0
s1
s2
s3
s4
#initial
s0
#accepting
s3
#alphabet
a
b
#transitions
s0:a>s1
s0:b>s0
s1:a>s1
s1:b>s2
s2:a>s1
s2:b>s3
s3:a>s1
s3:b>s0
s4:a>s1
s4:b>s2
```

Additionally, use `./bin/regex2dfa -p` to make the regex can be matched at any location of the string.

## Report

A copy of the report is also included at [report/pp-final.pdf](report/pp-final.pdf) with the original LaTeX code.