# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "click",
#     "matplotlib",
# ]
# ///
import subprocess
import click
import matplotlib.pyplot as plt
import re
import os

# Constants
COLORS = ["blue", "green", "orange"]
MARKERS = ["o", "s", "^"]
OUTPUT_DIR = "benchmark_results"


def run_parallel_regex_by_input(cuda: bool = False) -> None:
    """
    Run the `pp-parallel-regex` command for different input lengths and parse the output.

    Parameters:
    cuda (bool): If True, do not add the '-o' flag. If False, add the '-o' flag.
    """
    len_values = [10**i for i in range(3, 10)]
    results: dict[str, list[tuple[int, float]]] = {}

    for length in len_values:
        length_str = str(length)
        command = (
            f"bin/pp-parallel-regex -l {length_str} {'-o' if not cuda else ''} "
            "dataset/dfa/079.txt dataset/traffic"
        )
        print(command)

        try:
            output = subprocess.check_output(command, shell=True, text=True)
            for line in output.strip().splitlines():
                parts = line.split(":::")
                if len(parts) == 2:
                    name, speedup_str = parts[0].strip(), parts[1].strip()
                    try:
                        speedup = float(speedup_str)
                        results.setdefault(name, []).append((length, speedup))
                    except ValueError:
                        print(
                            f"Skipping invalid speedup value '{speedup_str}' for line: {line}"
                        )
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for len={length}: {e}")

    plot_results(
        results,
        "Input Length",
        200,
        "Threads: 4, States: 79",
        cuda,
    )


def run_parallel_regex_by_state(cuda: bool = False) -> None:
    """
    Run the `pp-parallel-regex` command for different state counts and plot speedups.

    Parameters:
    cuda (bool): If True, do not add the '-o' flag. If False, add the '-o' flag.
    """
    rules_folder = "dataset/dfa"
    results: dict[str, list[tuple[int, float]]] = {}

    for filename in sorted(os.listdir(rules_folder)):
        if filename.endswith(".txt"):
            match = re.match(r"(\d{3})\.txt", filename)
            if match:
                statecnt = int(match.group(1))
                path = os.path.join(rules_folder, filename)
                command = (
                    f"bin/pp-parallel-regex -l 1000000000 {'-o' if not cuda else ''} "
                    f"{path} dataset/traffic"
                )
                print(command)

                try:
                    output = subprocess.check_output(command, shell=True, text=True)
                    for line in output.strip().splitlines():
                        parts = line.split(":::")
                        if len(parts) == 2:
                            name, speedup_str = parts[0].strip(), parts[1].strip()
                            try:
                                speedup = float(speedup_str)
                                results.setdefault(name, []).append((statecnt, speedup))
                            except ValueError:
                                print(
                                    f"Skipping invalid speedup value '{speedup_str}' for line: {line}"
                                )
                except subprocess.CalledProcessError as e:
                    print(f"Error executing command for {filename}: {e}")

    plot_results(
        results,
        "Number of States",
        -30,
        "Threads: 4, Input Length: 1e9",
        cuda,
    )


def run_parallel_regex_by_thread() -> None:
    """
    Run the `pp-parallel-regex` command for different thread counts and plot speedups.
    """
    results: dict[str, list[tuple[int, float]]] = {}

    for thread_count in range(1, 13):
        command = (
            f"bin/pp-parallel-regex -t {thread_count} -l 1000000000 -o "
            "dataset/dfa/079.txt dataset/traffic"
        )
        print(command)

        try:
            output = subprocess.check_output(command, shell=True, text=True)
            for line in output.strip().splitlines():
                parts = line.split(":::")
                if len(parts) == 2:
                    name, speedup_str = parts[0].strip(), parts[1].strip()
                    try:
                        speedup = float(speedup_str)
                        results.setdefault(name, []).append((thread_count, speedup))
                    except ValueError:
                        print(
                            f"Skipping invalid speedup value '{speedup_str}' for line: {line}"
                        )
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for thread count {thread_count}: {e}")

    plot_results(
        results,
        "Number of Threads",
        0,
        "States: 79, Input Length: 1e9",
        False,
    )


def plot_results(
    results: dict[str, list[tuple[int, float]]],
    xlabel: str,
    num_off: int,
    desc: str,
    cuda: bool,
) -> None:
    """
    Plot the results of the benchmarks.

    Parameters:
    results (dict): The results to plot.
    xlabel (str): The label for the x-axis.
    num_off (int): The offset of max label.
    desc (str): Description show up at bottom-right corner.
    cuda (bool): If True, indicates that the plot is for CUDA, otherwise for OpenMP.
    """
    plt.figure(figsize=(8, 6))

    for (name, speedups), color, marker in zip(results.items(), COLORS, MARKERS):
        x_vals = [entry[0] for entry in speedups]
        y_vals = [entry[1] for entry in speedups]
        max_speedup = max(y_vals)

        plt.plot(x_vals, y_vals, label=name, marker=marker, color=color, linestyle="-")
        plt.axhline(y=max_speedup, color=color, linestyle=":", linewidth=1)
        plt.text(
            num_off,
            max_speedup,
            f"{max_speedup:.3f}",
            color=color,
            fontsize=10,
            verticalalignment="bottom",
        )

    plt.axhline(y=1, color="tomato", linestyle="--", linewidth=1.5, label="Serial")
    plt.text(
        1,
        -0.095,
        desc,
        transform=plt.gca().transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    plt.xscale("log" if xlabel == "Input Length" else "linear")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Speedup", fontsize=12)
    plt.title(
        f"Speedup vs {xlabel.title()} ({'CUDA' if cuda else 'OpenMP'})", fontsize=14
    )
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    fig_name = re.sub(r"[\s\-]+", "_", xlabel.lower())
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{fig_name}_{('cuda' if cuda else 'omp')}.jpg"),
        dpi=300,
    )


def validate_runs(ctx: click.Context, param: click.Parameter, value: str) -> list[str]:
    valid_options = ["len-omp", "len-cuda", "state-omp", "state-cuda", "thread"]

    if value == "all":
        return ["len-omp", "len-cuda", "state-omp", "state-cuda", "thread"]
    if value == "omp":
        return ["len-omp", "state-omp", "thread"]
    if value == "cuda":
        return ["len-cuda", "state-cuda"]

    runs = value.split(",")
    for run in runs:
        if run not in valid_options:
            raise click.BadParameter(
                f"Invalid run type: {run}. Valid options are {', '.join(valid_options)}"
            )

    return runs


@click.command()
@click.option(
    "-r",
    "--runs",
    default="all",
    help="""\b
    Comma-separated list of runs (len-omp, len-cuda, state-omp, state-cuda, thread),
    or one of the following keywords.
    'all': run all benchmarks.
    'omp': shorthand of 'len-omp,state-omp'.
    'cuda': shorthand of 'len-cuda,state-cuda,thread'.
    """,
    required=False,
    show_default=True,
    callback=validate_runs,
)
def main(runs: list[str]) -> None:
    """
    Runs benchmarks for "Parallelization of Regualr Expression Matching".
    """
    if not os.path.exists("dataset"):
        print("Please run this script from project root.")
        exit(1)
    print(f"Running the following tasks: {', '.join(runs)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if "len-omp" in runs:
        run_parallel_regex_by_input()
    if "len-cuda" in runs:
        run_parallel_regex_by_input(cuda=True)
    if "state-omp" in runs:
        run_parallel_regex_by_state()
    if "state-cuda" in runs:
        run_parallel_regex_by_state(cuda=True)
    if "thread" in runs:
        run_parallel_regex_by_thread()


if __name__ == "__main__":
    main()
