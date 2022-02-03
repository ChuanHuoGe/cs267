import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

import seaborn as sns

import pandas as pd

from pathlib import Path
import re

root = Path("./experiments")
fig_root = Path("./figures/")
fig_root.mkdir(exist_ok=True)

class Exp():
    def __init__(self, path):
        self.path = path
        self.sizes, self.mflops, self.percentages, self.avg_percent = self._parse()

    def _parse(self):
        pattern = re.compile("Size: ([0-9]+)\tMflops/s: ([0-9]+\.[0-9]+)\tPercentage: ([0-9]+\.[0-9]+)")
        avg_pattern = re.compile("Average percentage of Peak = ([0-9]+\.[0-9]+)")

        sizes = []
        mflops = []
        percentages = []
        avg_percent = 0.

        with self.path.open("r") as f:
            for line in f:
                line = line.strip()
                t = pattern.search(line)
                if t is not None:
                    sizes.append(int(t.group(1)))
                    mflops.append(float(t.group(2)))
                    percentages.append(float(t.group(3)))
                avg_t = avg_pattern.search(line)
                if avg_t is not None:
                    avg_percent = float(avg_t.group(1))
        return sizes, mflops, percentages, avg_percent

def plot_exps(exps, labels, filename, title):
    assert len(exps) == len(labels)
    for exp, label in zip(exps, labels):
        assert isinstance(exp, Exp)
        plt.plot(exp.sizes, exp.mflops, "-o", label=label)

    plt.xlabel("Matrix size")
    plt.ylabel("Mflops/s")
    plt.legend()
    plt.title(title)
    plt.savefig(fig_root / (filename + ".png"))
    plt.clf()

def plot_percent(exps, labels, xlabel, filename, title, rotate_x=False):
    plt.bar(labels, [exp.avg_percent for exp in exps])
    plt.xlabel(xlabel)
    plt.ylabel("Avg percentage of Peak (%)")
    if rotate_x:
        plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_root / (filename + ".png"))
    plt.clf()

def main():
    # compare baseline and SIMD
    exp_baseline = Exp(root / "job-blocked.baseline")
    exp1_b48 = Exp(root / "job-blocked.exp1_b48")

    plot_exps([exp_baseline, exp1_b48], ["Baseline (BLOCK_SIZE = 48)", "SIMD (BLOCK_SIZE = 48)"], "simd", "Baseline v.s SIMD (block-level jki)")
    plot_percent([exp_baseline, exp1_b48], ["Baseline", "SIMD"],
                "Method", "simd_bar",  "Baseline v.s SIMD (block-level jki)")

    # blocking
    exp1_b8 = Exp(root / "job-blocked.exp1_b8")
    exp1_b16 = Exp(root / "job-blocked.exp1_b16")
    exp1_b24 = Exp(root / "job-blocked.exp1_b24")
    exp1_b32 = Exp(root / "job-blocked.exp1_b32")
    exp1_b40 = Exp(root / "job-blocked.exp1_b40")
    # exp1_b48 = Exp(root / "job-blocked.exp1_b48")
    exp1_b56 = Exp(root / "job-blocked.exp1_b56")
    exp1_b64 = Exp(root / "job-blocked.exp1_b64")

    plot_percent([exp1_b8, exp1_b16, exp1_b24, exp1_b32, exp1_b40, exp1_b48, exp1_b56, exp1_b64],
             ["{}".format(num) for num in [8, 16, 24, 32, 40, 48, 56, 64]],
             "BLOCK_SIZE",
             "blocksize", "The effect of different BLOCK_SIZE")

    exp2 = Exp(root / "job-blocked.exp2")

    # loop order
    plot_exps([exp1_b48, exp2], ["block-level jki", "block-level kji"], "block_level", "Block-level loop order comparision")

    exp3 = Exp(root / "job-blocked.exp3")
    exp4 = Exp(root / "job-blocked.exp4")
    plot_exps([exp3, exp4], ["global-level jki", "global-level kji"], "global_level", "Global-level loop order comparision")


if __name__ == "__main__":
    main()
