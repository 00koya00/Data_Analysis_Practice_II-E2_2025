import numpy as np
import descriptive as desc
from matplotlib import pyplot as plt

def First_analysis(num=1):
    about = desc.df.groupby("group")["Grade"].agg(["mean", "std", "count"])
    about["se"] = about["std"] / np.sqrt(about["count"])
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    axes = axes.flatten()
    pairs_short = [("nnn", "nyn"), ("nny", "nyy"), ("nny", "yny"), ("nyn", "nyy"), ("ynn", "yyn")]
    pairs_long  = [ ["Study well", "Study less"], ["Fail", "No Fail"], ["Present", "Absent"]]
    for i, (c1, c2) in enumerate(pairs_short):
        g1 = desc.expand_code(c1)
        g2 = desc.expand_code(c2)
        ax = axes[i]
        m1 = about.loc[g1, "mean"]
        n1 = about.loc[g1, "count"]
        sd1 = about.loc[g1, "std"]
        se1 = sd1 / np.sqrt(n1)
        m2 = about.loc[g2, "mean"]
        n2 = about.loc[g2, "count"]
        sd2 = about.loc[g2, "std"]
        se2 = sd2 / np.sqrt(n2)
        means = [m1, m2]
        ses = [se1, se2]
        x = [0, 1]
        ax.bar(x, means, yerr=ses, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels([desc.translate_one(c1, pairs_long),  "\n" + desc.translate_one(c2, pairs_long)])
        ax.set_ylabel("Mean Grade")
        ax.set_title(desc.translate_one(c1, pairs_long) + "\n" + "vs" + "\n" + desc.translate_one(c2, pairs_long))
    if len(axes) > len(pairs_short):
        for j in range(len(pairs_short), len(axes)):
            axes[j].axis("off")
    desc.display_title("Significant pairs of groups (study time & absence & failures, p < 0.05)", pref='Figure', num=num)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def Second_analysis(num=1):
    about2 = desc.df.groupby("group_2")["Grade"].agg(["mean", "std", "count"])
    about2["se"] = about2["std"] / np.sqrt(about2["count"])
    pairs_short2 = [("nnn", "nny"), ("nny", "yny"), ("ynn", "yny")]
    pairs_long2  = [ ["high medu", "low medu-"], ["high fedu", "low fedu"], ["well ctrl", "bad ctrl"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    axes = axes.flatten()

    for i, (c1, c2) in enumerate(pairs_short2):
        g1 = desc.expand_code(c1)  # "nnn" -> "n_n_n"
        g2 = desc.expand_code(c2)
        ax = axes[i]
        m1 = about2.loc[g1, "mean"]
        n1 = about2.loc[g1, "count"]
        sd1 = about2.loc[g1, "std"]
        se1 = sd1 / np.sqrt(n1)
        m2 = about2.loc[g2, "mean"]
        n2 = about2.loc[g2, "count"]
        sd2 = about2.loc[g2, "std"]
        se2 = sd2 / np.sqrt(n2)
        means = [m1, m2]
        ses   = [se1, se2]
        x = np.arange(2)
        ax.bar(x, means, yerr=ses, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels([desc.translate_one(c1, pairs_long2),  "\n" + desc.translate_one(c2, pairs_long2)])
        ax.set_ylim(0, 20)
        ax.set_ylabel("Mean grade")
        ax.set_title(desc.translate_one(c1, pairs_long2) + "\n" + "vs" + "\n" + desc.translate_one(c2, pairs_long2))
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    desc.display_title("Significant pairs of groups (mother/father's education & going-out, p < 0.05)", pref='Figure', num=num)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()