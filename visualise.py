import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

from evaluate import join_scores
from process_ldopa import build_metadata


def performance_table():
    my_scores = pd.read_pickle("outputs/predictions/all_scores.pkl")
    my_scores["source"] = [
        "OxWalk" if "wrist" in pid else "Ldopa" for pid in my_scores.index
    ]
    scores = my_scores.groupby(["source"]).mean()
    scores.to_csv("outputs/predictions/performance_table.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", "-d", default="outputs/predictions")
    parser.add_argument("--metadir", "-m", default="data/Ldopa_Processed")
    parser.add_argument("--outdir", "-o", default="outputs/plots")
    args = parser.parse_args()

    score_df = join_scores(args.datadir)
    metadata = build_metadata(processeddir=args.metadir)["MeanUPDRS"]

    df = pd.concat([score_df, metadata], axis=1)

    dfm = df.melt("MeanUPDRS", var_name="Models", value_name="Macro F1 score")
    dfm.dropna(inplace=True)

    g = sns.lmplot(dfm, x="MeanUPDRS", y="Macro F1 score", hue="Models", ci=None)

    os.makedirs(args.outdir, exist_ok=True)
    plt.savefig(os.path.join(args.outdir, "main.png"))
