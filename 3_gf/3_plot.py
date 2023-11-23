# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

results_dir = Path(__file__).parent.parent / "results" / "3_gf"

# %%
preds_fname = "cv-LOO_corrsign-pos_sigthresh-0.01_cvpreds.csv"
preds_df = pd.read_csv(results_dir / preds_fname, index_col=0)

# explicitly initialise a figure to have more control
cm = 1 / 2.54  # inch to cm conversion rate
fig, axes = plt.subplots(1, 2, figsize=(18 * cm, 9 * cm))

r, p = pearsonr(preds_df["repeat0_p0"], preds_df["target"])

# make the actual plot
sns.regplot(
    data=preds_df,
    x="target",
    y="repeat0_p0",
    ax=axes[0],
    scatter_kws={"s": 10},
)


axes[0].annotate(
    f"r={r:.2f}\np={p:.0e}",
    xy=(9.5, 22.5),
)

axes[0].set_xlabel("Ground truth (#correct questions)")
axes[0].set_ylabel("Predicted")
axes[0].set_title("Predicted vs. ground truth\nusing LOO-CV")


all_kfold_scores = []
for t_fname in results_dir.glob("cv-Repeated10Fold_*_cvscores.csv"):
    t_df = pd.read_csv(t_fname, index_col=0)
    t_sign = t_fname.stem.split("_")[1].split("-")[1]
    t_thresh = float(t_fname.stem.split("_")[2].split("-")[1])
    t_df["sign"] = t_sign
    t_df["thresh"] = t_thresh
    all_kfold_scores.append(t_df)

all_kfold_scores_df = pd.concat(all_kfold_scores)
r_corr_df = all_kfold_scores_df.groupby(["sign", "thresh", "repeat"])[
    "test_r_corr"
].mean().reset_index()

sns.boxplot(
    x="sign",
    y="test_r_corr",
    hue="thresh",
    data=r_corr_df,
    ax=axes[1],
)
# sns.swarmplot(
#     x="sign",
#     y="test_r_corr",
#     hue="thresh",
#     dodge=True,
#     data=r_corr_df,
#     ax=axes[1],
#     color="k",
#     alpha=0.5,
#     legend=False,
#     size=4,
# )

axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set_ylim(0, 0.3)
axes[1].set_xlabel("Correlation sign (pos/neg)")
axes[1].set_ylabel("Correlation (r)")
axes[1].set_title("Correlation values across folds\nusing 10 times 10Fold-CV")
# %%
fig.savefig(results_dir / "fig5.pdf", bbox_inches="tight")

# %%
