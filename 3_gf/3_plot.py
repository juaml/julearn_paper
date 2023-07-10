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
fig, ax = plt.subplots(1, 1, figsize=(18 * cm, 18 * cm))

r, p = pearsonr(preds_df["repeat0_p0"], preds_df["target"])

# make the actual plot
sns.regplot(data=preds_df, x="target", y="repeat0_p0", ax=ax)


ax.annotate(
    f"r={r:.2f}\np={p:.0e}",
    xy=(9.5, 22.5),
)

ax.set_xlabel("Ground truth (#correct questions)")
ax.set_ylabel("Predicted")
ax.set_title("Predicted vs. ground truth using LOO-CV")

# %%
