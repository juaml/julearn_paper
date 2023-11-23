# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from statannotations.Annotator import Annotator

# %%
results_dir = Path(__file__).parent.parent / "results" / "2_confounds"
data_df = pd.read_csv(results_dir / "used_data.csv", index_col=0)
preds_corrected = pd.read_csv(results_dir / "preds_corrected.csv", index_col=0)
preds_uncorrected = pd.read_csv(results_dir / "preds_raw.csv", index_col=0)
# %% Convert to long format and add age for each subject
preds_corrected_long = (
    preds_corrected.reset_index().set_index(["index", "target"]).stack()
)
preds_corrected_long.index.names = ["index", "target", "repeat"]
preds_corrected_long.name = "prediction"
preds_corrected_long = preds_corrected_long.reset_index().set_index("index")
preds_corrected_long = preds_corrected_long.join(data_df["age"])

preds_uncorrected_long = (
    preds_uncorrected.reset_index().set_index(["index", "target"]).stack()
)
preds_uncorrected_long.index.names = ["index", "target", "repeat"]
preds_uncorrected_long.name = "prediction"
preds_uncorrected_long = preds_uncorrected_long.reset_index().set_index(
    "index"
)
preds_uncorrected_long = preds_uncorrected_long.join(data_df["age"])
# %% Check misclassified
preds_corrected_long["misclassified"] = (
    preds_corrected_long["prediction"] != preds_corrected_long["target"]
)
preds_uncorrected_long["misclassified"] = (
    preds_uncorrected_long["prediction"] != preds_uncorrected_long["target"]
)

# %% Keep only the missclassified ones and compute the mean age for each group
misclassified_corrected = preds_corrected_long.query(
    "misclassified == True"
).copy()
misclassified_corrected = (
    misclassified_corrected.groupby(["repeat", "target"])
    .mean("age")
    .reset_index()
)
misclassified_corrected["label"] = "misclassified corrected"

misclassified_uncorrected = preds_uncorrected_long.query(
    "misclassified == True"
).copy()
misclassified_uncorrected = (
    misclassified_uncorrected.groupby(["repeat", "target"])
    .mean("age")
    .reset_index()
)
misclassified_uncorrected["label"] = "misclassified uncorrected"

# Map the target to a group and filter the columns
misclassified_corrected["Group"] = misclassified_uncorrected.target.map(
    {1: "AD or MCI", 0: "control"}
)
misclassified_uncorrected["Group"] = misclassified_uncorrected.target.map(
    {1: "AD or MCI", 0: "control"}
)
valid_cols = ["age", "Group", "label"]
misclassified_corrected = misclassified_corrected[valid_cols]
misclassified_uncorrected = misclassified_uncorrected[valid_cols]

# Get the list of subjects and plot the age per group
all_subjects = data_df[["age", "current_diagnosis"]].copy()
all_subjects["target"] = (
    all_subjects["current_diagnosis"].isin(["AD", "MCI"]).astype(int)
)
all_subjects["label"] = "all subjects"
all_subjects["Group"] = all_subjects.target.map({1: "AD or MCI", 0: "control"})
# %%

all_dfs = [
    all_subjects,
    misclassified_corrected,
    misclassified_uncorrected,
]
df_to_plot = pd.concat(all_dfs)


order = [
    "all subjects",
    "misclassified corrected",
    "misclassified uncorrected",
]

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    x="label",
    y="age",
    hue="Group",
    hue_order=["AD or MCI", "control"],
    data=df_to_plot,
    order=order,
    ax=ax,
)
ax.set_ylim(64, 75)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
ax.set_xlabel("")
ax.set_ylabel("Mean age across repeats")

pairs = [
    (("all subjects", "AD or MCI"), ("all subjects", "control")),
    (
        ("misclassified corrected", "AD or MCI"),
        ("misclassified corrected", "control"),
    ),
    (
        ("misclassified uncorrected", "AD or MCI"),
        ("misclassified uncorrected", "control"),
    ),
]

annotator = Annotator(
    ax,
    pairs,
    data=df_to_plot,
    x="label",
    y="age",
    hue="Group",
    order=order,
    hue_order=["AD or MCI", "control"],
)
annotator.configure(test="t-test_ind", text_format="star", loc="inside")
annotator.apply_and_annotate()

fig.savefig(results_dir / "fig4.pdf", dpi=300)

# %%
