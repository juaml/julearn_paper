# %%
from pathlib import Path
import pandas as pd

# Needs to install skrvm
# pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip
from skrvm import RVR
from sklearn.gaussian_process.kernels import RBF

from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn.model_selection import RepeatedStratifiedGroupsKFold
from julearn.utils import configure_logging
from julearn.stats import corrected_ttest
from julearn.viz import plot_scores

import seaborn as sns


# %%
configure_logging(level="INFO")

#data_dir = Path(__file__).parent.parent / "data"
data_dir = Path('../data/')
output_dir = Path('../results/')

input_file = data_dir / "ixi.S4_R8.csv"
output_file = output_dir / "ixi.S4_R8_scores.csv"

data_df = pd.read_csv(input_file)
print(data_df.head())

# %%
# Set up the parameters for the cross-validation
rand_seed = 200
n_repeats = 5
n_splits = 5
scoring = [
    "neg_mean_absolute_error",
    "neg_mean_squared_error",
    "r2",
]

# %%
# Create age bins for a stratified k-fold cross-validation

n_bins = len(data_df) // n_splits

bins_on = data_df["age"]  # variable to be used for stratification
qc = pd.cut(bins_on.tolist(), n_bins)  # divides data in bins
data_df["bins"] = qc.codes
print(data_df.head())

# %%
# Model 1: RVR
creator = PipelineCreator(problem_type="regression")
creator.add("select_variance", threshold=1e-5)
creator.add("pca")
creator.add(
    RVR(), kernel=["linear", "poly"], degree=[1, 2], random_state=rand_seed
)

cv = RepeatedStratifiedGroupsKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=rand_seed
)

scores1, model1 = run_cross_validation(
   X=["f_.*"],
   y="age",
   groups="bins",
   data=data_df,
   model=creator,
   cv=cv,
   seed=rand_seed,
   scoring=[
       "neg_mean_absolute_error",
       "neg_mean_squared_error",
       "r2",
   ],
)
scores1["model"] = "rvr"
print(model1.best_params_)

# %%
# Model 2: GPR
creator = PipelineCreator(problem_type="regression")
creator.add("select_variance", threshold=1e-5)
creator.add("pca")
creator.add(
    "gauss",
    kernel=RBF(10.0, (1e-7, 10e7)),
    normalize_y=True,
    n_restarts_optimizer=100,
    random_state=rand_seed,
)

scores2, model2 = run_cross_validation(
   X=["f_.*"],
   y="age",
   groups="bins",
   data=data_df,
   model=creator,
   cv=cv,
   seed=rand_seed,
   scoring=[
       "neg_mean_absolute_error",
       "neg_mean_squared_error",
       "r2",
   ],
)
scores2["model"] = "gauss"
print(model2.best_params_)


# %%
# Model 3: SVM
creator = PipelineCreator(problem_type="regression")
creator.add("select_variance", threshold=1e-5)
creator.add("pca")
creator.add(
    "svm", 
    kernel=["linear", "rbf", "poly"], 
    C=[0.01, 0.1]
    )

scores3, model3 = run_cross_validation(
   X=["f_.*"],
   y="age",
   groups="bins",
   data=data_df,
   model=creator,
   cv=cv,
   seed=rand_seed,
   scoring=[
       "neg_mean_absolute_error",
       "neg_mean_squared_error",
       "r2",
   ],
)
scores3["model"] = "svm"
print(model3.best_params_)


# %%
all_scores = pd.concat([scores1, scores2, scores3], axis=0)
all_scores.to_csv(output_file, index=False)

# Plot
# sns.set_style("darkgrid")

# wide_df = all_scores.melt(
#     id_vars=["model", "fold", "repeat"], var_name="metric"
# )

# sns.catplot(
#     x="model", y="value", col="metric", data=wide_df, kind="box", sharey=False
# )

# %%
#corrected ttest
stats_df = corrected_ttest(scores1, scores2, scores3)
print(stats_df)

# %%
# Interactive plot
panel = plot_scores(scores1, scores2, scores3)
panel.show()