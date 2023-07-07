# %%
from pathlib import Path

import pandas as pd
from julearn.model_selection import RepeatedContinuousStratifiedKFold
from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging
from sklearn.gaussian_process.kernels import RBF

# Needs to install skrvm
# pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip
from skrvm import RVR

from julearn import run_cross_validation

# %%
configure_logging(level="INFO")

data_dir = Path(__file__).parent.parent / "data"
output_dir = Path(__file__).parent.parent / "results" / "1_brain_age"

output_dir.mkdir(parents=True, exist_ok=True)

input_file = data_dir / "ixi.S4_R8.csv"
output_file_prefix = "ixi.S4_R8_scores"

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
# Model 1: RVR
creator = PipelineCreator(problem_type="regression")
creator.add("select_variance", threshold=1e-5)
creator.add("pca")
creator.add(
    RVR(), kernel=["linear", "poly"], degree=[1, 2], random_state=rand_seed
)

cv = RepeatedContinuousStratifiedKFold(
    n_bins=20, n_splits=n_splits, n_repeats=n_repeats, random_state=rand_seed
)

scores1, model1 = run_cross_validation(
    X=["f_.*"],
    X_types={"continuous": ["f_.*"]},
    y="age",
    data=data_df,
    model=creator,
    cv=cv,
    return_estimator="final",
    seed=rand_seed,
    scoring=[
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "r2",
    ],
)
scores1["model"] = "rvr"
print(model1.best_params_)
scores1.to_csv(output_dir / f"{output_file_prefix}_rvr.csv", index=False)

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
    X_types={"continuous": ["f_.*"]},
    y="age",
    data=data_df,
    model=creator,
    cv=cv,
    return_estimator="final",
    seed=rand_seed,
    scoring=[
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "r2",
    ],
)
scores2["model"] = "gauss"
scores2.to_csv(output_dir / f"{output_file_prefix}_gauss.csv", index=False)

# %%
# Model 3: SVM
creator = PipelineCreator(problem_type="regression")
creator.add("select_variance", threshold=1e-5)
creator.add("pca")
creator.add("svm", kernel=["linear", "rbf", "poly"], C=[0.01, 0.1])

scores3, model3 = run_cross_validation(
    X=["f_.*"],
    X_types={"continuous": ["f_.*"]},
    y="age",
    data=data_df,
    model=creator,
    cv=cv,
    return_estimator="final",
    seed=rand_seed,
    scoring=[
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "r2",
    ],
)
scores3["model"] = "svm"
print(model3.best_params_)
scores3.to_csv(output_dir / f"{output_file_prefix}_svm.csv", index=False)
