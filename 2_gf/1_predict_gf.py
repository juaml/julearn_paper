import argparse
from pathlib import Path

from sklearn.model_selection import LeaveOneOut, RepeatedKFold

from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging, logger, raise_error
from utils import load_and_prepare_data

from julearn import run_cross_validation

corr_signs = ["pos", "neg", "posneg"]
parser = argparse.ArgumentParser(
    description="Run julearn example for CBPM based on HCP data."
)
parser.add_argument("significance_threshold", type=float)
parser.add_argument("corr_sign", choices=corr_signs)
parser.add_argument("cv", choices=["LOO", "Repeated10Fold"])

args = parser.parse_args()

configure_logging(level="INFO")

cv_kind = args.cv
corr_sign = args.corr_sign
significance_threshold = args.significance_threshold

logger.info("Running julearn example for CBPM based on HCP data.")
logger.info("Parameters:")
logger.info(f"  - cv: {cv_kind}")
logger.info(f"  - corr_sign: {corr_sign}")
logger.info(f"  - significance_threshold: {significance_threshold}")

out_path = Path(__file__).parent.parent / "results" / "2_gf"
scores_out_fname = (
    f"cv-{cv_kind}_corrsign-{corr_sign}"
    f"_sigthresh-{significance_threshold}_cvscores.csv"
)

preds_out_fname = (
    f"cv-{cv_kind}_corrsign-{corr_sign}"
    f"_sigthresh-{significance_threshold}_cvpreds.csv"
)

logger.info(f"Saving scores to: {out_path / scores_out_fname:s}")
logger.info(f"Saving predictions to: {out_path / preds_out_fname:s}")
prediction_data, features, target = load_and_prepare_data()

# prepare the pipeline creator for CBPM with the correct parameters
cbpm_pipeline = PipelineCreator(problem_type="regression")
cbpm_pipeline.add(
    step="cbpm",
    significance_threshold=args.significance_threshold,
    corr_sign=args.corr_sign,
)
cbpm_pipeline.add(step="linreg")
X_types = {"continuous": features}

if cv_kind == "LOO":
    cv = LeaveOneOut()
elif cv_kind == "Repeated10Fold":
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=10)
else:
    raise_error(f"Unknown cv_kind: {cv_kind}")

scoring = [
    "neg_mean_absolute_error",
    "neg_mean_squared_error",
    "neg_root_mean_squared_error",
    "neg_median_absolute_error",
    "r2",
    "corr"
]

subs, _ = prediction_data.shape
if subs != 368:
    raise_error("There is a problem with subject list/data!")

logger.info(f"Starting to run {args.cv} cross-validation...")
scores, final_estimator, inspector = run_cross_validation(
    model=cbpm_pipeline,
    data=prediction_data,
    X=features,
    y=target,
    cv=cv,
    scoring=scoring,
    return_estimator="all",
    return_inspector=True,
    X_types=X_types,
)

scores.to_csv(out_path / scores_out_fname)
predictions = inspector.folds.predict()
predictions.to_csv(out_path / preds_out_fname)
