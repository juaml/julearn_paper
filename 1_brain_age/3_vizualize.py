# Run this script as "panel serve 3_vizualize.py"
# %%
from pathlib import Path
import pandas as pd
from julearn.viz import plot_scores

output_dir = Path(__file__).parent.parent / "results" / "1_brain_age"
output_file_prefix = "ixi.S4_R8_scores"

models = ["gauss", "rvr", "svm"]

all_scores = []
for t_model in models:
    t_score = pd.read_csv(output_dir / f"{output_file_prefix}_{t_model}.csv")
    all_scores.append(t_score)

panel = plot_scores(*all_scores)
panel.servable()
