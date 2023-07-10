import pandas as pd
import numpy as np
from pathlib import Path
import re


def load_and_prepare_data():
    """Prepare data from different sources and compose them for julearn.

    Returns
    -------
    prediction_data : pd.DataFrame
        The data to be used for prediction.
    features : list of str
        The names of the features to be used for prediction.
    target : str
        The name of the target variable.
    """
    data_path = Path(__file__).parent.parent / "data" / "3_gf"
    # load the junifer preprocessed connectomes
    conn_path = (
        data_path
        / "dataset-HCPREST1_parc-Finn2015Shen268_marker-empiricalFC.csv"
    )
    hcp_connectomes = pd.read_csv(conn_path, index_col=0)
    hcp_connectomes.index = hcp_connectomes.index.astype(str)
    # new_names = {
    #     c: c.replace("~", re.escape("~")) for c in hcp_connectomes.columns
    # }
    # hcp_connectomes = hcp_connectomes.rename(columns=new_names)

    # this data contains the desired target
    behav_path = data_path / "unrestricted.csv"
    hcp_behav = pd.read_csv(behav_path, index_col=0)
    hcp_behav.index = hcp_behav.index.astype(str)

    # load and select only the subjects that are desired for this run:
    subjects_file = data_path / "unrelated_subjects_rms_filtered.txt"
    subjects = list(np.loadtxt(subjects_file, dtype=str))

    # put everyting into one dataframe
    prediction_data = hcp_connectomes.loc[subjects]
    hcp_behav = hcp_behav.loc[subjects]

    # prepare X and y for julearn
    target = "PMAT24_A_CR"
    features = list(hcp_connectomes.columns)

    # add the target
    prediction_data[target] = hcp_behav[target].loc[prediction_data.index]

    return prediction_data, features, target
