"""Create a string to parametrise condor jobs and print to terminal."""

from itertools import product
import os
from pathlib import Path

cwd = os.getcwd()
log_dir = Path(cwd) / 'logs'
log_dir.mkdir(exist_ok=True)

env = 'julearn'

exec_string = \
    "1_predict_gf.py $(sign_threshold) $(corr_sign) $(cv)"

PREAMBLE = f"""
universe       = vanilla
getenv         = True

request_cpus   = 1
request_memory = 25GB
request_disk   = 5GB

initial_dir    = {cwd}
executable     = {cwd}/run_in_venv.sh
transfer_executable = False

arguments      = {env} python {exec_string}

# Logs
log            = {log_dir.as_posix()}/predict_gf_$(sign_threshold)_$(corr_sign)_$(cv).log
output         = {log_dir.as_posix()}/predict_gf_$(sign_threshold)_$(corr_sign)_$(cv).out
error          = {log_dir.as_posix()}/predict_gf_$(sign_threshold)_$(corr_sign)_$(cv).err

"""


print(PREAMBLE)

# Kfold
for sign_threshold in [0.01, 0.05, 0.10]:
    for corr_sign in ["pos", "neg", "posneg"]:
        print("cv=Repeated10Fold")
        print(f"sign_threshold={sign_threshold}")
        print(f"corr_sign={corr_sign}")
        print("queue")
        print()

# LOO
print("cv=LOO")
print(f"sign_threshold=0.01")
print(f"corr_sign=pos")
print("queue")
print()