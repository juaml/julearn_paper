#
# Follow this steps to first download the data from LONI.
# If you already have an account and access to ADNI, you can skip to step 6
# 
# 1) Create an account in LONI: https://ida.loni.usc.edu
# 2) Once you login, select "ADNI" in feature studies.
# 3) Click on the "Apply for access" button.
# 4) Complete the application form and submit it.
# 5) Wait for the approval email. This may take a few days.
#
#
# 6) Login into LONI: https://ida.loni.usc.edu
# 7) Select ADNI
# 8) Download -> Study data
# 9) Select the following options:
# * Diagnosis -> Diagnosis Summary
# * Imaging -> MR Image Analysis -> 
# * UCSF - Cross-Sectional FreeSurfer (6.0) Dictionary [ADNI3] 
# * UCSF - Cross-Sectional FreeSurfer (6.0) [ADNI3] 
# * Subject characteristics -> Subject demographics
# 10) Download the files and place them in the `data/2_confounds` directory.

import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path(__file__).parent.parent / 'data' / '2_confounds'

d = pd.read_csv('ADNI/DXSUM_PDXCONV_ADNIALL.csv')
#w = pd.read_csv('ADNI_whitematter_hyperintensity/ADNI_UCD_WMH_05_02_22.csv')
#w = pd.read_csv('ADNI_FS_51/UCSFFSX51_11_08_19.csv')
w = pd.read_csv('ADNI/UCSFFSX6_08_17_22.csv')

w.rename(columns={'COLPROT':'Phase'}, inplace=True)

demog = pd.read_csv('ADNI/PTDEMOG.csv')
demog = demog[demog.Phase=='ADNI3']

#%%
dout = pd.merge(d, w, on=['RID','Phase', 'VISCODE2'], 
              how='inner', suffixes=('_diagnosis', '_fs'))

dout = pd.merge(dout, demog, on=['RID','Phase', 'VISCODE2'], 
              how='inner', suffixes=('_imaging_diagnosis', '_demog'))

print(dout.shape)

# get current diagnosis
# ADNI1: DXCURREN 1=NL; 2=MCI; 3=AD
adniphase = dout['Phase']=='ADNI1'
idx = (adniphase) & (dout['DXCURREN']==1)
dout.loc[idx,'current_diagnosis'] = 'NL'
idx = (adniphase) & (dout['DXCURREN']==2)
dout.loc[idx,'current_diagnosis'] = 'MCI'
idx = (adniphase) & (dout['DXCURREN']==3)
dout.loc[idx,'current_diagnosis'] = 'AD'

# ADNIGO/2: DXCHANGE 1=Stable: NL to NL; 2=Stable: MCI to MCI; 3=Stable: Dementia to De-
# mentia; 4=Conversion: NL to MCI; 5=Conversion: MCI to Dementia; 6=Conversion: NL to Dementia;
# 7=Reversion: MCI to NL; 8=Reversion: Dementia to MCI; 9=Reversion: Dementia to NL
adniphase = (dout['Phase']=='ADNI2') | (dout['Phase']=='ADNIGO')
idx = (adniphase) & (dout['DXCHANGE']==1)
dout.loc[idx,'current_diagnosis'] = 'NL'
idx = (adniphase) & (dout['DXCHANGE']==2)
dout.loc[idx,'current_diagnosis'] = 'MCI'
idx = (adniphase) & (dout['DXCHANGE']==3)
dout.loc[idx,'current_diagnosis'] = 'AD'

idx = (adniphase) & (dout['DXCHANGE']==4)
dout.loc[idx,'current_diagnosis'] = 'NL-MCI'
idx = (adniphase) & (dout['DXCHANGE']==5)
dout.loc[idx,'current_diagnosis'] = 'MCI-AD'
idx = (adniphase) & (dout['DXCHANGE']==6)
dout.loc[idx,'current_diagnosis'] = 'NL-AD'
idx = (adniphase) & (dout['DXCHANGE']==7)
dout.loc[idx,'current_diagnosis'] = 'MCI-NL'
idx = (adniphase) & (dout['DXCHANGE']==8)
dout.loc[idx,'current_diagnosis'] = 'AD-MCI'
idx = (adniphase) & (dout['DXCHANGE']==9)
dout.loc[idx,'current_diagnosis'] = 'AD-NL'

# ADNI3: DIAGNOSIS 1=CN; 2=MCI; 3=Dementia
adniphase = dout['Phase']=='ADNI3'
idx = (adniphase) & (dout['DIAGNOSIS']==1)
dout.loc[idx,'current_diagnosis'] = 'NL'
idx = (adniphase) & (dout['DIAGNOSIS']==2)
dout.loc[idx,'current_diagnosis'] = 'MCI'
idx = (adniphase) & (dout['DIAGNOSIS']==3)
dout.loc[idx,'current_diagnosis'] = 'AD'

# %%
# filter for first subject visit
dout_subonce = dout.drop_duplicates(subset=('Phase','RID'))
qcpass = dout_subonce.OVERALLQC=='Pass'
complete = dout_subonce.STATUS=='complete'
dout_subonce_qc = dout_subonce[qcpass & complete]

dout_subonce_qc.to_csv(data_dir / 'ADNI_subonce_qc.csv', index=False)

