import os
from glob import glob

from datalad.api import osf_credentials, install
from nilearn.glm.first_level import FirstLevelModel
from statsmodels.tsa.arima_model import ARMA
from numpy.linalg import LinAlgError
import numpy as np
import pandas as pd
from bids import BIDSLayout
import nibabel as nib


participant = "sub-GE120020"
task = "task-taskswitch"
BIDS = "atd6j"
FMRIPREP = "gzkqy"
N_VOXELS = 100
N_PARTICIPANTS = 10

def main():
    # set random state
    rng = np.random.default_rng(seed=0)
    osf_credentials()

    bids_dset = install(path="bids", source=f"osf://{BIDS}", return_type="item-or-list")
    fmriprep_dset = install(path="bids/derivatives/fmriprep", dataset="bids", source=f"osf://{FMRIPREP}", return_type="item-or-list")

    # get participants
    fmriprep_layout = BIDSLayout(fmriprep_dset.path, config=['bids', 'derivatives'], derivatives=True, validate=False, index_metadata=False)
    bids_layout = BIDSLayout(bids_dset.path, validate=False, index_metadata=False)
    participants = fmriprep_layout.get_subjects()
    # select random sampling of participants:
    subset = rng.choice(participants, N_PARTICIPANTS, replace=False)

    preproc_bold = fmriprep_layout.get(subject=list(subset), task="taskswitch", space="MNI152NLin6Asym", desc="preproc", suffix='bold', extension="nii.gz", return_type="file")
    preproc_mask = fmriprep_layout.get(subject=list(subset), task="taskswitch", space="MNI152NLin6Asym", desc="brain", suffix='mask', extension="nii.gz", return_type="file")
    preproc_regr = fmriprep_layout.get(subject=list(subset), task="taskswitch", desc="confounds", suffix='regressors', extension="tsv", return_type="file")
    fmriprep_dset.get(preproc_bold + preproc_mask + preproc_regr)

    bids_events = bids_layout.get(subject=list(subset), task="taskswitch", suffix="events", extension="tsv", return_type="file")
    bids_dset.get(bids_events)

    auto_reg_order = 1
    ma_order = 1
    min_confounds = ["white_matter", "csf", "cosine[0-9]{2,}", "framewise_displacement", "non_steady_state_outlier[0-9]{2,}", "motion_outlier[0-9]{2,}"]
    ar_list = []
    ma_list = []
    for bold, mask, regr, events in zip(preproc_bold, preproc_mask, preproc_regr, bids_events):
        confounds_df = _select_confounds(regr, min_confounds)
        events_df = pd.read_csv(events, sep='\t')

        # select 100 + BUFFER random voxels (reduce computational load)
        # buffer is added since ARMA does not always converge
        buffer = 50
        mask_img = nib.load(mask)
        mask_idxs = np.array(mask_img.get_fdata().nonzero()).T
        rnd_idxs = rng.choice(mask_idxs, N_VOXELS + buffer, replace=False)
        rnd_mask = np.zeros(mask_img.shape).astype(int)
        rnd_mask[tuple(rnd_idxs.T)] = 1
        rnd_mask_img = nib.Nifti1Image(rnd_mask, mask_img.affine, mask_img.header)

        flm = FirstLevelModel(t_r=2.0, drift_model=None, mask_img=rnd_mask_img, smoothing_fwhm=None, noise_model='ols', minimize_memory=False)

        flm.fit(bold, events=events_df, confounds=confounds_df)

        residual_array = flm.masker_.transform(flm.residuals[0])

        # code modified from:
        # https://github.com/brainiak/brainiak/blob/ab1126a0fb13600d51c883e077602fab56d68f22/brainiak/utils/fmrisim.py#L1273-L1285
        participant_ar_list = []
        participant_ma_list = []
        idx = 0
        while len(participant_ar_list) < N_VOXELS and idx < residual_array.shape[1]:
            try:
                params = ARMA(residual_array[:, idx], [auto_reg_order, ma_order]).fit(disp=False).params
            except (ValueError, LinAlgError):
                idx += 1
                continue

            participant_ar_list.append(abs(params[1:auto_reg_order + 1]))
            participant_ma_list.append(abs(params[auto_reg_order + 1:]))
            idx += 1

        ar_list.append(np.array(participant_ar_list).mean())
        ma_list.append(np.array(participant_ma_list).mean())
    
    # subject means
    ar_mean = np.array(ar_list).mean()
    ma_mean = np.array(ma_list).mean()

    print(f"Average AutoRegressive Coefficient: {ar_mean}")
    print(f"Average Moving Average Coefficient: {ma_mean}")


# useful function from nibetaseries 
# https://github.com/HBClab/NiBetaSeries/blob/04a679c164de1316efbac67a289a9ea33d46f64f/src/nibetaseries/interfaces/nistats.py#L278
def _select_confounds(confounds_file, selected_confounds):
    """Process and return selected confounds from the confounds file
    Parameters
    ----------
    confounds_file : str
        File that contains all usable confounds
    selected_confounds : list
        List containing all desired confounds.
        confounds can be listed as regular expressions (e.g., "motion_outlier.*")
    Returns
    -------
    desired_confounds : DataFrame
        contains all desired (processed) confounds.
    """
    import pandas as pd
    import numpy as np
    import re

    confounds_df = pd.read_csv(confounds_file, sep='\t', na_values='n/a')
    # regular expression to capture confounds specified at the command line
    confound_expr = re.compile(r"|".join(selected_confounds))
    expanded_confounds = list(filter(confound_expr.fullmatch, confounds_df.columns))
    imputables = ('framewise_displacement', 'std_dvars', 'dvars', '.*derivative1.*')

    # regular expression to capture all imputable confounds
    impute_expr = re.compile(r"|".join(imputables))
    expanded_imputables = list(filter(impute_expr.fullmatch, expanded_confounds))
    for imputable in expanded_imputables:
        vals = confounds_df[imputable].values
        if not np.isnan(vals[0]):
            continue
        # Impute the mean non-zero, non-NaN value
        confounds_df[imputable][0] = np.nanmean(vals[vals != 0])

    desired_confounds = confounds_df[expanded_confounds]
    # check to see if there are any remaining nans
    if desired_confounds.isna().values.any():
        msg = "The selected confounds contain nans: {conf}".format(conf=expanded_confounds)
        raise ValueError(msg)
    return desired_confounds


if __name__ == "__main__":
    main()