# first let's load in all our packages
import matplotlib.pyplot as plt

from mne_bids import (read_raw_bids, BIDSPath)
from mne_hfo.compare import correlation_matrix
from mne_hfo.viz import plot_corr_matrix
from mne_hfo import LineLengthDetector, RMSDetector, HilbertDetector

from pathlib import Path

# root of BIDs dataset
root = Path('../mne-hfo/mne_hfo/tests/data/')

# BIDS entities
subject = '01'
task = 'interictalsleep'
run = '01'
datatype = 'ieeg'

bids_path = BIDSPath(subject=subject, task=task,
                     run=run, datatype=datatype, root=root,
                     suffix='ieeg', extension='.vhdr')
raw = read_raw_bids(bids_path)

kwargs = {
    'filter_band': (80, 250),  # (l_freq, h_freq)
    'threshold': 3,  # Number of st. deviations
    'win_size': 100,  # Sliding window size in samples
    'overlap': 0.25,  # Fraction of window overlap [0, 1]
    'hfo_name': "ripple"
}
kwargs2 = {
    'filter_band' : (80, 250),
    'threshold' : 5,
    'win_size' : 50,
    'overlap' : 0.5,
    'hfo_name' : "ripple"
}

rms_detector = RMSDetector(**kwargs)
ll_detector = LineLengthDetector(**kwargs)
rms_detector2 = RMSDetector(**kwargs2)
ll_detector2 = LineLengthDetector(**kwargs2)
hilbert = HilbertDetector()

# run detector
rms_detector.fit(X=raw)
ll_detector.fit(X=raw)
rms_detector2.fit(X=raw)
ll_detector2.fit(X=raw)
hilbert.fit(X=raw)

comp_kw = {
    'label_method' : 'overlap-predictions',
    'comp_method' : 'cohen-kappa'
}
det_list = [rms_detector, ll_detector, rms_detector2, ll_detector2, hilbert]
corr_matrix = correlation_matrix(det_list, **comp_kw)
plot_corr_matrix(corr_matrix, det_list)
plt.show()
