from pathlib import Path
import matplotlib.pyplot as plt
from mne_bids import BIDSPath, read_raw_bids
from mne_hfo.viz import plot_hfo_event
import mne
from mne_hfo import RMSDetector


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
    'filter_band': (80, 250), # (l_freq, h_freq)
    'threshold': 3, # Number of st. deviations
    'win_size': 100, # Sliding window size in samples
    'overlap': 0.25, # Fraction of window overlap [0, 1]
    'hfo_name': "ripple"
}

rms_detector = RMSDetector(**kwargs)

# run detector
rms_detector.fit(X=raw)
# annots = rms_detector.to_data_frame(format="bids")
annots = rms_detector.hfo_annotations
for i in range(0,36):
    plot_hfo_event(raw, rms_detector, i)
    plt.savefig(f"/home/logan/nspm/hfo_figures/Event{i}.jpg")
#plot_hfos(raw, annots)