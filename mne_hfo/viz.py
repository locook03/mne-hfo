import mne
import numpy as np
import matplotlib.pyplot as plt
from mne_hfo import merge_channel_events
from mne_hfo.io import create_annotations_df


def plot_hfo_event(raw, detector, eventId, show_filtered=False):
    annotations = detector.hfo_annotations
    mne.viz.set_browser_backend("qt")

    # convert annotations to annotations dataframe
    onset = annotations.onset
    duration = annotations.duration
    label = annotations.description
    sfreq = raw.info["sfreq"]

    # each annotation only has one channel associated with it
    annot_ch_names = [ch[0] for ch in annotations.ch_names]

    # create an annotations dataframe
    annotations_df = create_annotations_df(
        onset, duration, annot_ch_names, annotation_label=label, sfreq=sfreq
    )
    annotations.description = [f"hfo_{ch}" for ch in annot_ch_names]
    raw.set_annotations(annotations)

    merged = merge_channel_events(annotations_df).sort_values("onset")
    print(f"Number of events: {merged.shape[0]}")

    event = merged.iloc[eventId]
    onset = event["onset"]
    duration = event["duration"]
    event_channels = event["channels"]
    electrode = ''.join([c for c in event["channels"][0] if not c.isdigit()])
    orig_indices = event["orig_indices"]

    t_start = max(0, onset - 1)
    t_end = min(raw.tmax, onset + duration + 1)

    # include non event channels
    electrode_channels = [
        ch for ch in raw.ch_names if ''.join([c for c in ch if not c.isdigit()]) == electrode
    ]
    subset = raw.get_data(tmin=t_start, tmax=t_end, picks=electrode_channels)
    t = np.linspace(t_start, t_end, num=subset.shape[1])

    fig, axs = plt.subplots(subset.shape[0], 2, sharex='col', gridspec_kw={'hspace': 0})
    if not isinstance(axs, (list, tuple, np.ndarray)):
        axs = [axs]
    for i, ax in enumerate(axs[:, 0]):
        electrode_ch = electrode_channels[i]
        ax.plot(t, subset[i])
        ax.set_ylabel(electrode_ch, rotation=20, labelpad=20)
        ax.set_yticks([])
        ax.set_xlabel("Time (s)")

        if electrode_ch in event_channels:
            ch_index = event_channels.index(electrode_ch)
            orig_index = orig_indices[ch_index]
            orig_event = annotations_df.iloc[orig_index]
            orig_onset = orig_event.onset
            orig_duration = orig_event.duration
            ax.axvspan(orig_onset, orig_onset + orig_duration, facecolor='red', alpha=0.5)

    subset_filtered = mne.filter.filter_data(
        subset,
        sfreq=sfreq,
        l_freq=detector.l_freq,
        h_freq=detector.h_freq,
        method="fir",
        verbose=False,
    )
    for i, ax in enumerate(axs[:, 1]):
        electrode_ch = electrode_channels[i]
        ax.plot(t, subset_filtered[i])
        ax.set_ylabel(electrode_ch, rotation=20, labelpad=20)
        ax.set_yticks([])
        ax.set_xlabel("Time (s)")

        if electrode_ch in event_channels:
            ch_index = event_channels.index(electrode_ch)
            orig_index = orig_indices[ch_index]
            orig_event = annotations_df.iloc[orig_index]
            orig_onset = orig_event.onset
            orig_duration = orig_event.duration
            ax.axvspan(orig_onset, orig_onset + orig_duration, facecolor='red', alpha=0.5)

    axs[0, 0].set_title("Raw")
    axs[0, 1].set_title("Filtered")
    fig.suptitle(f"Event {eventId} Detections")
    return fig, axs


def plot_corr_matrix(corr_matrix: np.ndarray, det_list: list, fig=None, ax=None):
    """
    Compares similarity between detector results.
    Creates a plot of the comparison values in a len(det_list) x len(det_list) plot.


    The detectors should be fit to the same data.

    Parameters
    ----------
    corr_matrix : np.ndarray
        A numpy 2D matrix with all the comparison values for each detector listed in
        det_list.
    det_list : List
        A list containing all Detector instances. Detectors should already be fit to the
        data.
    fig : matplotlib.figure.Figure (optional)
        The figure to plot the chart.
    ax : matplotlib.axes.Axes (optional)
        The axes to which to plot the chart. If no ax given, it will create a new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object with comparison chart plotted.
    fig : matplotlib.figure.Figure
        Figure object with comparison chart plotted.
    """

    # If no axis is provided, the figure will be created
    if ax is None:
        fig, ax = plt.subplots()
        ax = plt.gca()

    # Creates image using correlation matrix
    im = ax.imshow(corr_matrix, cmap='inferno')
    ax.set_xticks(np.arange(len(det_list)), labels=[det.__class__() for det in det_list])
    ax.set_yticks(np.arange(len(det_list)), labels=[det.__class__() for det in det_list])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(det_list)):
        for j in range(len(det_list)):
            if round(float(corr_matrix[i, j]), 3) > 0.5:
                color = 'k'
            else:
                color = 'w'
            ax.text(j, i, round(float(corr_matrix[i, j]), 3),
                    ha="center", va="center", color=color)

    # Generates colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom")
    ax.set_title("Detector Comparison")

    return fig, ax
