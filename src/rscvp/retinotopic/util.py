from typing import Literal

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, animation
from matplotlib.patches import Ellipse
from tqdm import tqdm

from neuralib.typing import PathLike
from neuralib.util.verbose import fprint
from stimpyp import AbstractStimlog

__all__ = [
    'compute_wfield_trial_average',
    'combine_cycles_within_trial',
    'plot_retinotopic_circular'
]


def compute_wfield_trial_average(sequences: np.ndarray,
                                 cam_time: np.ndarray,
                                 stim: AbstractStimlog,
                                 pre_time: int | np.ndarray | None = None,
                                 normalize_f: Literal['zscore'] | None = 'zscore',
                                 dff: bool = False,
                                 to_numpy: bool = False) -> list[np.ndarray] | np.ndarray:
    """
    Trial average of fluorescence activity sequences

    `Dimension parameters`:

        N = number of visual stimulation (on-off pairs) = (T * S)

        T = number of trials

        S = number of stim Type

        U = Unique of stim type


    .. seealso:: wfield_manual.rawutils.triggered_average

    :param sequences: `Array[float, [F, H, W]]`
    :param cam_time: Camera time in second. `Array[float, F]`
    :param stim: ``StimlogBase``
    :param pre_time: Pre stim time in second
    :param normalize_f: Image frames normalize function
    :param dff: Divide activity with baseline.
    :param to_numpy: Convert return list to numpy array. i.e., if there is no cycle for each trial
    :return: per stim average. len = U. `Array[float, [F', H, W]`, which F' = Trial averaged = F/S
    """
    i_stim = stim.profile_dataframe['i_stims'].to_numpy()
    i_trial = stim.profile_dataframe['i_trials'].to_numpy()

    interval = stim.stimulus_segment  # (N,)
    starts = interval[:, 0]  # (N,)
    ends = interval[:, 1]  # (N,)

    ustim = list(np.unique(i_stim))

    # If pre-stim time
    if pre_time is None:
        pre_time = 0

    if not isinstance(pre_time, np.ndarray):
        tpre = [pre_time for _ in ustim]
    else:
        tpre = pre_time

    # Calculate from unique stim type
    n_trials = []  # (U,)
    wpads = []  # (U,)
    wdurs = []  # (U,)
    dt = np.mean(np.diff(cam_time))
    for i, ist in enumerate(ustim):
        idx = i_stim == ist
        i_trial = i_trial[idx]
        start = starts[idx]
        end = ends[idx]
        dur = np.max(np.round(end - start))

        n_trials.append(len(i_trial))
        wpads.append(int(np.ceil(tpre[i] / dt)))
        wdurs.append(int(np.ceil(dur / dt)))

    # For each stim calculation
    h, w = sequences.shape[1:]
    ret = [np.zeros([wdurs[ist] + 2 * wpads[ist], h, w], dtype=np.float32)
           for ist in range(len(ustim))]  # (F, H, W)

    for i, t in tqdm(enumerate(i_stim), unit='stim number'):
        ist = ustim.index(t)  # TR
        ii = np.searchsorted(cam_time, starts[i], side='left')  # F

        if ii + wdurs[ist] + wpads[ist] >= len(sequences):
            continue

        image_idx = slice(ii - wpads[ist], ii + wdurs[ist] + wpads[ist])
        act = sequences[image_idx, :, :].astype(np.float32)

        if normalize_f is not None:
            act = zscore_normalized(act, wpads[ist])

        baseline = np.mean(act, axis=0)

        match dff, np.all(baseline > 0):
            case (True, True):
                stim_avg = (act - baseline) / baseline
            case (True, False):
                fprint('skip dff because 0 in baseline image', vtype='warning')
                stim_avg = act - baseline
            case _:
                stim_avg = act - baseline

        ret[ist] += stim_avg / n_trials[ist]

    if to_numpy:
        ret = np.squeeze(ret)

    return ret


def zscore_normalized(act: np.ndarray, f_pre: int) -> np.ndarray:
    if f_pre > 0:
        act = act[:f_pre]
    return (act - np.mean(act, axis=0)) / np.std(act, axis=0)


def combine_cycles_within_trial(trial_avg: list[np.ndarray],
                                n_cycles: list[int],
                                pre_frame: int | list[int] | None = None) -> np.ndarray:
    """
    Combine each cycle within each trial

    .. seealso:: wfield_manual.scripts_pyvstim.combine_loops

    :param trial_avg: list of image array. `Array[float, [F, H, W]]`.
    :param n_cycles: List of number of cycle for each trial.
    :param pre_frame: constant value of S-length list
    :return: Image array after average across cycle within trial. `Array[float, [F', H, W]]`.
    """
    ret = []

    for i, act in enumerate(trial_avg):
        match pre_frame:
            case None:
                tpad = 0
            case int():
                tpad = pre_frame
            case list():
                tpad = pre_frame[i]
            case _:
                raise TypeError('')

        #
        if n_cycles[i] > 1:
            f, w, h = act.shape
            sti_dur = f - 2 * tpad
            cyc_dur = int(sti_dur / n_cycles[i])
            cyc_avg = np.zeros((cyc_dur, w, h), dtype=np.float32)

            for c in range(n_cycles[i]):
                frame_range = slice(c * cyc_dur + tpad, (c + 1) * cyc_dur + tpad)
                cyc_avg += act[frame_range]

            cyc_avg /= float(n_cycles[i])
            ret.append(cyc_avg)

        else:
            ret.append(act)

    return np.array(ret[0])


def plot_retinotopic_circular(
        nframes: int,
        display_dim: tuple[int, int] = (2560, 1440),
        size: int = 300,
        output: PathLike | None = None,
        output_fps: int = 30,
        background_pattern: PathLike | None = None,
        invert_direction: bool = False,
        temporal_to_nasal: bool = True,
):
    """
    Generate circular patch visual stimulation scheme for retinotopy using matplotlib backend.
    **Only for example illustration, not for actual experiment pattern**

    :param nframes: Number of frames per cycle
    :param display_dim: Dimensions of the display
    :param size: Diameter of the circle
    :param output: output image sequence file. If None then show
    :param output_fps: Frames per second of the output image sequence
    :param background_pattern: Background pattern image file
    :param invert_direction: Present in clockwise vs. counter-clockwise
    :param temporal_to_nasal: If the stimulation is presented from temporal to nasal visual field, otherwise, vice versa
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    x, y = display_dim
    ax.set_xlim(0, x)
    ax.set_ylim(0, y)
    ax.axis('off')

    # Load and set the background image
    if background_pattern:
        bg_img = Image.open(background_pattern).convert("RGBA")
        bg_img = bg_img.resize(display_dim, Image.Resampling.LANCZOS)
        bg_array = np.array(bg_img)

        # Ensure the mask dimensions match (height, width, channels)
        bg_array = np.pad(bg_array,
                          ((0, y - bg_array.shape[0]), (0, x - bg_array.shape[1]), (0, 0)),
                          mode="constant",
                          constant_values=192)
    else:
        bg_array = np.full((*display_dim[::-1], 4), [255, 255, 255, 255], dtype=np.uint8)

    # Define circle properties
    diameter = size
    start_pos = (diameter, y / 2)
    circle = Ellipse(start_pos, diameter, diameter, edgecolor=None, facecolor="none")  # Circle properties
    ax.add_patch(circle)

    # Precalculate coordinates for the circular path
    radius_x = (x - diameter) / 2
    radius_y = (y - diameter) / 2  # Ensure the circle doesn't exceed vertical bounds
    center = (x / 2, y / 2)

    angles = np.linspace(0, 2 * np.pi, nframes, endpoint=False)
    if not invert_direction:
        angles *= -1

    if temporal_to_nasal:
        x_path = center[0] + radius_x * np.cos(angles)
    else:
        x_path = center[0] - radius_x * np.cos(angles)
    y_path = center[1] + radius_y * np.sin(angles)

    # Initialize a masked image for the animation
    masked_img = ax.imshow(np.full((*display_dim[::-1], 4), [192, 192, 192, 255], dtype=np.uint8),
                           extent=(0, x, 0, y))

    def animate(i):
        # Create a mask where the circle will reveal the background
        mask = np.full((*display_dim[::-1], 4), [192, 192, 192, 255], dtype=np.uint8)

        yy, xx = np.ogrid[:display_dim[1], :display_dim[0]]
        circle_mask = ((xx - x_path[i]) / (diameter / 2)) ** 2 + ((yy - y_path[i]) / (diameter / 2)) ** 2 <= 1
        mask[circle_mask] = bg_array[circle_mask]
        masked_img.set_data(mask)
        return masked_img,

    ani = animation.FuncAnimation(fig, animate, frames=nframes, interval=1000 / output_fps, blit=True)

    if output is not None:
        ani.save(output, writer='pillow', fps=output_fps)
    else:
        plt.show()
