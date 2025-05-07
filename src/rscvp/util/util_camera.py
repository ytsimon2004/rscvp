import numpy as np

__all__ = ['truncate_video_to_pulse']


def truncate_video_to_pulse(track_res: np.ndarray,
                            camera_time: np.ndarray) -> np.ndarray:
    """
    Truncate the video frame number in a tracking result to match with the camera event pulse.
    Only Applicable if the tracked result has more frame than camera pulse

    `Dimension parameters`:

        F = number of tracking frames

        P = number of camera's pulse

        T = number of track object type

    :param track_res: (F, T) | (F,)
    :param camera_time: (P, ) number of camera's pulse
    :return: (P,)
    """
    n_track_data = track_res.shape[0]
    n_cam_pulse = len(camera_time)

    if n_track_data <= n_cam_pulse:
        raise ValueError('tracking data frames must be more than camera pulses for truncation to be needed')

    diff = n_track_data - n_cam_pulse

    return track_res[:-diff]
