from pathlib import Path

import cv2

from neuralib.typing import PathLike
from rscvp.util.util_sort import anatomical_sort_key


def make_dir_animation(directory: PathLike,
                       output: PathLike,
                       suffix: str = '*.png',
                       fps: int = 20):
    """make animation for all the figures inside a directory"""
    image_files = sorted(Path(directory).glob(suffix), key=anatomical_sort_key)

    first_frame = cv2.imread(image_files[0])
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    for file in image_files:
        frame = cv2.imread(file)
        video_writer.write(frame)

    video_writer.release()
