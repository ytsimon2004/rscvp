from datetime import datetime
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np


class RigScreenCalibrate(NamedTuple):
    rig_type: str
    screen_type: str
    date: datetime
    value: np.ndarray


RIG_1 = RigScreenCalibrate(
    'rig1',
    screen_type='asus',
    date=datetime(year=2024, month=3, day=7),
    value=np.array([1.2, 11.2, 23.7, 34.8, 47.5, 62, 75.0, 99.4])
)


def screen_contrast_calibrate(rc: RigScreenCalibrate):
    v = rc.value
    x = np.linspace(0, 1, len(v))
    plt.plot(x, v)

    xx = x.min(), x.max()
    plt.plot(xx, (v.min(), v.max()))
    plt.xlabel('gray contrast')
    plt.ylabel('lux')
    plt.show()


def main():
    screen_contrast_calibrate(RIG_1)


if __name__ == '__main__':
    main()
