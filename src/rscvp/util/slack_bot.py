import functools
import socket
from datetime import datetime
from pathlib import Path

from neuralib.tools.slack_bot import send_slack_message
from neuralib.typing import PathLike

__all__ = ['slack_bot']


def slack_bot(*, env_file: PathLike | None = None, timestamp: bool = True):
    """sending error message to slack if raise error"""
    if env_file is None:
        env_file = Path(__file__).parents[3] / 'res' / 'env' / 'slack.env'

    def _decorator(f):
        @functools.wraps(f)
        def _bot(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except BaseException as e:
                text = f'`{socket.gethostname()}` ERROR: {e} in `{Path(f.__code__.co_filename).name}`'

                # find opt as first arg, get detailed information
                if len(args) > 0:
                    opt = args[0]
                    exp_date = getattr(opt, 'EXP_DATE', None)
                    animal = getattr(opt, 'ANIMAL', None)

                    if exp_date is not None and animal is not None:
                        text += f'while running DATA: `{exp_date}_{animal}`'

                    if timestamp:
                        text += f'at `{datetime.today().strftime("%y-%m-%d %H:%M:%S")}`'

                send_slack_message(env_file, text)

                raise e

        return _bot

    return _decorator
