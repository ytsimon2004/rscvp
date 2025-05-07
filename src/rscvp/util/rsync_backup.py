import json
import os.path
import subprocess
import textwrap
from pathlib import Path


def load_src_dst(name: str = 'phys',
                 src_name: str = 'bkrunch2',
                 dst_name: str = 'nerf2024') -> tuple[str, str]:
    p = Path('../../../res/sync') / 'backup.json'

    with p.open('rb') as f:
        res = json.load(f)
        src = res[name][src_name]
        dst = os.path.dirname(res[name][dst_name])

        return src, dst


def run_rsync(src: str,
              dst: str,
              show_individual: bool = False) -> None:
    """
    Run rsync cli

    :param src: src file path
    :param dst: destination directory path
    :param show_individual:
    :return:
    """
    cmds = ['rsync', '-au', '--delete']

    if show_individual:
        cmds.append('--progress')
    else:
        cmds.append('--info=progress2')

    cmds.extend([src])
    cmds.extend([dst])

    subprocess.check_call(cmds)


def main():
    import argparse
    msg = textwrap.dedent("""\
           Example:
           HISTOLOGY >> python run_backup.py  -N histology -S bkrunch2 -D nerf2024
           PHYSIOLOGY >> python run_backup.py -N analysis -S bkrunch2 -D nerf2024
           """)
    ap = argparse.ArgumentParser(description='argparse for backup data', epilog=msg)

    ap.add_argument('-N', '--name', metavar='NAME', help='sync object name')
    ap.add_argument('-S', '--src', help='source name')
    ap.add_argument('-D', '--dst', help='destination name')

    opt = ap.parse_args()

    src, dst = load_src_dst(opt.name, opt.src, opt.dst)
    run_rsync(src, dst)


if __name__ == '__main__':
    main()
