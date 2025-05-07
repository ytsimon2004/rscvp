from typing import Literal, get_args

SERVER_TYPE = Literal['behavior', 'behavior_date', 'histology', 'retinotopic']


def main():
    import argparse
    epilog = """
    python -m rscvp.dashboard.run -B retinotopic -- -F <TRIAL_AVG_TIF_SEQUENCES>
    """

    ap = argparse.ArgumentParser(
        description='run the bokeh server',
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter  # preserve formating
    )

    ap.add_argument(
        '-B', '--bokeh',
        choices=get_args(SERVER_TYPE),
        required=True,
        help='server type',
    )

    ap.add_argument('ARGS', nargs='*')

    opt = ap.parse_args()

    match opt.bokeh:
        case 'behavior':
            from rscvp.dashboard.rig import beh_dashboard
            beh_dashboard.main(opt.ARGS)
        case 'behavior_date':
            from rscvp.dashboard.rig import multi_date_viewer
            multi_date_viewer.main(opt.ARGS)
        case 'histology':
            from rscvp.dashboard.altas import hist_dashboard
            hist_dashboard.main(opt.ARGS)
        case 'retinotopic':
            from rscvp.dashboard.wfield import retinotopic_dashboard
            retinotopic_dashboard.main(opt.ARGS)
        case _:
            raise ValueError(f'unknown server type: {opt.bokeh}')


if __name__ == '__main__':
    main()
