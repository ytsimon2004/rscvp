from typing import TypeVar

from rscvp.statistic.cli_gspread import CliGspreadGenerator, CliGspreadLUT
from rscvp.util.cli import Region

from stimpyp import Session

T = TypeVar('T')

PERSISTENCE_AGG_TYPE: dict[T, CliGspreadLUT] = {
    # ====================== #
    # topo celltype fraction #
    # ====================== #

    'topo_celltype': CliGspreadLUT(
        'apcls_tac',
        ['Data', 'region', 'n_planes'],
        module_prefix='rscvp.statistic.persistence_agg',
        file='main_topo_celltype',
        opt_args=[
            '--smooth', '3',
            '--ctype', 'spatial'
        ]
    ),

    # =================================== #
    # decode error for each position bins #
    # =================================== #

    'position_bins_error_group': CliGspreadLUT(
        'apcls_tac',
        ['Data', 'region', 'n_planes'],
        module_prefix='rscvp.statistic.persistence_agg',
        file='main_decode_err',
        group_mode=True,
        opt_args=[
            '--type', 'position_bins_error',
            '--CV', 'odd',
            '--random', '200',
            '-s', 'light',
            '-w', '100',
            '--used_session', 'light'
        ]
    ),

    'position_bins_error_foreach': CliGspreadLUT(
        'apcls_tac',
        ['Data', 'region', 'n_planes'],
        module_prefix='rscvp.statistic.persistence_agg',
        file='main_decode_err',
        group_mode=False,
        opt_args=[
            '--type', 'position_bins_error',
            '--CV', 'odd',
            '--random', '200',
            '-s', 'light',
            '-w', '100',
            '--used_session', 'light'
        ]
    ),

    # ========================= #
    # decoding confusion matrix #
    # ========================= #

    'confusion_matrix_group': CliGspreadLUT(
        'apcls_tac',
        ['Data', 'region', 'n_planes'],
        module_prefix='rscvp.statistic.persistence_agg',
        file='main_decode_err',
        group_mode=True,
        opt_args=[
            '--type', 'confusion_matrix',
            '--CV', 'odd',
            '--random', '200',
            '-s', 'light',
            '-w', '100',
            '--used_session', 'light'
        ]
    ),

    'confusion_matrix_foreach': CliGspreadLUT(
        'apcls_tac',
        ['Data', 'region', 'n_planes'],
        module_prefix='rscvp.statistic.persistence_agg',
        file='main_decode_err',
        group_mode=False,
        opt_args=['--type', 'confusion_matrix',
                  '--CV', 'odd',
                  '--random', '200',
                  '-s', 'light',
                  '-w', '100',
                  '--used_session', 'light']
    ),

    # ============================================== #
    # sorted trial-averaged position binned activity #
    # ============================================== #

    'sorted_pos_bin_ldl': CliGspreadLUT(
        'ap_ldl',
        ['Data', 'region', 'n_planes', 'prot'],
        module_prefix='rscvp.statistic.persistence_agg',
        file='main_trial_avg_position',
        opt_args=['--sort', 'light_bas']  # sort index
    ),

    'sorted_pos_bin': CliGspreadLUT(
        'ap_place',
        ['Data', 'region', 'n_planes', 'prot'],
        module_prefix='rscvp.statistic.persistence_agg',
        file='main_trial_avg_position',
        opt_args=['--sort', 'light']  # sort index
    ),
    # ============================================== #
    # alignment position binned activity based on si #
    # ============================================== #

    'sorted_si_alignment': CliGspreadLUT(
        'ap_place',
        ['Data', 'region', 'n_planes'],
        module_prefix='rscvp.statistic.persistence_agg',
        file='main_si_sorted_occ',
        opt_args=['--top', '500']
    ),
    # =========== #
    # Visual Maps #
    # =========== #

    'sftf_map': CliGspreadLUT(
        'apcls_tac',
        ['Data', 'region', 'n_planes'],
        module_prefix='rscvp.statistic.persistence_agg',
        file='main_sftf_tuning',
        opt_args=['--sftf', '0.04,4',
                  '--dir-agg', 'max']
    )

}


def run_persistence_agg(t: T, region: Region | None = None,
                        session: Session | None = None,
                        used_session: Session | None = None,
                        remote_disk: str | None = None):
    """Based on spreadsheet data to do the persistence aggregate, and plot"""
    lut: CliGspreadLUT = PERSISTENCE_AGG_TYPE[t]
    cli = CliGspreadGenerator(lut, region=region, session=session, used_session=used_session, remote_disk=remote_disk)
    cli.call()


def main():
    import argparse
    ap = argparse.ArgumentParser(description='run persistence-based statistic')
    ap.add_argument('-T', '--type', help='analysis/plot type', choices=list(PERSISTENCE_AGG_TYPE.keys()))
    ap.add_argument('-R', '--region', type=Region, default=None, help='region-specific analysis')
    ap.add_argument('--session', default=None, help='Session')
    ap.add_argument('--used_session', default=None, help='Session')
    ap.add_argument('--group-region', action='store_true', help='whether group as region. i.e., aRSC/pRSC')
    ap.add_argument('--remote', default=None, help='Remote disk name for remotely testing using locally resources')

    opt = ap.parse_args()

    run_persistence_agg(opt.type, opt.region, opt.session, opt.used_session, opt.remote)


if __name__ == '__main__':
    main()
