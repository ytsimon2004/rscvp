from typing import NamedTuple, NewType

__all__ = ['CodeAlias', 'HEADER', 'CodeInfo',
           #
           'CELLULAR_IO',
           'WFIELD_IO',
           'BEH_IO',
           #
           'get_headers_from_code']

CodeAlias = str
HEADER = NewType('HEADER', str)


def h(*header: str) -> list[HEADER]:
    return [HEADER(it) for it in header]


class CodeInfo(NamedTuple):
    code: CodeAlias
    """``CodeAlias``"""

    directory: str
    """Directory name"""

    suffix: str | None
    """For per cell, or csv suffix"""

    summary: str | None
    """For summary data, population analysis"""

    headers: list[HEADER] = tuple()
    """Header(s)"""


CELLULAR_IO: dict[CodeAlias, CodeInfo] = {
    'info': CodeInfo('info', 'info', None, None, h('cell_prob', 'red_cell_prob')),
    # selection
    'np': CodeInfo('np', 'np', 'np', None, h('error_perc')),
    'tr': CodeInfo('tr', 'tr', 'tr', None, h('trial_reliability*')),
    'apc': CodeInfo('apc', 'apc', 'apc', None),
    'md': CodeInfo('md', 'motion', None, 'md'),
    'pres': CodeInfo('pres', 'pres', 'pres', 'pres'),
    'event': CodeInfo('event', 'event', 'event', None, h('event')),
    # visual
    'ta': CodeInfo('ta', 'v_trace', 'vt', None),
    'pa': CodeInfo('pa', 'v_polar', 'vp', 'OSI_DSI', h(
        'preferred_sftf',
        'ori_resp',
        'preferred ori_1', 'OSI_1', 'DSI_1',  # idx referred to `SFTF_IDX`
        'preferred ori_2', 'OSI_2', 'DSI_2',
        'preferred ori_3', 'OSI_3', 'DSI_3',
        'preferred ori_4', 'OSI_4', 'DSI_4',
        'preferred ori_5', 'OSI_5', 'DSI_5',
        'preferred ori_6', 'OSI_6', 'DSI_6',
        'OSI_1_cpx', 'DSI_1_cpx',
        'OSI_2_cpx', 'DSI_2_cpx',
        'OSI_3_cpx', 'DSI_3_cpx',
        'OSI_4_cpx', 'DSI_4_cpx',
        'OSI_5_cpx', 'DSI_5_cpx',
        'OSI_6_cpx', 'DSI_6_cpx')),
    'vc': CodeInfo('vc', 'vis', 'vis', None, h('reliability', 'max_vis_resp', 'perc95_vis_resp')),
    'vf': CodeInfo('vf', 'vf', 'sftf', 'sftf_population'),
    'st': CodeInfo('st', 'st', 'st', 'st_population',
                   h(*['sftf_amp_' + it
                       for it in ['0.04 1', '0.04 4', '0.08 1', '0.08 4', '0.16 1', '0.16 4']])),
    # spatial
    'ba': CodeInfo('ba', 'calbelt', 'calbelt', None),
    'sa': CodeInfo('sa', 'calbelt_sort', None, 'sa'),
    'spv': CodeInfo('spv', 'calbelt_sort_trial', None, 'spv'),
    'cm': CodeInfo('cm', 'correlation_matrix', None, 'cm'),
    'am': CodeInfo('am', 'align_map', None, 'am'),
    'at': CodeInfo('at', 'at', 'at', None),
    'si': CodeInfo('si', 'si', 'si', 'si_cumulative', h('si*', 'shuffled_si*')),
    'slb': CodeInfo('slb', 'slb', 'slb', 'slb_hist', h('nbins_exceed*')),
    'sinb': CodeInfo('sinb', 'sinb', 'sinb', None, h('p_value')),
    'pf': CodeInfo('pf', 'pf', 'pf', 'pf_distribution',
                   h('pf_reliability*', 'pf_width_raw*', 'pf_width*', 'pf_peak*', 'n_pf*')),
    'spr': CodeInfo('spr', 'sparsity', 'sparsity', 'spr_tr_lincorr', h('sparsity*')),
    'ev': CodeInfo('ev', 'ev', 'ev', 'ev', h('ev_trial_avg*')),
    'tcc': CodeInfo('tcc', 'tcc', 'tcc', 'tcc_population', h('trial_cc*')),
    # sig
    'ds': CodeInfo('ds', 'ds', 'ds', None, h('mean_dff*', 'median_dff*', 'perc95_dff*', 'max_dff*')),
    'npc': CodeInfo('npc', 'neuropil_corr', None, 'neuropil_corr', ),
    # concat
    'cf': CodeInfo('cf', 'concat_csv', 'concat', 'concat'),
    # classification
    'cls': CodeInfo('cls', 'cls_summary', 'cls_summary', 'cls_summary'),
    'vpc': CodeInfo('vpc', 'classification', None, 'vpc'),
    'cue': CodeInfo('cue', 'cue', 'cue', 'cue'),
    # topo
    'topo': CodeInfo('topo', 'topo', None, 'topo'),
    # stat
    'var': CodeInfo('var', 'var', 'var', 'var'),
    'stat_ses': CodeInfo('stat_ses', 'stat_ses', 'stat_ses', 'stat_ses'),
    'cord': CodeInfo('cord', 'cord', 'cord', 'cord', h('ap_cords', 'ml_cords', 'ap_cords_scale', 'ml_cords_scale', 'dv_cords')),
    # model
    'lnp_pos': CodeInfo('lnp_pos', 'lnp_pos', 'lnp_pos', 'lnp_pos'),
    'lnp_speed': CodeInfo('lnp_speed', 'lnp_speed', 'lnp_speed', 'lnp_speed'),
    'lnp_lick_rate': CodeInfo('lnp_lick_rate', 'lnp_lick_rate', 'lnp_lick_rate', 'lnp_lick_rate'),
    'lnp_acceleration': CodeInfo('lnp_acc', 'lnp_acc', 'lnp_acc', 'lnp_acc'),
    'bayes_decode': CodeInfo('bayes_decode', 'bayes_decode', 'bayes_decode', 'bayes_decode'),
    'bayes_disengagement': CodeInfo('bayes_disengagement', 'bayes_disengagement', None, 'bayes_disengagement'),
    'rastermap': CodeInfo('rastermap', 'rastermap', None, 'rastermap'),
    # model customized
    'clnp_pos': CodeInfo('clnp_pos', 'clnp_pos', 'clnp_pos', 'clnp_pos'),
    'clnp_speed': CodeInfo('clnp_speed', 'clnp_speed', 'clnp_speed', 'clnp_speed'),
    'clnp_lick_rate': CodeInfo('clnp_lick_rate', 'clnp_lick_rate', 'clnp_lick_rate', 'clnp_lick_rate'),
}

# =============== #
# Wide-Field Data #
# =============== #

WFIELD_IO: dict[CodeAlias, CodeInfo] = {
    'rastermap': CodeInfo('rastermap', 'rastermap_wfield', None, 'rastermap'),
    'retinotopic': CodeInfo('retinotopic', 'retinotopic_map', None, 'retinotopic_map')
}

# =============== #
# Behavioral Data #
# =============== #

BEH_IO: dict[CodeAlias, CodeInfo] = {
    'bs': CodeInfo('bs', 'behavior/sum', None, 'bs'),
    'lick': CodeInfo('lick', 'behavior/lick', None, 'lick'),
}


# ----------- #

def get_headers_from_code(code: CodeAlias, remove_starkey=True) -> list[HEADER]:
    for _, io in CELLULAR_IO.items():
        if code == io.code:
            if remove_starkey:
                return list(map(lambda it: it[:-1] if it.endswith('*') else it, io.headers))
            else:
                return io.headers

    raise ValueError(f'{code} not found')
