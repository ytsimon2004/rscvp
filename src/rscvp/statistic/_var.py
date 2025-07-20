from rscvp.util.util_gspread import GSPREAD_SHEET_PAGE
from rscvp.visual.util import SFTF_ARRANGEMENT

__all__ = ['GENERIC_HEADERS',
           'SPATIAL_HEADERS',
           'PF_HEADERS',
           #
           'VIS_HEADERS',
           'VIS_DIR_HEADERS',
           'VIS_SFTF_HEADERS',
           #
           'var_page_dict',
           'var_module_dict']

#
GENERIC_HEADERS = ('perc95_dff', 'max_dff')

#
SPATIAL_HEADERS = (
    'si', 'trial_cc', 'ev_trial_avg', 'trial_reliability',
    'speed_score', 'speed_score_run',
    'ap_cords', 'ml_cords', 'ap_cords_scale', 'ml_cords_scale', 'dv_cords'
)

PF_HEADERS = ('n_pf', 'pf_width', 'pf_peak')

#
VIS_HEADERS = (
    'reliability', 'max_vis_resp', 'perc95_vis_resp',
    'ap_cords', 'ml_cords', 'ap_cords_scale', 'ml_cords_scale', 'dv_cords'
)

VIS_DIR_HEADERS = ('dsi', 'osi', 'pdir', 'pori')

VIS_SFTF_HEADERS = ('dff', 'fraction', *SFTF_ARRANGEMENT)


def var_page_dict() -> dict[str, GSPREAD_SHEET_PAGE]:
    s = {
        it: 'ap_place'
        for it in SPATIAL_HEADERS
        if it not in ('ap_cords', 'ml_cords', 'dv_cords')  # special case (in both stat)
    }

    v = {
        it: 'ap_vz'
        for it in VIS_HEADERS + VIS_DIR_HEADERS + VIS_SFTF_HEADERS
        if it not in ('ap_cords', 'ml_cords', 'dv_cords')  # special case
    }

    g = {
        it: 'ap_generic'
        for it in GENERIC_HEADERS
    }

    pf = {
        it: 'ap_place'
        for it in PF_HEADERS
    }

    return {**s, **v, **g, **pf}


def var_module_dict() -> dict[str, str]:
    s = {
        it: 'main_spatial_agg'
        for it in SPATIAL_HEADERS
        if it not in ('ap_cords', 'ml_cords', 'dv_cords')  # special case (in both stat)
    }

    v = {
        it: 'main_visual_agg'
        for it in VIS_HEADERS
        if it not in ('ap_cords', 'ml_cords', 'dv_cords')  # special case
    }

    g = {
        it: 'main_generic_agg'
        for it in GENERIC_HEADERS
    }

    vdir = {
        it: 'main_visual_dir_agg'
        for it in VIS_DIR_HEADERS
    }

    vsftf = {
        it: 'main_visual_sftf_agg'
        for it in VIS_SFTF_HEADERS
    }

    pf = {
        it: 'main_pf_agg'
        for it in PF_HEADERS
    }

    return {**s, **v, **g, **vdir, **vsftf, **pf}
