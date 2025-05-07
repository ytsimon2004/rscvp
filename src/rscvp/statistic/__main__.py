from argclz.commands import parse_command_args
from .csv_agg.main_pf_agg import PFStatAggOptions
from .csv_agg.main_spatial_agg import SpatialStatAggOptions
from .csv_agg.main_visual_agg import VisStatAggOptions
from .csv_agg.main_visual_dir_agg import VisDirStatAggOption
from .csv_agg.main_visual_sftf_agg import VZSFTFAggOption
from .main_normality import NormTestOption
from .main_para_mtx import ParaCorrMatOptions
from .main_ses_overview import OverviewSessionStat
from .main_ses_var import SpatialSessionStat

parse_command_args(
    usage='python -m rscvp.stat CMD ...',
    description='variable statistic and comparison',
    parsers=dict(
        stat_corr=ParaCorrMatOptions,
        stat_norm=NormTestOption,
        ses_var=SpatialSessionStat,
        ses_overview=OverviewSessionStat,
        #
        spatial_stat=SpatialStatAggOptions,
        pf_stat=PFStatAggOptions,
        vz_stat=VisStatAggOptions,
        vzdir_stat=VisDirStatAggOption,
        vzsftf_stat=VZSFTFAggOption
    )
)
