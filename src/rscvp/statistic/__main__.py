from argclz.commands import parse_command_args
from .csv_agg.main_pf_agg import PFAggOptions
from .csv_agg.main_spatial_agg import SpatialAggOptions
from .csv_agg.main_visual_agg import VisAggOptions
from .csv_agg.main_visual_dir_agg import VisDirAggOption
from .csv_agg.main_visual_sftf_agg import VisSFTFAggOptions
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
        spatial_stat=SpatialAggOptions,
        pf_stat=PFAggOptions,
        vz_stat=VisAggOptions,
        vzdir_stat=VisDirAggOption,
        vzsftf_stat=VisSFTFAggOptions
    )
)
