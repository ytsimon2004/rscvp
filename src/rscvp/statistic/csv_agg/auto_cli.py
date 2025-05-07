from rscvp.statistic._var import var_page_dict, var_module_dict
from rscvp.statistic.cli_gspread import CliGspreadLUT, CliGspreadGenerator
from rscvp.util.cli.cli_statistic import StatisticOptions
from rscvp.util.util_gspread import GSPREAD_SHEET_PAGE

from argclz import AbstractParser, argument

__all__ = ['AutoCLIAgg']


class AutoCLIAgg(AbstractParser, StatisticOptions):
    DESCRIPTION = 'based on gspread to find the data info for statistic'

    _agg_name: str | None = argument(
        '--agg',
        metavar='main_*',
        default=None,
        help='agg module name, if None, used the specified name in code_io'
    )

    arg_lut: CliGspreadLUT
    cli: CliGspreadGenerator

    def post_parsing(self):
        self.arg_lut = CliGspreadLUT(
            self.page,
            ['Data', 'region', 'n_planes', 'prot'],
            module_prefix='rscvp.statistic.csv_agg',
            file=self.agg_name,
            group_mode=False
        )

        self.cli = CliGspreadGenerator(
            self.arg_lut,
            session=self.session,
            used_session=self.used_session,
            remote_disk=self.remote_disk
        )

    def run(self):
        self.post_parsing()
        self.cli.call(self._option_args())

    @property
    def page(self) -> GSPREAD_SHEET_PAGE:
        """
        general mode -> infer from code_info
        session mode -> specified by sheet_name opt
        """
        sheet_name = self.sheet_name
        if sheet_name is not None:
            return sheet_name

        # infer
        try:
            return var_page_dict()[self.header]
        except KeyError:
            raise KeyError(f'{self.header} not valid or can not be inferred, please specify one variable in '
                           f'{list(var_page_dict().keys())}, OR use arg --page manually')

    @property
    def agg_name(self) -> str:
        agg_name = self._agg_name
        if agg_name is not None:
            return agg_name

        return var_module_dict()[self.header]

    def _option_args(self) -> list[str]:
        ret = []
        if self.truncate_session_agg:
            ret.extend(['--trunc-session'])

        ret.extend(['-H', self.header])
        ret.extend(['--page', self.page])

        if self.update:
            ret.extend(['--update'])

        return ret


if __name__ == '__main__':
    AutoCLIAgg().main()
