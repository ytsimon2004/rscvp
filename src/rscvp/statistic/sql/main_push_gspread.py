from typing import Literal

from argclz import AbstractParser, argument
from rscvp.util.database import *

__all__ = ['DBtoGspreadOptions']


class DBtoGspreadOptions(AbstractParser):
    DESCRIPTION = 'Push a local sql database to a self gspread'

    db_type: Literal['base', 'bayes', 'sftf', 'dark', 'blank', 'vr'] = argument(
        '--db',
        required=True,
        help='push db type'
    )

    push: bool = argument('--push', help='Push to the gspread, otherwise print')

    def run(self):
        match self.db_type:
            case 'base':
                db = BaseClassDB
            case 'bayes':
                db = BayesDecodeDB
            case 'sftf':
                db = VisualSFTFDirDB
            case 'dark':
                db = DarkClassDB
            case 'blank':
                db = BlankClassDB
            case 'vr':
                db = VRClassDB
            case _:
                raise ValueError(f'unknown db type: {self.db_type}')

        if self.push:
            RSCDatabase().submit_gspread(db, 'YWAnalysis')
        else:
            print(db)


if __name__ == '__main__':
    DBtoGspreadOptions().main()
