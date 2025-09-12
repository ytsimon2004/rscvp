from typing import Literal

from argclz import AbstractParser, argument
from rscvp.util.database import *

__all__ = ['DBtoGspreadOptions']


class DBtoGspreadOptions(AbstractParser):
    DESCRIPTION = 'Push a DB table to a gspread'

    db_type: Literal['generic', 'bayes', 'visual', 'generic_darkness', 'generic_blankbelt', 'generic_vr'] = argument(
        '--db',
        required=True,
        help='push db type'
    )

    push: bool = argument('--push', help='Push to the spreadsheet, otherwise print')

    def run(self):
        match self.db_type:
            case 'generic':
                db = GenericDB
            case 'bayes':
                db = BayesDecodeDB
            case 'visual':
                db = VisualSFTFDirDB
            case 'generic_darkness':
                db = DarknessGenericDB
            case 'generic_blankbelt':
                db = BlankBeltGenericDB
            case 'generic_vr':
                db = VRGenericDB
            case _:
                raise ValueError(f'unknown db type: {self.db_type}')

        if self.push:
            RSCDatabase().upload_to_gspread(db, 'YWAnalysis')
        else:
            print(db)


if __name__ == '__main__':
    DBtoGspreadOptions().main()
