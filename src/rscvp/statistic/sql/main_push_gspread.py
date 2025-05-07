from typing import Literal

from rscvp.util.database import GenericDB, BayesDecodeDB, VisualSFTFDirDB, RSCDatabase, DarknessGenericDB

from argclz import AbstractParser, argument

__all__ = ['DBtoGspreadOptions']


class DBtoGspreadOptions(AbstractParser):
    DESCRIPTION = 'Push a DB table to a gspread'

    db_type: Literal['generic', 'bayes', 'visual', 'generic_darkness'] = argument(
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
            case _:
                raise ValueError(f'unknown db type: {self.db_type}')

        if self.push:
            RSCDatabase().upload_to_gspread(db, 'YWAnalysis')
        else:
            print(db)


if __name__ == '__main__':
    DBtoGspreadOptions().main()
