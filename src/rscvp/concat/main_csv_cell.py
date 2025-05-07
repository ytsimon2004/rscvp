import numpy as np
import polars as pl
from polars.exceptions import SchemaError
from rscvp.util.cli.cli_io import CELLULAR_IO
from rscvp.util.cli.cli_output import DataOutput
from rscvp.util.cli.cli_selection import SelectionOptions

from argclz import AbstractParser
from neuralib.imaging.suite2p import Suite2PResult
from neuralib.util.verbose import fprint, print_save

__all__ = ['ConcatCellCSVOptions']


class ConcatCellCSVOptions(AbstractParser, SelectionOptions):
    DESCRIPTION = 'concat all .csv file (per cell) in separated planes analysis'

    output: DataOutput
    reuse_output = True

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        self.output = self.get_data_output('cf')
        self.concat()
        print_save(self.output.csv_output)

    def concat(self):
        """
        Integrate cell information and concate the csv files and accumulate neuron_id in multiple planes scanning
        if analysis not yet run, then fill in nan
        """

        concate_csv = {
            'neuron_id': pl.Series(dtype=pl.Int64),
            'neuron_ID_accumulate': pl.Series(dtype=pl.Int64),
            'cell_prob': pl.Series(dtype=pl.Float64),
            'plane_idx': pl.Series(dtype=pl.Int64),
            'red_cell_prob': pl.Series(dtype=pl.Float64),
        }

        root = self.get_src_path('suite2p')

        foreach_plane_s2p = list(sorted(root.glob(f'plane*')))  # suite2p data
        foreach_plane = list(sorted(root.glob(f'../plane*')))  # analysis data

        # validate
        match len(foreach_plane_s2p), len(foreach_plane):
            case (0, _) | (_, 0):
                raise FileNotFoundError(f'Error in {root}, or no plane data')
            case (1, _) | (_, 1):
                raise RuntimeError('Only single plane file, no need to concat')
            case (s2p_len, plane_len) if s2p_len != plane_len:
                raise RuntimeError('Data incomplete')

        # list(plane_idx) of dict(string(iscell, redcell) to np.array(cell or redcell probability))
        # list[dict[str, np.array]]
        s2p_pdata = []
        for s2p_dir in foreach_plane_s2p:
            s2p = Suite2PResult.load(s2p_dir, 0.5, runtime_check_frame_rate=None)
            s2p_pdata.append({
                'iscell': s2p.cell_prob,
                'redcell': s2p.red_cell_prob if s2p.has_chan2 else np.full(s2p.n_neurons, np.nan)
            })

        # list[dict[str, pl.DataFrame]]
        csv_data = []
        for p in foreach_plane:
            d = {}
            for csv_path in p.glob(f'*/*.csv'):
                filename = csv_path.parent.name
                data = pl.read_csv(csv_path)
                d[filename] = data

            csv_data.append(d)

        def get_series(k: str, plane_idx: int, patterns: list[str]) -> list[pl.Series]:
            """
            get pl.Series via patterns from specific key

            :param k: 'keys from SELECT_HEADER' or 'fname prefix '
            :param plane_idx: plane index
            :param patterns: `CodeInfo.headers`
            :return:
            """
            ret = []
            for fname, df in csv_data[plane_idx].items():  # type: str, pl.DataFrame

                if fname.startswith(k):

                    fields = list(df.columns)
                    for pattern in patterns:
                        if pattern.endswith('*'):
                            for i, f in enumerate(fields):
                                if f.startswith(pattern[:-1]):
                                    break  # append
                            else:
                                continue
                        else:
                            f = pattern  # field
                            try:
                                i = fields.index(pattern)
                            except ValueError:
                                fprint(f'{pattern} not found', vtype='warning')
                                continue

                        ret.append(df[f])
                        del fields[i]  # remove matched field to prevent from captured by other patterns again

            return ret

        def concat_to(field: str, dat: pl.Series, length: int) -> None:
            """
            concat pl.Series

            :param field: field name of ds
            :param dat: data need to be concatenated to original ds
            :param length: if field not found, numbers of np.nan concat
            :return:
            """
            try:
                ds = concate_csv[field]
            except KeyError:
                ds = pl.Series(np.full((length,), np.nan), nan_to_null=True)

            try:
                concate_csv[field] = pl.concat([ds, dat])
            except SchemaError:  # casting special dtype
                ds = ds.cast(dat.dtype)
                concate_csv[field] = pl.concat([ds, dat])

        for s2p_d in s2p_pdata:
            concat_to('cell_prob', pl.Series(s2p_d['iscell']), 0)
            concat_to('red_cell_prob', pl.Series(s2p_d['redcell']), 0)

        #
        for plane_idx in range(len(csv_data)):
            length = len(concate_csv['neuron_id'])
            neuron_id = get_series('np', plane_idx, ['neuron_id'])[0]
            concat_to('neuron_id', neuron_id, length)
            concat_to('plane_idx', pl.Series(np.full_like(neuron_id, plane_idx)), length)

            for code, info in CELLULAR_IO.items():
                headers = [it for it in info.headers if it not in ('cell_prob', 'red_cell_prob')]
                for data in get_series(info.directory, plane_idx, headers):
                    concat_to(str(data.name), data, length)

        concate_csv['neuron_ID_accumulate'] = np.arange(len(concate_csv['neuron_id']))

        try:
            ret = pl.DataFrame(concate_csv)
        except ValueError as e:
            fprint(f'{e}', vtype='error')
            raise KeyError('check csv header are the same across plane for concat')

        ret_csv = ret.write_csv(self.output.csv_output)

        return ret_csv


if __name__ == '__main__':
    ConcatCellCSVOptions().main()
