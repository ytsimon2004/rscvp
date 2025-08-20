import logging
from pathlib import Path
from typing import Literal, Iterable, ClassVar, TypeVar, Any

from typing_extensions import Self

from argclz import argument
from neuralib.io import mkdir_version
from neuralib.util.utils import uglob, joinn
from rscvp.util.io import IOConfig, get_io_config, DATA_SRC_TYPE
from .cli_io import CELLULAR_IO, WFIELD_IO, BEH_IO, CodeAlias
from .cli_output import DataOutput, TempDirWrapper

__all__ = [
    'DAQ_TYPE',
    'Region',
    'CommonOptions',
    #
    'cast_opt'
]

DAQ_TYPE = Literal['2P', '1P', 'no_cam']
"""experiment type for data acquisition"""

Region = str
"""region name for the imaging"""


class CommonOptions:
    """Common options for physiological data path"""

    GROUP_PRIMARY: ClassVar[str] = 'Data Primary Key Options'
    """group data primary key options"""

    GROUP_IO: ClassVar[str] = 'Data Input/Output Options'
    """group data input/output options"""

    # ----- PRIMARY KEY ----- #

    exp_date: str = argument(
        '-D', '--ed', '--exp-date',
        metavar='YYMMDD',
        required=True,
        group=GROUP_PRIMARY,
        help='experiment date',
    )

    animal_id: str = argument(
        '-A', '--id', '--animal-id', '--mouse-id',
        metavar='NAME',
        required=True,
        group=GROUP_PRIMARY,
        help='animal (mouse) ID or experimental name (if multiple sessions in one day)',
    )

    daq_type: DAQ_TYPE = argument(
        '--exp-type', '--daq', '--daq-type',
        metavar='RECORDING',
        default='2P',
        group=GROUP_PRIMARY,
        help='experiment type for data acquisition',
    )

    username: str = argument(
        '-U', '--user',
        metavar='USER_NAME',
        default='YW',
        group=GROUP_PRIMARY,
        help='user name',
    )

    run_number: str | None = argument(
        '--run-num', '--run-batch',
        group=GROUP_PRIMARY,
        default=None,
        help='running time for a certain protocol in single day, used in new stimpy version',
    )

    rec_region: Region | None = argument(
        '--region',
        group=GROUP_PRIMARY,
        default=None,
        help='region name for the imaging',
    )

    # ----- DATA IO ----- #

    remote_disk: str | None = argument(
        '--disk',
        metavar='NAME',
        default=None,
        group=GROUP_IO,
        help='remote disk name'
    )

    mnt_prefix: str = argument(
        '--mount',
        metavar='PATH',
        default='/mnt',
        group=GROUP_IO,
        help='mount path prefix',
    )

    reuse_output: bool = argument(
        '-R', '--re',
        group=GROUP_IO,
        help='reuse the latest file for storage. If True, store the new item in '
             'latest version directory, otherwise create a new version suffix'
    )

    use_default: bool = argument(
        '--use-default',
        group=GROUP_IO,
        help='use default io config'
    )

    debug_mode: bool = argument(
        '--debug',
        group=GROUP_IO,
        help='preview qt without save and generate files'
    )

    # after extend
    __config: IOConfig | None = None

    def get_io_config(self) -> IOConfig:
        """get io config based on the running machine"""
        if self.__config is None:
            self.__config = get_io_config(
                remote_disk=self.remote_disk,
                force_use_default=self.use_default,
                mnt_prefix=self.mnt_prefix,
            )
        return self.__config

    @property
    def stimpy_filename(self) -> str:
        """stimpy filename with given primary key options"""
        return f'{self.exp_date}_{self.animal_id}__{self.daq_type}_{self.username}'

    @property
    def pyvstim_filename(self) -> str:
        """pyvstim legacy filename with given primary key options"""
        return f'{self.exp_date}_{self.animal_id}_{self.daq_type}_{self.username}'

    @property
    def cache_directory(self) -> Path:
        """cached directory for physiological data processing"""
        return self.get_io_config().cache

    @property
    def statistic_dir(self) -> Path | TempDirWrapper:
        """statistics directory under base physiology directory"""
        return self.get_io_config().statistic_dir

    @property
    def phy_base_dir(self) -> Path:
        """base physiology directory"""
        return self.get_io_config().phy_base_dir

    @property
    def concat_csv_path(self) -> Path:
        """get concat csv from multiple ETL optic plans"""
        p = self.phy_base_dir / self.stimpy_filename
        p = uglob(uglob(p, 'concat_plane_v*', is_dir=True), 'concat_csv_v*', is_dir=True)

        if not p.exists():
            raise FileNotFoundError(f'{p} not exit')

        return uglob(p, '*.csv')

    def extend_src_path(self, *item: str):
        """
        find the path with specified item text, return modified data_root.

        **Example**

        `extend_src_path(opt, '20200101', 'AA01')`

        input:
            data/path
        output (if existed):
            data/path/20200101_AA01
        input:
            data/path/%/sub
        output (if existed):
            data/path/20200101_AA01/sub


        :param item: name components. should be in order.
        """
        if any([len(it) == 0 for it in item]):
            raise ValueError('')

        config = self.get_io_config()

        # not assign or foreach usage
        if ('%' in config.phy_animal_dir.parts) or (self.stimpy_filename not in config.phy_animal_dir.parts):
            d = config.source_root['physiology']
            g = '*' + '*'.join(item) + '*'  # i.e., '*210302*YW008*1P*YW*'
            if self.run_number is not None:
                g += f'/run{self.run_number}_*'

            f = uglob(d, g, is_dir=True)
            assert f.name in (self.stimpy_filename, self.pyvstim_filename), 'f.name != self.filename'

            self.__config.phy_animal_dir = f

    def foreach_dataset(self, **field) -> Iterable[Self]:
        """Foreach different animal and experimental date.

        ::

            $ python ... -A YW01,YW02 -D 210513,210519 ...

        >>> opt: CommonOptions
        >>> for _ in opt.foreach_dataset():
        ...     print(opt.animal_id, opt.exp_date)
        YW01 210513
        YW02 210519

        support foreach on other fields defined in other option class.

        ::

            $ python ... -A YW01,YW02 -D 210513,210519 --plane 0,1 ...

        >>> for _ in opt.foreach_dataset(plane_idx=int):
        ...     print(opt.animal_id, opt.exp_date, opt.plane_idx)
        YW01 210513 0
        YW02 210519 1

        :param field: foreach options, beside animal and exp date.
        """

        root = self.get_io_config().source_root.copy()

        # store value from cli
        old_animal = self.animal_id
        old_date = self.exp_date
        old_fields = {
            f: getattr(self, f, None)
            for f in field
        }

        animal_list = old_animal.split(',')
        date_list = old_date.split(',')

        #
        try:
            field_list = {
                f: None if v is None else v.split(',')
                for f, v in old_fields.items()
            }

        except AttributeError as e:
            print(repr(e))
            raise RuntimeError('only single animal?')

        #
        if len(animal_list) != len(date_list):
            raise RuntimeError(f'number mis-match : {old_animal} != {old_date}')
        for f in field_list.keys():
            v = field_list[f]
            if v is None:
                field_list[f] = [None] * len(animal_list)
            elif len(v) == len(animal_list):
                field_list[f] = list(map(field[f], v))
            else:
                raise RuntimeError(f'{f} number mis-match : {old_animal} != {old_fields[f]}')

        try:
            # generate each opt
            for i in range(len(animal_list)):
                self.animal_id = animal_list[i]
                self.exp_date = date_list[i]
                self.__config.source_root = root.copy()
                for f, v in field_list.items():
                    setattr(self, f, v[i])

                self.extend_src_path(self.exp_date, self.animal_id)
                yield self
        finally:
            # restore to cli given value
            self.animal_id = old_animal
            self.exp_date = old_date
            self.__config.source_root = root.copy()
            for f, v in old_fields.items():
                setattr(self, f, v)

    def get_src_path(self, src: DATA_SRC_TYPE) -> Path:
        """
        get path from specific src type

        :param src: ``DATA_SRC_TYPE``
        :return: path
        """
        return getattr(self.get_io_config(), src)

    def get_data_output(self, code: CodeAlias,
                        *prefix: str,
                        running_epoch: bool = False,
                        virtual_env: bool = False,
                        latest: bool = False,
                        output_type: Literal['cellular', 'wfield', 'behavior'] = 'cellular') -> DataOutput:
        """
        Get ``DataOutput``

        :param code: ``cli_io.CODE``
        :param prefix: Directory prefix for analysis condition. (i.e., session, signal type)
        :param running_epoch: Whether limit analysis in running epoch, If True, add `_run` flag in directory and filename
        :param virtual_env: Whether data is calculated in based on virtual environment position space
        :param latest: If True, store the new item in latest version file, otherwise create a new version
        :param output_type Output data type
        :return: ``DataOutput``
        """
        output_directory = self.phy_base_dir / self.stimpy_filename

        match output_type:
            case 'cellular':
                dy = CELLULAR_IO
                plane_index = getattr(self, 'plane_index', None)
                if plane_index is not None:
                    output_directory = output_directory / f'plane{plane_index}'
                else:
                    output_directory = output_directory / f'concat_plane_v0'
            case 'wfield':
                dy = WFIELD_IO
            case 'behavior':
                dy = BEH_IO
            case _:
                raise ValueError(f'output type {output_type} not supported')

        code_io = dy[code]
        d = code_io.directory
        x = code_io.suffix
        s = code_io.summary

        #
        if None in prefix:
            prefix = tuple(x for x in prefix if x is not None)
        if len(prefix):
            d += f"_{joinn('-', *prefix)}"

        if running_epoch:
            d += '_run'
            x += '_run'

        if virtual_env:
            d += '_vr'
            x += '_vr'

        filename = f'{self.exp_date}_{self.animal_id}_{x}'
        summary_filename = None if s is None else f'{self.exp_date}_{self.animal_id}_{s}'

        if not self.debug_mode:
            output_directory = mkdir_version(output_directory, d, self.reuse_output or latest)
            return DataOutput(code, output_directory, filename, summary_filename)
        else:
            return DataOutput.of_tmp_output(code, output_directory, filename, summary_filename)

    # logging
    logger: logging.Logger | None = None

    def setup_logger(self, caller_name: str | None = None):
        """setup logging for current class"""
        from neuralib.util.logging import setup_clogger
        self.logger = setup_clogger(level=logging.DEBUG, caller_name=caller_name)


T = TypeVar('T')


def cast_opt(t: type[T], o: Any) -> T:
    if not isinstance(o, t):
        raise TypeError(f'{o} not a {t.__name__}')
    return o
