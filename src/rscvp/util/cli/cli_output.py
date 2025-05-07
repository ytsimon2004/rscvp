from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from neuralib.util.utils import joinn

__all__ = [
    'DataOutput',
    'TempDirWrapper'
]


class DataOutput(NamedTuple):
    name: str | None
    """cli_io.code"""
    directory: Path | TempDirWrapper
    """cli_io.directory(code)"""
    filename: str | None
    """cli_io.fig(cell)"""
    summary_filename: str | None
    """cli_io.figs([cell])"""

    #
    default_fig_ext: str = '.pdf'

    @classmethod
    def of_tmp_output(cls, name: str, directory: Path,
                      filename: str | None,
                      summary_filename: str | None) -> DataOutput:
        """For testing"""
        wrapper = TempDirWrapper(directory.parent)
        return DataOutput(name, wrapper, filename, summary_filename)

    def with_fig_ext(self, ext: str) -> DataOutput:
        return self._replace(default_fig_ext=ext)

    def data_output(self, name: str, ext: str | None = None) -> Path | None:
        """
        Create file or dir under directory
        :param name: filename or file extension name
        :param ext
        :return:
        """
        if isinstance(self.directory, TempDirWrapper):
            return

        ret = (self.directory / name)

        if ext is not None:
            ret = ret.with_suffix(ext)

        return ret

    def figure_output(self, neuron: int = None, *suffix, sep='-') -> Path | None:
        """for plot all neuron"""
        if isinstance(self.directory, TempDirWrapper):
            return

        filename = self.filename
        if filename is None:
            raise RuntimeError(f'output {self.name} in INFO do not contain figure name')

        filename = joinn(sep, filename, neuron, *suffix)
        return self.directory / (filename + self.default_fig_ext)

    def summary_figure_output(self, *suffix, sep='-') -> Path | None:
        """
        For population analysis in single dataset, or analysis based on local csv

        :rtype: object
        :param suffix: filename suffix, any non-str will convert to str,
            except None which will be ignored.
        :param sep:
        :return:
        """
        if isinstance(self.directory, TempDirWrapper):
            return

        filename = self.summary_filename
        if filename is None:
            raise RuntimeError(f'output {self.name} in INFO do not contain summary figure name')

        filename = joinn(sep, filename, *suffix)
        return self.directory / (filename + self.default_fig_ext)

    @property
    def csv_output(self) -> Path | None:
        if isinstance(self.directory, TempDirWrapper):
            return
        return self.directory / (self.filename + '.csv')

    def mk_subdir(self, folder: str, fname: str, suffix: str) -> Path:
        """Create another file under a sub-folder"""
        filepath = (self.directory / folder / fname).with_suffix(suffix)
        if not filepath.parent.exists():
            filepath.parent.mkdir(exist_ok=True, parents=True)

        return filepath


class TempDirWrapper:
    """for testing usage"""

    def __init__(self, parent: Path, *,
                 truediv_as_none: bool = False):
        """

        :param parent: parent attr
        :param truediv_as_none:
        """
        import tempfile

        self.parent = parent
        self.tmp = tempfile.TemporaryDirectory(prefix='.tmp')
        self.truediv_as_none = truediv_as_none

    def __truediv__(self, other) -> Path | None:
        """path usage"""
        if self.truediv_as_none:
            return
        return Path(self.tmp.name) / other

    def glob(self, pattern: str):
        raise NotImplementedError('glob not implemented in TempDirWrapper')
