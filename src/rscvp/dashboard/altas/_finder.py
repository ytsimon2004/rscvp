from pathlib import Path

from rscvp.util.io import get_io_config

from neuralib.util.utils import uglob, filter_matched

__all__ = ['HistPathFinder']


class HistPathFinder:

    def __init__(self, root: Path | None = None,
                 remote_disk: str | None = None):
        if root is None:
            root = get_io_config(remote_disk=remote_disk).histology

        self.data_root = Path(root)

    def find_animal(self) -> list[str]:
        pattern = r'^[A-Z]{2}[0-9]+$'
        strings = [p.stem for p in list(self.data_root.glob('*')) if p.is_dir()]
        return filter_matched(pattern, strings)

    def _find_tif_fname(self, animal: str) -> list[str]:
        tif_path = list((self.data_root / animal / 'resize').glob('*.tif'))
        return [f.stem for f in tif_path]

    def find_glass_slide(self, animal: str) -> list[str]:
        tif_fname = self._find_tif_fname(animal)
        ret = []
        for tif in tif_fname:
            part = tif.split('_')
            ret.append(part[1])

        return sorted(set(ret))

    def find_slice_id(self, animal: str, glass_slide: str) -> list[str]:
        tif_fname = self._find_tif_fname(animal)
        ret = []
        for tif in tif_fname:
            part = tif.split('_')
            if part[1] == glass_slide:
                ret.append(part[2])

        return sorted(set(ret))

    def find_resize_image_path(self, animal: str, glass_slide: str, slice_id: str) -> Path:
        """resize tif"""
        p = self.data_root / animal / 'resize'
        pattern = f'{animal}_{glass_slide}_{slice_id}'
        return uglob(p, f'{pattern}*.tif')

    def find_parsed_csv_path(self, animal: str) -> Path:
        """final output parsed csv"""
        f = 'resize/processed/transformations/labelled_regions/parsed_data/parsed_csv_merge.csv'
        p = self.data_root / animal / f
        if p.exists():
            return p
        else:
            raise FileNotFoundError(f'{p} not found')
