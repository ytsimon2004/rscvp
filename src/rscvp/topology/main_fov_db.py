import json

from argclz import AbstractParser
from rscvp.util.cli import SQLDatabaseOptions, StimpyOptions
from rscvp.util.database import FieldOfViewDB

__all__ = ['InsertFOVDBOptions']


class InsertFOVDBOptions(AbstractParser, StimpyOptions, SQLDatabaseOptions):
    DESCRIPTION = ''

    _default_objective_magnification = 16
    _kwargs = {'page': 'fov_table'}

    def run(self):
        self.extend_src_path(self.exp_date, self.animal_id, self.daq_type, self.username)
        self.populate_database()

    def populate_database(self):
        d, n, rot, mag = self._import_imaging_paras()
        ma, mp, lp, la = self._import_fov_paras()

        db = FieldOfViewDB(
            date=self.exp_date,
            animal=self.animal_id,
            max_depth=d,
            n_planes=n,
            objective_rotation=rot,
            objective_magnification=mag,
            medial_anterior=ma,
            medial_posterior=mp,
            lateral_anterior=la,
            lateral_posterior=lp
        )

        self.print_replace(db)

    def _import_imaging_paras(self):
        depth = self.get_primary_key_field('depth', **self._kwargs)
        n_planes = self.get_primary_key_field('n_planes', **self._kwargs)
        rot = self.get_primary_key_field('rotation', **self._kwargs)
        mag = self._default_objective_magnification

        return depth, n_planes, rot, mag

    def _import_fov_paras(self):
        def _json(cords: str):
            """Convert coordinate string like '0;-1.17' to JSON array [0.0, -1.17]"""
            coords_str = cords.replace(';', ',')
            coords_list = [float(x.strip()) for x in coords_str.split(',')]
            return json.dumps(coords_list)

        ma = self.get_primary_key_field('loc_MA', **self._kwargs)
        mp = self.get_primary_key_field('loc_MP', **self._kwargs)
        lp = self.get_primary_key_field('loc_LP', **self._kwargs)
        la = self.get_primary_key_field('loc_LA', **self._kwargs)

        return _json(ma), _json(mp), _json(lp), _json(la)


if __name__ == '__main__':
    InsertFOVDBOptions().main()
