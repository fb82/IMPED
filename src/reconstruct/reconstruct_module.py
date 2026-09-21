import os
import shutil
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pycolmap

from core import set_args


class reconstruct_module:
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = True
        self.add_to_cache = False

        self.args = {
            'id_more': '',
            'db': 'colmap.db',
            'images': '.',
            'output': 'colmap_model',
            'worklist': None,
        }

        self.id_string, self.args = set_args('reconstruct', args, self.args)

        self._model_path = ''
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = None

    def get_id(self):
        return self.id_string

    def run(self, **pipe_data):
        return {}

    def _snapshot(self):
        folder = tempfile.mkdtemp()
        path = os.path.join(folder, 'snapshot.db')

        source = sqlite3.connect(self.args['db'])
        target = sqlite3.connect(path)
        source.backup(target)
        target.close()
        source.close()

        return path

    def _reconstruct(self, database_path):
        try:
            self._map(database_path)
        finally:
            shutil.rmtree(os.path.dirname(database_path), ignore_errors=True)

    def _map(self, database_path):
        with tempfile.TemporaryDirectory() as tmp:
            models = pycolmap.incremental_mapping(
                database_path=database_path,
                image_path=self.args['images'],
                output_path=tmp,
                input_path=self._model_path,
            )

            if not models:
                print("reconstruct_module: no model reconstructed this round")
                return

            best = max(models.values(), key=lambda m: m.num_reg_images())

            out = self.args['output']
            shutil.rmtree(out, ignore_errors=True)
            os.makedirs(out, exist_ok=True)
            best.write(out)
            self._model_path = out

            print(f"reconstruct_module: {best.num_reg_images()} images registered, "
                  f"{best.num_points3D()} points, {len(models)} model(s) found")

    def finalize(self):
        if self._future is not None:
            self._future.result()

        self._future = self._executor.submit(self._reconstruct, self._snapshot())

        worklist = self.args.get('worklist')
        if worklist is None or not worklist.args.get('continue', False):
            self._future.result()
            self._future = None
            self._executor.shutdown(wait=True)
