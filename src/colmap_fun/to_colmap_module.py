import os
import numpy as np
import torch
from PIL import Image

import pickled_hdf5.pickled_hdf5 as pickled_hdf5
from core import device as global_device
from ensemble import pipe_union

from .colmap_ext import SIMPLE_RADIAL, coldb_ext


class to_colmap_module:
    """
    A data-export module that saves pipeline results into a COLMAP database.

    This module handles the heavy lifting of:
    1. Registering new cameras and images in the database.
    2. Converting PyTorch keypoints to COLMAP's binary blob format.
    3. Merging new matches with existing ones using various 'sampling_modes'.
    4. Injecting verified geometric models (Homography, Essential, Fundamental).

    Attributes:
        db (str): Path to the target COLMAP '.db' file.
        focal_cf (float): A multiplier for focal length if camera intrinsics 
            are unknown (defaults to 1.2 * max(width, height)).
        sampling_mode (str): Determines how to handle overlapping keypoints 
            (e.g., 'raw', 'avg_all_matches').
        include_two_view_geometry (bool): If True, saves the 'Verified' 
            matches into the two_view_geometry table.
    """
  
    def __init__(self, **args):
        from core import set_args

        self.single_image = False
        self.pipeliner = False
        self.pass_through = True
        self.add_to_cache = True


        self.args = {
            'id_more': '',
            'db': 'colmap.db',
            'aux_hdf5': 'colmap_aux.hdf5',
            'focal_cf': 1.2,
            'only_keypoints': False,
            'unique': True,
            'only_matched': False,
            'no_unmatched': True,
            'include_two_view_geometry': True,
            'sampling_mode': 'raw',
            'overlapping_cells': False,
            'sampling_scale': 1,
            'sampling_offset': 0,
            'commit_every_pairs': 10,
        }
        self.device = torch.device(self.args.get('device', str(global_device)))


        if 'add_to_cache' in args:
            self.add_to_cache = args['add_to_cache']

        self.id_string, self.args = set_args('to_colmap', args, self.args)

        # ---------------- DB ----------------
        self.db = coldb_ext(self.args['db'])
        self.db.create_tables()
        self._pending_writes = 0

        # ---------------- IMAGE CACHE ----------------
        self._image_id_cache = {}
        self._image_size_cache = {}
        self._image_set = set()
        self._load_image_cache()

        # ---------------- PAIR CACHE (NEW) ----------------
        self._pair_cache = set()
        self._load_pair_cache()

        # ---------------- DB BATCH BUFFER (NEW) ----------------
        self._db_buffer = {
            "keypoints": [],
            "matches": [],
            "two_view": []
        }

        # ---------------- AUX ----------------
        self.aux_hdf5 = None
        if self.args['sampling_mode'] in [
            'avg_all_matches',
            'avg_inlier_matches'
        ]:
            self.aux_hdf5 = pickled_hdf5.pickled_hdf5(
                self.args['aux_hdf5'],
                mode='a',
                label_prefix='pickled/' + self.id_string
            )

    # =========================================================
    # CACHE LOADING
    # =========================================================

    def _load_image_cache(self):
        images = self.db.get_images()  # (image_id, name)
        self._image_id_cache = {name: image_id for image_id, name in images}
        self._image_set = set(self._image_id_cache.keys())

    def _load_pair_cache(self):
        """
        Attempts to load existing pairs from DB.
        Falls back to empty set if DB does not expose them.
        """
        self._pair_cache = set()

        try:
            if hasattr(self.db, "get_existing_pairs"):
                pairs = self.db.get_existing_pairs()
                self._pair_cache = set(tuple(sorted(p)) for p in pairs)
        except Exception:
            self._pair_cache = set()

    # =========================================================
    # FINALIZATION
    # =========================================================

    def finalize(self):
        self._flush_db()
        self.db.commit()
        self.db.close()

        if self.aux_hdf5 is not None:
            self.aux_hdf5.close()
            if os.path.isfile(self.args['aux_hdf5']):
                os.remove(self.args['aux_hdf5'])

    # =========================================================
    # DB FLUSH (NEW)
    # =========================================================

    def _flush_db(self):
        for image_id, pts in self._db_buffer["keypoints"]:
            self.db.update_keypoints(image_id, pts)

        for i1, i2, m_idx in self._db_buffer["matches"]:
            self.db.update_matches(i1, i2, m_idx)

        for i1, i2, m_idx, models in self._db_buffer["two_view"]:
            self.db.update_two_view_geometry(i1, i2, m_idx, model=models)

        self.db.commit()

        self._db_buffer = {
            "keypoints": [],
            "matches": [],
            "two_view": []
        }

    # =========================================================
    # MAIN PIPELINE
    # =========================================================

    def run(self, **args):
        im_ids = []
        imgs = []

        # ---------------- IMAGE REGISTRATION ----------------
        for idx in [0, 1]:
            im = args['img'][idx]
            _, img = os.path.split(im)

            im_id = self._image_id_cache.get(img)

            if im_id is None:
                w, h = self._get_image_size(im)
                cam_id = self.db.add_camera(
                    SIMPLE_RADIAL,
                    w,
                    h,
                    np.array([
                        self.args['focal_cf'] * max(w, h),
                        w / 2,
                        h / 2,
                        0
                    ])
                )
                im_id = self.db.add_image(img, cam_id)

                self._image_id_cache[img] = im_id
                self._image_set.add(img)

            imgs.append(img)
            im_ids.append(im_id)

        # ---------------- PAIR SKIP (NEW) ----------------
        pair = tuple(sorted((imgs[0], imgs[1])))
        if pair in self._pair_cache:
            return {}

        # ====================================================
        # PIPELINE PREPARATION (UNCHANGED LOGIC)
        # ====================================================

        pipe_old = {}

        kp_old0 = self.db.get_keypoints(im_ids[0])
        if kp_old0 is None:
            w_old0 = torch.zeros((0, 6), device=self.device)
            kp_old0 = torch.zeros((0, 2), device=self.device)
        else:
            w_old0 = torch.tensor(kp_old0, device=self.device)
            kp_old0 = torch.tensor(kp_old0[:, :2], device=self.device)

        kp_old1 = self.db.get_keypoints(im_ids[1])
        if kp_old1 is None:
            w_old1 = torch.zeros((0, 6), device=self.device)
            kp_old1 = torch.zeros((0, 2), device=self.device)
        else:
            w_old1 = torch.tensor(kp_old1, device=self.device)
            kp_old1 = torch.tensor(kp_old1[:, :2], device=self.device)


        kH_old0 = torch.zeros((kp_old0.shape[0], 3, 3), device=self.device)
        kr_old0 = torch.full((kp_old0.shape[0], ), torch.inf, device=self.device)

        kH_old1 = torch.zeros((kp_old1.shape[0], 3, 3), device=self.device)
        kr_old1 = torch.full((kp_old1.shape[0], ), torch.inf, device=self.device)


        pipe_old['kp'] = [kp_old0, kp_old1]
        pipe_old['kH'] = [kH_old0, kH_old1]
        pipe_old['kr'] = [kr_old0, kr_old1]
        pipe_old['w'] = [w_old0, w_old1]

        # ---------------- NEW FEATURES ----------------
        w0 = kpts_as_colmap(0, **args)
        w1 = kpts_as_colmap(1, **args)
        args['w'] = [w0, w1]

        counter = (
            self.args['sampling_mode'] in [
                'avg_all_matches',
                'avg_inlier_matches'
            ]
        )

        pipe_out = pipe_union(
            [pipe_old, args],
            unique=self.args['unique'],
            no_unmatched=self.args['no_unmatched'],
            only_matched=self.args['only_matched'],
            sampling_mode=self.args['sampling_mode'],
            sampling_scale=self.args['sampling_scale'],
            sampling_offset=self.args['sampling_offset'],
            overlapping_cells=self.args['overlapping_cells'],
            preserve_order=True,
            counter=counter
        )

        pts0 = pipe_out['w'][0].to('cpu').numpy()
        pts1 = pipe_out['w'][1].to('cpu').numpy()

        # ---------------- AUX COUNTERS ----------------
        if counter and self.aux_hdf5 is not None:
            self.aux_hdf5.add(imgs[0], pipe_out['k_counter'][0])
            self.aux_hdf5.add(imgs[1], pipe_out['k_counter'][1])

        # ====================================================
        # DB WRITE (BUFFERED)
        # ====================================================

        self._db_buffer["keypoints"].append((im_ids[0], pts0))
        self._db_buffer["keypoints"].append((im_ids[1], pts1))

        if not self.args['only_keypoints']:
            m_idx = pipe_out['m_idx'].to('cpu').numpy()
            self._db_buffer["matches"].append((im_ids[0], im_ids[1], m_idx))

            if self.args['include_two_view_geometry']:
                models = {}
                for m in ['H', 'E', 'F']:
                    if m in args and args[m] is not None:
                        models[m] = args[m].to('cpu').numpy()

                self._db_buffer["two_view"].append(
                    (im_ids[0], im_ids[1], m_idx, models)
                )

        # ---------------- CACHE PAIR ----------------
        self._pair_cache.add(pair)

        # ---------------- FLUSH CONTROL ----------------
        self._pending_writes += 1
        commit_every = max(1, int(self.args['commit_every_pairs']))

        if self._pending_writes >= commit_every:
            self._flush_db()
            self._pending_writes = 0

        return {}

    # =========================================================
    # UTIL
    # =========================================================

    def _get_image_size(self, path):
        if path not in self._image_size_cache:
            self._image_size_cache[path] = Image.open(path).size
        return self._image_size_cache[path]

    def get_id(self):
        return self.id_string





def kpts_as_colmap(idx, **args): 
    """
    Decomposes a local homography (kH) into COLMAP's affine components.

    COLMAP expects keypoints in the format: (x, y, a11, a12, a21, a22). 
    This function extracts the (x, y) coordinates and solves for the 
    2x2 affine matrix by removing the translation component from the 
    full 3x3 local homography matrix.

    Args:
        idx (int): The image index (0 or 1) in the current processing pair.
        **args: The pipeline data dictionary containing 'kp' and 'kH'.

    Returns:
        torch.Tensor: A tensor of shape (N, 6) formatted for COLMAP storage.
    """
    kp = args['kp'][idx]
    kH = args['kH'][idx]
     
    t = torch.zeros((kp.shape[0], 3, 3), device=global_device)        
    t[:, [0, 1], 2] = -kH[:, [0, 1], 2]
    t[:, 0, 0] = 1
    t[:, 1, 1] = 1
    t[:, 2, 2] = 1           
     
    h = t.bmm(kH.inverse())
     
    v = torch.zeros((kp.shape[0], 3, 3), device=global_device)        
    v[:, 2, :] = h[:, 2, :]
    v[:, 0, 0] = 1
    v[:, 1, 1] = 1
     
    w = h.bmm(v.inverse())
    w = w[:, :2, :2].reshape(-1, 4)
         
    return torch.cat((kp[:, :2], w), dim=1)
