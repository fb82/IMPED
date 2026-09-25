import os
import shutil
import tempfile

import numpy as np
import pycolmap

from core import set_args
from .reconstruct_module import reconstruct_module


class hierarchical_reconstruct_module(reconstruct_module):
    def __init__(self, **args):
        super().__init__(**args)

        self.args = {
            **self.args,
            'overlap_threshold': 0.8,
            'reproj_error_factor': 1.5,
            'top_n': None,
            'align_min_common_observations': 3,
            'align_max_error': 0.005,
            'align_min_inlier_ratio': 0.9,
        }
        self.id_string, self.args = set_args('hierarchical_reconstruct', args, self.args)

        self._pool = []
        self.n_models_built = 0
        self.n_merges = 0

    def _map(self, database_path):
        with tempfile.TemporaryDirectory() as tmp:
            candidates = []

            for i, model_path in enumerate(self._pool):
                models = pycolmap.incremental_mapping(
                    database_path=database_path,
                    image_path=self.args['images'],
                    output_path=os.path.join(tmp, f'cont_{i}'),
                    input_path=model_path,
                )
                if models:
                    candidates.append(max(models.values(), key=lambda m: m.num_reg_images()))

            fresh = pycolmap.incremental_mapping(
                database_path=database_path,
                image_path=self.args['images'],
                output_path=os.path.join(tmp, 'fresh'),
            )
            candidates.extend(fresh.values())
            self.n_models_built += len(candidates)

            if not candidates:
                print("hierarchical_reconstruct_module: no model reconstructed this round")
                return

            candidates = self._merge_all(candidates, database_path, tmp)

            top_n = self.args['top_n']
            if top_n is not None:
                candidates.sort(key=lambda m: m.num_reg_images(), reverse=True)
                candidates = candidates[:top_n]

            out = self.args['output']
            shutil.rmtree(out, ignore_errors=True)
            os.makedirs(out, exist_ok=True)
            self._pool = []
            for i, model in enumerate(candidates):
                path = os.path.join(out, str(i))
                os.makedirs(path, exist_ok=True)
                model.write(path)
                self._pool.append(path)

            total_images = sum(m.num_reg_images() for m in candidates)
            print(f"hierarchical_reconstruct_module: {len(candidates)} model(s), "
                  f"{total_images} images registered in total")

    def _overlap(self, a, b):
        shared = {im.name for im in a.images.values() if im.has_pose} & \
                 {im.name for im in b.images.values() if im.has_pose}
        smaller = min(a.num_reg_images(), b.num_reg_images())
        return len(shared) / smaller if smaller else 0.0

    def _merge_all(self, candidates, database_path, tmp):
        changed = True
        while changed and len(candidates) > 1:
            changed = False
            pairs = sorted(
                ((i, j) for i in range(len(candidates)) for j in range(i + 1, len(candidates))),
                key=lambda ij: self._overlap(candidates[ij[0]], candidates[ij[1]]),
                reverse=True,
            )
            for i, j in pairs:
                merged = self._try_merge(candidates[i], candidates[j], database_path, tmp)
                if merged is not None:
                    candidates = [c for k, c in enumerate(candidates) if k not in (i, j)]
                    candidates.append(merged)
                    changed = True
                    break
        return candidates

    def _try_merge(self, a, b, database_path, tmp):
        if self._overlap(a, b) >= self.args['overlap_threshold']:
            return self._pick_better(a, b)

        try:
            return self._align_and_merge(a, b, database_path, tmp)
        except Exception as e:
            print(f"hierarchical_reconstruct_module: merge attempt failed ({e}), keeping models separate")
            return None

    def _align_and_merge(self, a, b, database_path, tmp):
        shared = {im.name for im in a.images.values() if im.has_pose} & \
                 {im.name for im in b.images.values() if im.has_pose}
        print(f"hierarchical_reconstruct_module: aligning a({a.num_reg_images()} images, {a.num_points3D()} points) "
              f"with b({b.num_reg_images()} images, {b.num_points3D()} points), {len(shared)} shared images")

        sim3d = pycolmap.align_reconstructions_via_points(
            a, b,
            min_common_observations=self.args['align_min_common_observations'],
            max_error=self.args['align_max_error'],
            min_inlier_ratio=self.args['align_min_inlier_ratio'],
        )
        if sim3d is None:
            print("hierarchical_reconstruct_module: align_reconstructions_via_points returned None")
            return None

        a_err = a.compute_mean_reprojection_error()
        b_err = b.compute_mean_reprojection_error()

        a = pycolmap.Reconstruction(a)
        a.transform(sim3d)

        merged = pycolmap.Reconstruction()
        for model in (b, a):
            for camera in model.cameras.values():
                if not merged.exists_camera(camera.camera_id):
                    merged.add_camera_with_trivial_rig(camera)
            for image in model.images.values():
                if image.has_pose and not merged.exists_image(image.image_id):
                    fresh_image = pycolmap.Image(
                        name=image.name,
                        keypoints=np.array([p.xy for p in image.points2D]).reshape(-1, 2),
                        camera_id=image.camera_id,
                        image_id=image.image_id,
                    )
                    merged.add_image_with_trivial_frame(fresh_image, image.cam_from_world())

        merged = pycolmap.triangulate_points(
            merged, database_path, self.args['images'], tempfile.mkdtemp(dir=tmp)
        )
        pycolmap.bundle_adjustment(merged, pycolmap.BundleAdjustmentOptions())

        merged_err = merged.compute_mean_reprojection_error()
        worst = max(a_err, b_err)
        print(f"hierarchical_reconstruct_module: aligned ok, merged has {merged.num_reg_images()} images, "
              f"err {merged_err:.4g} vs worst input err {worst:.4g} (factor {self.args['reproj_error_factor']})")

        if merged.num_reg_images() < max(a.num_reg_images(), b.num_reg_images()):
            print("hierarchical_reconstruct_module: rejected, merged model lost registered images")
            return None

        if worst and merged_err > self.args['reproj_error_factor'] * worst:
            print("hierarchical_reconstruct_module: rejected, reprojection error too high")
            return None

        self.n_merges += 1
        return merged

    def _pick_better(self, a, b):
        if a.num_reg_images() != b.num_reg_images():
            return a if a.num_reg_images() > b.num_reg_images() else b
        return a if a.compute_mean_reprojection_error() <= b.compute_mean_reprojection_error() else b
