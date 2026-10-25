import os
import shutil
import tempfile

import numpy as np
import pycolmap

from core import set_args
from .louvain_module import db_graph, louvain_module
from .reconstruct_module import largest_model, snapshot_db


def median_cameras(rec, images):
    params = {}
    for image_id in images:
        camera = rec.cameras[rec.images[image_id].camera_id]
        params.setdefault((camera.model, camera.width, camera.height), []).append(camera.params)
    return {key: np.median(p, axis=0) for key, p in params.items()}


def repair_reconstruction(rec, clusters, db, images, output):
    drop = set().union(*clusters)
    keep = {image_id for image_id in rec.reg_image_ids() if image_id not in drop}
    medians = median_cameras(rec, keep)

    database = pycolmap.Database.open(db)
    clean = pycolmap.Reconstruction()
    for image_id in keep:
        image = rec.images[image_id]
        if not clean.exists_camera(image.camera_id):
            clean.add_camera_with_trivial_rig(rec.cameras[image.camera_id])
        keypoints = database.read_keypoints(image_id)[:, :2]
        clean.add_image_with_trivial_frame(
            pycolmap.Image(name=image.name, keypoints=keypoints, camera_id=image.camera_id, image_id=image_id),
            image.cam_from_world(),
        )
    for point in rec.points3D.values():
        track = pycolmap.Track([e for e in point.track.elements if e.image_id not in drop])
        if track.length() >= 2:
            clean.add_point3D(point.xyz, track, point.color)

    names = []
    for image in database.read_all_images():
        if image.image_id in keep or image.image_id in drop:
            names.append(image.name)
        if image.image_id in keep:
            continue
        camera = database.read_camera(image.camera_id)
        key = (camera.model, camera.width, camera.height)
        if key in medians:
            camera.params = medians[key]
            database.update_camera(camera)
    database.close()

    print(f"repair_reconstruction: {rec.num_reg_images() - len(keep)} images removed, {len(keep)} kept, {clean.num_points3D()} points")

    options = pycolmap.IncrementalPipelineOptions()
    options.image_names = names
    options.ba_global_max_refinements = 1
    options.multiple_models = False
    options.fix_existing_frames = True
    options.structure_less_registration_fallback = False
    options.ba_refine_focal_length = False
    options.ba_refine_extra_params = False
    options.mapper.abs_pose_refine_focal_length = False
    options.mapper.abs_pose_refine_extra_params = False

    with tempfile.TemporaryDirectory() as tmp:
        clean_path = os.path.join(tmp, 'clean')
        mapped_path = os.path.join(tmp, 'mapped')
        os.makedirs(clean_path)
        os.makedirs(mapped_path)
        clean.write(clean_path)
        models = pycolmap.incremental_mapping(
            database_path=db,
            image_path=images,
            output_path=mapped_path,
            input_path=clean_path,
            options=options,
        )

    repaired = max(models.values(), key=lambda m: m.num_reg_images())
    repaired.transform(pycolmap.align_reconstructions_via_proj_centers(repaired, clean, max_proj_center_error=0.01))
    os.makedirs(output, exist_ok=True)
    repaired.write(output)
    back = len([i for i in drop if repaired.exists_image(i) and repaired.images[i].has_pose])
    print(f"repair_reconstruction: {repaired.num_reg_images()} images registered, {back} of {len(drop)} checked images in the model")
    return repaired


class repair_reconstruction_module:
    """
    Builds the repaired model from the old one: removes the cameras of
    `clusters` from the largest model of `model`, registers them again with
    the matches now in `db`, the rest of the model kept fixed, and writes
    the result to `output` (default: the model folder with a `_fixed`
    suffix), also kept in `self.repaired`.

    `model` and `clusters` are the ones louvain_module found (its
    output['model'] and output['clusters']); `clusters` can also be the
    `out_path` louvain_module saved them to. Like reconstruct_module it
    works in finalize(), so it goes at the end of the pipeline matching the
    pairs of louvain_module, after the to_colmap_module writing into `db`,
    or is called directly; with `worklist` it waits until that list is
    empty.
    """
    def __init__(self, **args):
        self.single_image = False
        self.pipeliner = False
        self.pass_through = True
        self.add_to_cache = False

        self.args = {
            'id_more': '',
            'db': 'colmap.db',
            'images': '.',
            'model': None,
            'clusters': [],
            'output': None,
            'worklist': None,
        }

        self.id_string, self.args = set_args('repair_reconstruction', args, self.args)
        self.repaired = None

    def get_id(self):
        return self.id_string

    def run(self, **pipe_data):
        return {}

    def finalize(self):
        if self.args['worklist']:
            return

        clusters = self.args['clusters']
        if isinstance(clusters, str):
            clusters = louvain_module.load(clusters)['clusters']
        if not clusters:
            print("repair_reconstruction_module: no cluster to repair")
            return

        model = self.args['model']
        _, rec = largest_model(model)
        if rec is None:
            print("repair_reconstruction_module: no model to repair")
            return

        graph = db_graph(self.args['db'])
        id_of = {graph.nodes[image_id]['name']: image_id for image_id in graph}
        clusters = [{id_of[name] for name in cluster if name in id_of} for cluster in clusters]

        output = self.args['output'] or os.path.normpath(model.args['output'] if hasattr(model, 'args') else model) + '_fixed'
        database_path = snapshot_db(self.args['db'])
        try:
            self.repaired = repair_reconstruction(rec, clusters, database_path, self.args['images'], output)
        finally:
            shutil.rmtree(os.path.dirname(database_path), ignore_errors=True)
