
import numpy as np

#from core import device, pipe_color, show_progress, image_pairs, laf2homo, homo2laf, apply_homo, change_patch_homo, decompose_H_other, decompose_H, compressed_pickle, decompress_pickle, qvec2rotmat, vector_norm, quaternion_matrix, affine_matrix_from_points, set_args, enable_quadtree
import colmap_db.database as coldb

SIMPLE_RADIAL = 2

UNDEFINED = 0  # Not provided
DEGENERATE = 1 # Degenerate configuration (e.g., no overlap or not enough inliers).
CALIBRATED = 2 # Essential matrix.
UNCALIBRATED = 3 # Fundamental matrix.
PLANAR = 4 # Homography, planar scene with baseline.
PANORAMIC = 5 # Homography, pure rotation without baseline.
PLANAR_OR_PANORAMIC = 6 # Homography, planar or panoramic.
WATERMARK = 7 # Watermark, pure 2D translation in image borders.
MULTIPLE = 8 # Multi-model configuration, i.e. the inlier matches result from multiple individual, non-degenerate configurations.

class coldb_ext(coldb.COLMAPDatabase):
    """
    An extended interface for interacting with COLMAP SQLite databases.

    This class provides helper methods to abstract away complex SQL operations 
    and binary data conversions. It handles:
    1. Image/Camera Lookups: Resolving filenames to internal database IDs.
    2. Blob Management: Converting NumPy arrays to SQLite BLOBs and back.
    3. Geometric Integrity: Ensuring that when image pairs are flipped 
       (e.g., Image 2 to Image 1), the corresponding matrices (H, E, F) 
       and match indices are correctly transposed or inverted.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def get_image_id(self, image):
        cursor = self.execute(
            "SELECT image_id FROM images WHERE name=?",
            (image, ),
        )
        image_id = cursor.fetchone()
        if image_id is not None: image_id = image_id[0]
        return image_id


    def get_camera(self, camera_id):
        cursor = self.execute("SELECT model, width, height, params, prior_focal_length FROM cameras where camera_id=?", (camera_id, ))
        cam = cursor.fetchone()
        if cam is None:
            return None
        else:
            c, w, h, p, f = cam
            p = coldb.blob_to_array(p, np.float64)
            return c, w, h, p, f


    def get_image(self, image_id):
        cursor = self.execute("SELECT name, camera_id FROM images where image_id=?", (image_id, ))
        img = cursor.fetchone()
        if img is None:
            return None
        else:
            return img


    def get_keypoints(self, image_id):
        cursor = self.execute("SELECT data, rows, cols FROM keypoints where image_id=?", (image_id, ))
        kpts = cursor.fetchone()
        if kpts is None:
            return None
        else:
            k, r, c = kpts
            return np.reshape(coldb.blob_to_array(kpts[0], np.float32), (r, c))


    def update_keypoints(self, image_id, keypoints):
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]

        keypoints = np.asarray(keypoints, np.float32)
        
        if keypoints.shape[0] > 0:
            self.execute(
                "INSERT OR REPLACE INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (image_id, keypoints.shape[0], keypoints.shape[1], coldb.array_to_blob(keypoints)),
            )
        else:
            self.execute(
                "DELETE FROM keypoints WHERE image_id=?",
                (image_id, ),
                )

    def get_matches(self, image_id1, image_id2):
        pair_id = coldb.image_ids_to_pair_id(image_id1, image_id2)
        cursor = self.execute("SELECT data, rows, cols FROM matches where pair_id=?", (pair_id, ))
        m = cursor.fetchone()
        if m is None:
            return None
        else:
            m, r, c = m
            
            if r == 0: return None
            
            m = np.reshape(coldb.blob_to_array(m, np.uint32), (r, c))

            if image_id1 > image_id2: m = m[:, ::-1]

            return m


    def get_two_view_geometry(self, image_id1, image_id2):
        pair_id = coldb.image_ids_to_pair_id(image_id1, image_id2)
        cursor = self.execute("SELECT data, rows, cols, config, E, F, H FROM two_view_geometries where pair_id=?", (pair_id, ))
        m = cursor.fetchone()
        if m is None:
            return None, None
        else:
            model = {}
            m, r, c, config, E, F, H = m
            
            if r == 0: return None, None
            
            m = np.reshape(coldb.blob_to_array(m, np.uint32), (r, c))

            if (config == PLANAR) or (config == PANORAMIC) or (config == PLANAR_OR_PANORAMIC):
                model['H'] = np.reshape(coldb.blob_to_array(H, np.float64), (3, 3))

            if (config == CALIBRATED):
                model['E'] = np.reshape(coldb.blob_to_array(E, np.float64), (3, 3))

            if (config == UNCALIBRATED):
                model['F'] = np.reshape(coldb.blob_to_array(F, np.float64), (3, 3))

            if image_id1 > image_id2:
                m = m[:, ::-1]
                if 'H' in model: model['H'] = np.linalg.inv(model['H']) 
                if 'F' in model: model['F'] = np.transpose(model['F']) 
                if 'E' in model: model['E'] = np.transpose(model['E']) 

            return m, model


    def update_matches(self, image_id1, image_id2, matches):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        pair_id = coldb.image_ids_to_pair_id(image_id1, image_id2)

        matches = np.asarray(matches, np.uint32)

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        if matches.shape[0] > 0:
            self.execute(
                "INSERT OR REPLACE INTO matches(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (pair_id, matches.shape[0], matches.shape[1], coldb.array_to_blob(matches)),
            )
        else:
            self.execute(
                "DELETE FROM matches WHERE pair_id=?",
                (pair_id, ),
                )


    def get_images(self):
        cursor = self.execute("SELECT image_id, name FROM images")
        m = cursor.fetchall()        
        
        return m


    def get_match_image_pairs(self, include_two_view_geometry=True):
        cursor = self.execute("SELECT pair_id FROM matches")
        pair_ids = {row[0] for row in cursor.fetchall()}
        if include_two_view_geometry:
            cursor = self.execute("SELECT pair_id FROM two_view_geometries")
            pair_ids.update(row[0] for row in cursor.fetchall())

        out = []
        for pair_id in pair_ids:
            image_id1, image_id2 = coldb.pair_id_to_image_ids(pair_id)
            out.append((int(image_id1), int(image_id2)))
        return out
    

    def update_two_view_geometry(self, image_id1, image_id2, matches, model=None):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if model is None: model = {}

        pair_id = coldb.image_ids_to_pair_id(image_id1, image_id2)

        matches = np.asarray(matches, np.uint32)

        if image_id1 > image_id2:
            matches = matches[:, ::-1]
            if 'H' in model: model['H'] = np.linalg.inv(model['H'])
            if 'F' in model: model['F'] = np.transpose(model['F'])
            if 'E' in model: model['E'] = np.transpose(model['E'])

        how_many_models = 0
        F_blob = None
        E_blob = None
        H_blob = None
        if 'H' in model:
            config = PLANAR_OR_PANORAMIC
            how_many_models += 1
            H_blob = coldb.array_to_blob(model['H'])
        if 'F' in model:
            config = UNCALIBRATED
            how_many_models += 1
            F_blob = coldb.array_to_blob(model['F'])
        if 'E' in model:
            config = CALIBRATED
            how_many_models += 1
            E_blob = coldb.array_to_blob(model['E'])
        if how_many_models != 1:
            config = MULTIPLE

        if matches.shape[0] > 0:
            self.execute(
                "INSERT OR REPLACE INTO two_view_geometries(pair_id, rows, cols, data, config, F, E, H, qvec, tvec) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pair_id,
                    matches.shape[0],
                    matches.shape[1],
                    coldb.array_to_blob(matches),
                    config,
                    F_blob,
                    E_blob,
                    H_blob,
                    None,
                    None,
                ),
            )
        else:
            self.execute(
                "DELETE FROM two_view_geometries WHERE pair_id=?",
                (pair_id, ),
                )
