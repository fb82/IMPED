#!/usr/bin/env python3
import argparse
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def legacy_like_mode():
    """Temporarily patch coldb_ext to approximate old slow merge behavior."""
    from colmap_fun.colmap_ext import coldb_ext
    import colmap_db.database as coldb

    orig_get_match_image_pairs = coldb_ext.get_match_image_pairs
    orig_update_keypoints = coldb_ext.update_keypoints
    orig_update_matches = coldb_ext.update_matches
    orig_update_two_view_geometry = coldb_ext.update_two_view_geometry

    def exhaustive_pairs(self, include_two_view_geometry=True):
        imgs = self.get_images()
        out = []
        for i, (id0, _) in enumerate(imgs):
            for j, (id1, _) in enumerate(imgs):
                if j <= i:
                    continue
                out.append((int(id0), int(id1)))
        return out

    def legacy_update_keypoints(self, image_id, keypoints):
        keypoints = np.asarray(keypoints, np.float32)
        if self.get_keypoints(image_id) is None:
            if keypoints.shape[0] > 0:
                self.add_keypoints(image_id, keypoints)
        else:
            if keypoints.shape[0] > 0:
                self.execute(
                    "UPDATE keypoints SET rows=?, cols=?, data=? WHERE image_id=?",
                    keypoints.shape + (coldb.array_to_blob(keypoints),) + (image_id,),
                )
            else:
                self.execute("DELETE FROM keypoints WHERE image_id=?", (image_id,))

    def legacy_update_matches(self, image_id1, image_id2, matches):
        pair_id = coldb.image_ids_to_pair_id(image_id1, image_id2)
        if self.get_matches(image_id1, image_id2) is None:
            if matches.shape[0] > 0:
                self.add_matches(image_id1, image_id2, matches)
        else:
            matches = np.asarray(matches, np.uint32)
            if image_id1 > image_id2:
                matches = matches[:, ::-1]
            if matches.shape[0] > 0:
                self.execute(
                    "UPDATE matches SET rows=?, cols=?, data=? WHERE pair_id=?",
                    matches.shape + (coldb.array_to_blob(matches),) + (pair_id,),
                )
            else:
                self.execute("DELETE FROM matches WHERE pair_id=?", (pair_id,))

    def legacy_update_two_view_geometry(self, image_id1, image_id2, matches, model=None):
        if model is None:
            model = {}
        pair_id = coldb.image_ids_to_pair_id(image_id1, image_id2)

        if self.get_two_view_geometry(image_id1, image_id2)[0] is None:
            if matches.shape[0] > 0:
                self.add_two_view_geometry(image_id1, image_id2, matches, config=8, **model)
        else:
            matches = np.asarray(matches, np.uint32)
            if image_id1 > image_id2:
                matches = matches[:, ::-1]
            if matches.shape[0] > 0:
                self.execute(
                    "UPDATE two_view_geometries SET rows=?, cols=?, data=?, config=? WHERE pair_id=?",
                    matches.shape + (coldb.array_to_blob(matches),) + (8, pair_id),
                )
            else:
                self.execute("DELETE FROM two_view_geometries WHERE pair_id=?", (pair_id,))

    coldb_ext.get_match_image_pairs = exhaustive_pairs
    coldb_ext.update_keypoints = legacy_update_keypoints
    coldb_ext.update_matches = legacy_update_matches
    coldb_ext.update_two_view_geometry = legacy_update_two_view_geometry

    try:
        yield
    finally:
        coldb_ext.get_match_image_pairs = orig_get_match_image_pairs
        coldb_ext.update_keypoints = orig_update_keypoints
        coldb_ext.update_matches = orig_update_matches
        coldb_ext.update_two_view_geometry = orig_update_two_view_geometry


def run_once(db_names, output_db, mode):
    from colmap_fun import merge_colmap_db
    if os.path.exists(output_db):
        os.remove(output_db)

    t0 = time.perf_counter()
    if mode == "legacy":
        with legacy_like_mode():
            merge_colmap_db(db_names, output_db, profile=False)
    else:
        merge_colmap_db(db_names, output_db, profile=False)
    return time.perf_counter() - t0


def profile_once(db_names, output_db):
    from colmap_fun import merge_colmap_db
    if os.path.exists(output_db):
        os.remove(output_db)
    return merge_colmap_db(db_names, output_db, profile=True, return_profile=True)


def main():
    parser = argparse.ArgumentParser(description="Benchmark old-vs-new COLMAP merge behavior")
    parser.add_argument("--db", nargs="+", required=True, help="Input COLMAP DB files")
    parser.add_argument("--out-dir", default="test_merge/out", help="Output directory")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per mode")
    parser.add_argument("--profile", action="store_true", help="Run one profiled optimized merge and print timing breakdown")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_names = [str(Path(d).resolve()) for d in args.db]
    print("Input DBs:")
    for d in db_names:
        print(f"  - {d}")

    results = {"legacy": [], "optimized": []}

    for mode in ["legacy", "optimized"]:
        print(f"\nRunning mode={mode}")
        for i in range(args.repeats):
            output_db = out_dir / f"merged_{mode}_{i}.db"
            dt = run_once(db_names, str(output_db), mode)
            results[mode].append(dt)
            print(f"  run {i + 1}/{args.repeats}: {dt:.3f}s")

    legacy_avg = float(np.mean(results["legacy"]))
    opt_avg = float(np.mean(results["optimized"]))
    speedup = legacy_avg / opt_avg if opt_avg > 0 else float("inf")

    print("\nSummary")
    print(f"  legacy avg:    {legacy_avg:.3f}s")
    print(f"  optimized avg: {opt_avg:.3f}s")
    print(f"  speedup:       {speedup:.2f}x")

    if args.profile:
        print("\nRunning profiled optimized merge")
        prof_db = out_dir / "merged_profiled.db"
        prof = profile_once(db_names, str(prof_db))
        print("Profile dictionary keys:", ", ".join(sorted(prof.keys())))


if __name__ == "__main__":
    main()
