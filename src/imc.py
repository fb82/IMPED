import base_tools as imped
import torch
import pycolmap
import numpy as np
import os


def compute_3D(db, img_dir, output_path):    
    pycolmap.match_exhaustive(db)
    database_path = db
    return pycolmap.incremental_mapping(database_path=database_path, image_path=img_dir, output_path=output_path)
    
    
def arr_to_str(a):
    return ';'.join([str(x) for x in a.reshape(-1)])


def create_submission(dataset, scene, map_3d, mode='w', csv_name='submission.csv'):
    with open(csv_name, mode) as f:
        if mode == 'w':
            f.write('scene,image,rotation_matrix,translation_vector\n')

        for i in map_3d.images:
            img = map_3d.image(i).name
            R = map_3d.image(i).cam_from_world.rotation.matrix()
            T = np.array(map_3d.image(i).cam_from_world.translation)
            f.write(f'{dataset},{scene},{img},{arr_to_str(R)},{arr_to_str(T)}\n')


if __name__ == '__main__':    

### GT    

    submission = 'GT.csv'

    dataset_name = 'dataset'    
    dataset = [
        {'scene': 'ET', 'db': 'ET.db', 'imgs': '../data/ET', 'models': 'ET_3D'},
        {'scene': 'another_ET', 'db': 'another_ET.db', 'imgs': '../data/another_ET', 'models': 'another_ET_3D'},
#       {'scene': 'kermit', 'db': 'kermit.db', 'imgs': '../data/kermit', 'models': 'kermit_3D'},
#       {'scene': 'another_kermit', 'db': 'another_kermit.db', 'imgs': '../data/another_kermit', 'models': 'another_kermit_3D'},
        ]

    mode = 'w'    
    for data in dataset:    
        scene = data['scene']
        db = data['db']
        imgs = data['imgs']
        models = data['models']
    
        with torch.inference_mode():    
            pipeline = [
                imped.loftr_module(),
                imped.magsac_module(),
                imped.to_colmap_module(db=db),
            ]        
                    
            imped.run_pairs(pipeline, imgs, db_name=scene + '.hdf5')
            del pipeline[-1]
            
        maps = compute_3D(db, imgs, models)
        best_map = maps[np.argmax([maps[i].num_reg_images() for i in range(len(maps))])]
        create_submission(dataset_name, scene, best_map, mode=mode, csv_name=submission)

        mode = 'a'
        
### test data

    submission = 'submission.csv'
    dataset_name = 'dataset'
    
    db = 'ETs.db'
    imgs = '../data/ETs'
    models = 'ETs_3D'

#   db = 'kermits.db'
#   imgs = '../data/kermits'
#   models = 'kermits_3D'

    with torch.inference_mode():    
        pipeline = [
            imped.loftr_module(),
            imped.magsac_module(),
            imped.to_colmap_module(db=db),
        ]        
                
        imped.run_pairs(pipeline, imgs, db_name='submission.hdf5')
        del pipeline[-1]

    maps = compute_3D(db, imgs, models)
        
    mode = 'w'
    for i in range(len(maps)):    
        
        scene = 'unknown_' + str(i)
        unknown_map = maps[i]
        
        create_submission(dataset_name, scene, unknown_map, mode=mode, csv_name=submission)

        mode = 'a'
