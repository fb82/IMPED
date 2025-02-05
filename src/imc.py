import base_tools as imped
import torch
import pycolmap
import numpy as np
import os


def compute_3D(db, img_dir, output_path):

    os.system('colmap exhaustive_matcher --database_path ' + db)
    # pycolmap.match_exhaustive(db)
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


def make_gt(img_file='../aux/imgs', rec_file='../aux/3D'):
    data = []

    datasets = os.listdir(img_file)

    for dataset in datasets:
        abs_dataset = os.path.join(img_file, dataset)
        
        if os.path.isdir(abs_dataset):
            tmp_dataset = {}
            tmp_dataset['dataset_name'] = dataset
            tmp_dataset['dataset_scenes'] = []
            tmp_dataset['dataset_models'] = []
            tmp_dataset['dataset_outliers'] = False
            
            scenes = os.listdir(abs_dataset)

            for scene in scenes:
                abs_scene = os.path.join(img_file, dataset, scene)
                
                if os.path.isdir(abs_scene):
                    if scene == 'outliers':
                        tmp_dataset['dataset_outliers'] = True
                    else:  
                        abs_3d = os.path.join(rec_file, dataset, scene)

                        db = os.path.join(abs_3d, 'database.db')
                        models = os.path.join(abs_3d, 'models')                            

                        if not os.path.isfile(db):
                            os.makedirs(abs_3d, exist_ok=True)  

                            with torch.inference_mode():    
                                pipeline = [
                                    imped.loftr_module(),
                                    imped.magsac_module(),
                                    imped.to_colmap_module(db=db),
                                ]        
                                        
                                imped.run_pairs(pipeline, abs_scene, db_name=os.path.splitext(db)[0] + '.hdf5')
                                del pipeline[-1]            

                        if not os.path.isdir(models):
                            os.makedirs(models, exist_ok=True)  
                            
                            compute_3D(db, abs_3d, models)
                            
                        best_n = 0
                        best_model = None
                        for model in os.listdir(models):
                            abs_model = os.path.join(models, model)
                            
                            n = pycolmap.Reconstruction(abs_model).num_images()
                            if n > best_n:
                                best_n = n
                                best_model = abs_model
                                    
                        tmp_dataset['dataset_scenes'].append(scene)
                        tmp_dataset['dataset_models'].append(best_model)
                            
            data.append(tmp_dataset)

    return data                
    

if __name__ == '__main__':    

    dataset = make_gt()    

# ### GT    

#     submission = '../tmp/GT.csv'
#     os.makedirs('../tmp/db', exist_ok=True)    
#     os.makedirs('../tmp/3D', exist_ok=True)    

#     dataset = [
#         {
#             'dataset_name' : 'ETs',
#             'dataset_scenes':
#             [
#                 {'scene': 'ET', 'db': '../tmp/db/ET.db', 'imgs': '../data/ET', 'models': '../tmp/3D/ET'},
#                 {'scene': 'another_ET', 'db': '../tmp/db/another_ET.db', 'imgs': '../data/another_ET', 'models': '../tmp/3D/another_ET'},
#             ],
#         },        
#         {
#             'dataset_name' : 'kermits',
#             'dataset_scenes':
#             [
#                 {'scene': 'kermit', 'db': '../tmp/db/kermit.db', 'imgs': '../data/kermit', 'models': '../tmp/3D/kermit_3D'},
#                 {'scene': 'another_kermit', 'db': '../tmp/db/another_kermit.db', 'imgs': '../data/another_kermit', 'models': '../tmp/3D/another_kermit'},
#             ],
#         },
#     ]
                
#     mode = 'w'    
#     for data in dataset:    
#         dataset_name = data['dataset_name']
#         for subscene in data['dataset_scenes']:
#             scene = subscene['scene']
#             db = subscene['db']
#             imgs = subscene['imgs']
#             models = subscene['models']
            
#             with torch.inference_mode():    
#                 pipeline = [
#                     imped.loftr_module(),
#                     imped.magsac_module(),
#                     imped.to_colmap_module(db=db),
#                 ]        
                        
#                 imped.run_pairs(pipeline, imgs, db_name=os.path.splitext(db)[0] + '.hdf5')
#                 del pipeline[-1]            
                                
#             maps = compute_3D(db, imgs, models)
#             best_map = maps[np.argmax([maps[i].num_reg_images() for i in range(len(maps))])]
#             create_submission(dataset_name, scene, best_map, mode=mode, csv_name=submission)
    
#             mode = 'a'            
            
        
# ### test data

#     submission = '../tmp/submission.csv'

#     dataset = [
#         {'dataset_name' : 'ETs', 'db': '../tmp/db/ETs.db', 'imgs': '../data/ETs', 'models': '../tmp/3D/ETs'},       
#         {'dataset_name' : 'kermits', 'db': '../tmp/db/kermits.db', 'imgs': '../data/kermits', 'models': '../tmp/3D/kermits_3D'},
#     ]

#     mode = 'w'    
#     for data in dataset:  
#         dataset_name = data['dataset_name']

#         db = data['db']
#         imgs = data['imgs']
#         models = data['models']
        
#         with torch.inference_mode():    
#             pipeline = [
#                 imped.loftr_module(),
#                 imped.magsac_module(),
#                 imped.to_colmap_module(db=db),
#             ]

#             imped.run_pairs(pipeline, imgs, db_name=os.path.splitext(db)[0] + '.hdf5')
#             del pipeline[-1]            
            
#         maps = compute_3D(db, imgs, models)        
        
#         for i in range(len(maps)):    
            
#             scene = 'unknown_' + str(i)
#             unknown_map = maps[i]
            
#             create_submission(dataset_name, scene, unknown_map, mode=mode, csv_name=submission)
    
#             mode = 'a'
