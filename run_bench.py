import os
import numpy as np
import src.base_modules as pipe_base
import src.bench_utils as bench
import pickled_hdf5
import time
import src.fessac_modules as fessac
from PIL import Image
import warnings
import torch
import cv2
import h5py
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def csv(pipe_head, dataset_data, load_from='res.hdf5'):
    key_list = []
    def check_item(key, what):
        if isinstance(what, h5py.Dataset): key_list.append(what.name)

    all_data = pickled_hdf5.pickled_hdf5(load_from, 'r')

    all_data_pt = all_data.get_hdf5()[all_data.label_prefix + '/stats/' + pipe_head]
    all_data_pt.visititems(check_item)

    header = None
    csv_data = []
    
    for i in key_list:
        ii = re.sub('/eval$', '', i)
        pipeline = re.sub('^' + all_data.label_prefix + '/stats/', '', ii)

        data_key_ = '/stats/' + pipeline + '/eval'
        eval_data_, is_found = all_data.get(data_key_)
        
        if header is None:
            eval_keys = sorted(eval_data_.keys())

            header = 'pipeline' + ';' + ';'.join(eval_keys) + '\n'
            csv_data.append(header)

        csv_row = pipeline + ';' + ';'.join([str(eval_data_[j]) for j in eval_keys]) + '\n'
        csv_data.append(csv_row)

    save_to = os.path.splitext(load_from)[0] + '_' + pipe_head + '.csv' 
    with open(save_to, 'w') as f:
        for csv_line in csv_data:
            f.write(csv_line)

    all_data.close()


def stats(pipe, dataset_data, bench_path='bench_data', bench_im='imgs', force=False, save_to='res.hdf5'):
    angular_thresholds = [5, 10, 20]
    all_data = pickled_hdf5.pickled_hdf5(save_to)

    n = len(dataset_data['im1'])
    pipeline = ''
    for j, pipe_module in enumerate(pipe):
        pipeline = os.path.join(pipeline, pipe_module.get_id())
        if pipe_module.to_eval:
            data_key_ = '/stats/' + pipeline + '/eval'
            eval_data_, is_found = all_data.get(data_key_)
            
            if (not is_found) or force:
                eval_data_ = {}
                R_err = []
                t_err = []
                inl = []
                prec = []
    
                for i in range(n):
                    pipe_name_base = os.path.join(dataset_data['im1'][i] + '/$/' + dataset_data['im2'][i] + '/$', pipeline)
            
                    data_key = '/' + pipe_name_base + '/eval'
                    eval_data, is_found = all_data.get(data_key)
                        
                    R_err.append(eval_data['R_error'])
                    t_err.append(eval_data['t_error'])
                    inl.append(eval_data['epi_inliers'])
                    prec.append(eval_data['epi_precision'])
    
                aux = np.asarray([R_err, t_err]).T
                max_Rt_err = np.max(aux, axis=1)        
                tmp = np.concatenate((aux, np.expand_dims(max_Rt_err, axis=1)), axis=1)
        
                for a in angular_thresholds:       
                    auc_R = bench.error_auc(np.squeeze(R_err), a)
                    auc_t = bench.error_auc(np.squeeze(t_err), a)
                    auc_max_Rt = bench.error_auc(np.squeeze(max_Rt_err), a)
                    eval_data_['R_AUC@' + '{:02}'.format(a)] = '{:0.3f}'.format(auc_R)
                    eval_data_['R_accuracy@' + '{:02}'.format(a)] = '{:0.3f}'.format(np.sum(tmp[:, 0] < a)/np.shape(tmp)[0])

                    eval_data_['t_AUC@' + '{:02}'.format(a)] = '{:0.3f}'.format(auc_t)
                    eval_data_['t_accuracy@' + '{:02}'.format(a)] = '{:0.3f}'.format(np.sum(tmp[:, 1] < a)/np.shape(tmp)[0])

                    eval_data_['AUC@' + '{:02}'.format(a)] = '{:0.3f}'.format(auc_max_Rt)
                    eval_data_['accuracy@' + '{:02}'.format(a)] = '{:0.3f}'.format(np.sum(tmp[:, 2] < a)/np.shape(tmp)[0])

    
                eval_data_['precision'] = '{:0.3f}'.format(torch.tensor(prec, device=device).mean().item())
                eval_data_['inliers'] = '{:0.3f}'.format(torch.tensor(np.asarray(inl), device=device, dtype=torch.float).mean().item())
                
                all_data.add(data_key_, eval_data_)

    all_data.close()
                

def run_and_eval(pipe, dataset_data, dataset_name, bar_name, bench_path='bench_data' , bench_im='imgs', force=False, ext='.png', use_scale=False, save_to='res.hdf5'):
    all_data = pickled_hdf5.pickled_hdf5(save_to)

    n = len(dataset_data['im1'])
    im_path = os.path.join(bench_im, dataset_name)
    with bench.progress_bar(bar_name + ' - pipeline completion') as p:
        for i in p.track(range(n)):
            pipe_name_base = dataset_data['im1'][i] + '/$/' + dataset_data['im2'][i] + '/$'

            data_key = '/' + pipe_name_base + '/pair'
            pair_data, is_found = all_data.get(data_key)

            # image pair data & GT
            if (not is_found) or force:
                pair_data = pair_info(dataset_data, idx=i, use_scale=use_scale, bench_path=bench_path, im_path=im_path, ext=ext)
                all_data.add(data_key, pair_data)

            pipe_data = pair_data.copy()
            for j, pipe_module in enumerate(pipe):
                pipe_name_base = os.path.join(pipe_name_base, pipe_module.get_id())

                data_key = '/' + pipe_name_base + '/data'
                out_data, is_found = all_data.get(data_key)

                # module computation
                if (not is_found) or force:
                    start_time = time.time()
                    out_data = pipe_module.run(**pipe_data)
                    stop_time = time.time()

                    out_data['running_time'] = stop_time - start_time
                    all_data.add(data_key, out_data)

                for k, v in out_data.items(): pipe_data[k] = v

                if pipe_module.to_eval:
                    data_key = '/' + pipe_name_base + '/eval'
                    eval_data, is_found = all_data.get(data_key)

                    # evaluation
                    if (not is_found) or force:
                        eval_data = eval_fundamental(pipe_data)
                        all_data.add(data_key, eval_data)

    all_data.close()


def pair_info(dataset_data, idx=0, use_scale=False, bench_path='bench_data', im_path='imgs', ext='.png'):
    im1 = os.path.join(bench_path, im_path, os.path.splitext(dataset_data['im1'][idx])[0]) + ext
    im2 = os.path.join(bench_path, im_path, os.path.splitext(dataset_data['im2'][idx])[0]) + ext

    if use_scale:
        scales = dataset_data['im_pair_scale'][idx]
    else:
        scales = np.asarray([[1.0, 1.0], [1.0, 1.0]])

    pair_data = {
        'im1': im1,
        'im2': im2,
        'sz1': np.asarray(Image.open(im1).size),
        'sz2': np.asarray(Image.open(im2).size),
        'K1': dataset_data['K1'][idx],
        'K2': dataset_data['K2'][idx],
        'R_gt': dataset_data['R'][idx],
        't_gt': dataset_data['T'][idx],
        'scales': scales,
    }

    return pair_data


def eval_fundamental(out_data, err_th_list=list(range(1,16))):
    warnings.filterwarnings("ignore", category=UserWarning)

    eval_data = {}
    eval_data['R_error'] = np.inf
    eval_data['t_error'] = np.inf
#   eval_data['epi_max_error'] = None
    eval_data['epi_inliers'] = np.zeros(len(err_th_list), dtype=np.int32)
    eval_data['epi_precision'] = 0

    K1 = out_data['K1']
    K2 = out_data['K2']
    R_gt = out_data['R_gt']
    t_gt = out_data['t_gt']

    pts1 = out_data['pt1']
    pts2 = out_data['pt2']

    if torch.is_tensor(pts1):
        pts1 = pts1.detach().cpu().numpy()
        pts2 = pts2.detach().cpu().numpy()

    scales = out_data['scales']

    pts1 = pts1 * scales[0]
    pts2 = pts2 * scales[1]

    nn = pts1.shape[0]

    if nn < 8:
        Rt_ = None
    else:
        F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]
        if F is None:
            Rt_ = None
        else:
            E = K2.T @ F @ K1
            Rt_ = cv2.decomposeEssentialMat(E)

    if nn > 0:
        F_gt = torch.tensor(K2.T, device=device, dtype=torch.float64).inverse() @ \
               torch.tensor([[0, -t_gt[2], t_gt[1]],
                            [t_gt[2], 0, -t_gt[0]],
                            [-t_gt[1], t_gt[0], 0]], device=device) @ \
               torch.tensor(R_gt, device=device) @ \
               torch.tensor(K1, device=device, dtype=torch.float64).inverse()
        F_gt = F_gt / F_gt.sum()

        pt1_ = torch.vstack((torch.tensor(pts1.T, device=device), torch.ones((1, nn), device=device)))
        pt2_ = torch.vstack((torch.tensor(pts2.T, device=device), torch.ones((1, nn), device=device)))

        l1_ = F_gt @ pt1_
        d1 = pt2_.permute(1,0).unsqueeze(-2).bmm(l1_.permute(1,0).unsqueeze(-1)).squeeze().abs() / (l1_[:2]**2).sum(0).sqrt()

        l2_ = F_gt.T @ pt2_
        d2 = pt1_.permute(1,0).unsqueeze(-2).bmm(l2_.permute(1,0).unsqueeze(-1)).squeeze().abs() / (l2_[:2]**2).sum(0).sqrt()

        epi_max_err = torch.maximum(d1, d2)
        inl_sum = (epi_max_err.unsqueeze(-1) < torch.tensor(err_th_list, device=device).unsqueeze(0)).sum(dim=0).type(torch.int)
        avg_prec = inl_sum.type(torch.double).mean()/nn

        epi_max_err = epi_max_err.detach().cpu().numpy().astype(np.single)
        inl_sum = inl_sum.detach().cpu().numpy()
        avg_prec = avg_prec.item()

#       eval_data['epi_max_error'] = epi_max_err
        eval_data['epi_inliers'] = inl_sum
        eval_data['epi_precision'] = avg_prec

    if Rt_ is None:
        eval_data['R_error'] = np.inf
        eval_data['t_error'] = np.inf
    else:
        R_a, t_a, = Rt_[0], Rt_[2].squeeze()
        t_err_a, R_err_a = bench.relative_pose_error_angular(R_gt, t_gt, R_a, t_a)

        R_b, t_b, = Rt_[1], Rt_[2].squeeze()
        t_err_b, R_err_b = bench.relative_pose_error_angular(R_gt, t_gt, R_b, t_b)

        if max(R_err_a, t_err_a) < max(R_err_b, t_err_b):
            R_err, t_err = R_err_a, t_err_b
        else:
            R_err, t_err = R_err_b, t_err_b

        eval_data['R_error'] = R_err
        eval_data['t_error'] = t_err

    return eval_data

if __name__ == '__main__':
    # available RANSAC: pydegensac, magsac, poselib

    pipe_head = lambda: None
    pipe_head.placeholder = 'head'

    pipes = [
        [
            pipe_head,
            fessac.magsac_module(px_th=1.00, output_previous_pts=True),
            fessac.magsac_scaled_module(px_th=1.00, scale_ref_big=True, scale_mode='mean'),
        ],

        [
            pipe_head,
            fessac.magsac_module(px_th=1.00, output_previous_pts=True),
            fessac.magsac_scaled_module(px_th=1.00, scale_ref_big=False, scale_mode='mean'),
        ],

        [
            pipe_head,
            fessac.magsac_module(px_th=1.00, output_previous_pts=True),
            fessac.magsac_scaled_module(px_th=1.00, scale_ref_big=True, scale_mode='median'),
        ],

        [
            pipe_head,
            fessac.magsac_module(px_th=1.00, output_previous_pts=True),
            fessac.magsac_scaled_module(px_th=1.00, scale_ref_big=False, scale_mode='median'),
        ],

        [
            pipe_head,
            fessac.poselib_module(px_th=1.00, output_previous_pts=True),
            fessac.poselib_scaled_module(px_th=1.00, scale_ref_big=True, scale_mode='mean'),
        ],

        [
            pipe_head,
            fessac.poselib_module(px_th=1.00, output_previous_pts=True),
            fessac.poselib_scaled_module(px_th=1.00, scale_ref_big=False, scale_mode='mean'),
        ],

        [
            pipe_head,
            fessac.poselib_module(px_th=1.00, output_previous_pts=True),
            fessac.poselib_scaled_module(px_th=1.00, scale_ref_big=True, scale_mode='median'),
        ],

        [
            pipe_head,
            fessac.poselib_module(px_th=1.00, output_previous_pts=True),
            fessac.poselib_scaled_module(px_th=1.00, scale_ref_big=False, scale_mode='median'),
        ],
    ]

    pipe_heads = [
        pipe_base.keynetaffnethardnet_module(num_features=8000, upright=True, th=0.99),
      # pipe_base.sift_module(num_features=8000, upright=True, th=0.95, rootsift=True),
        pipe_base.lightglue_module(num_features=8000, upright=True, what='superpoint'),
      # pipe_base.lightglue_module(num_features=8000, upright=True, what='aliked'),
      # pipe_base.lightglue_module(num_features=8000, upright=True, what='disk'),
      # pipe_base.loftr_module(num_features=8000, upright=True),
      # dedode2.dedode2_module(num_features=8000, upright=True),
        ]

    for pipe_module in pipe_heads: pipe_module.placeholder = 'head'
    pipe_save_to = [pipe_head.get_id() for pipe_head in pipe_heads]

###

    bench_path = '../bench_data'
    bench_mode = 'fundamental_matrix'
    save_to = 'res'
    show_matches = False

    benchmark_data = {
            'megadepth': {'name': 'megadepth', 'Name': 'MegaDepth', 'setup': bench.megadepth_bench_setup, 'is_outdoor': True, 'is_not_planar': True, 'ext': '.png', 'use_scale': True, 'also_metric': False, 'index': None},
            'scannet': {'name': 'scannet', 'Name': 'ScanNet', 'setup': bench.scannet_bench_setup, 'is_outdoor': False, 'is_not_planar': True, 'ext': '.png', 'use_scale': False, 'also_metric': False, 'index': None},
        }

    for b in benchmark_data.keys():
        b_data, _ = benchmark_data[b]['setup'](bench_path=bench_path, upright=True)

        # select image pair subset
        b_index = benchmark_data[b]['index']
        if not (b_index is None):
            for bname in b_data.keys():
                if isinstance(b_data[bname], list): b_data[bname] = [b_data[bname][bb] for bb in b_index]
                if isinstance(b_data[bname], np.ndarray): b_data[bname] = b_data[bname][b_index]

        for ip in range(len(pipe_heads)):
            pipe_head = pipe_heads[ip]

            to_save_file =  os.path.join(bench_path, save_to, save_to + '_' + benchmark_data[b]['name'])
            os.makedirs(os.path.join(bench_path, save_to), exist_ok=True)

            for i, pipe in enumerate(pipes):
                for k, pipe_module in enumerate(pipe):
                    setattr(pipe[k], 'to_eval', True)
                    if hasattr(pipe_module, 'placeholder'):
                        if pipe_module.placeholder == 'head':
                             pipe[k] = pipe_head
                             setattr(pipe[k], 'to_eval', False)

                for pipe_module in pipe:
                    if hasattr(pipe_module, 'mode'): setattr(pipe_module, 'mode', bench_mode)
                    if hasattr(pipe_module, 'outdoor'): setattr(pipe_module, 'outdoor', benchmark_data[b]['is_outdoor'])

                run_and_eval(pipe, b_data, benchmark_data[b]['name'], benchmark_data[b]['Name'], bench_path=bench_path, ext=benchmark_data[b]['ext'], use_scale=benchmark_data[b]['use_scale'], save_to=to_save_file + '.hdf5')
                stats(pipe, b_data, bench_path=bench_path, save_to=to_save_file + '.hdf5')

            csv(pipe_head.get_id(), b_data, load_from=to_save_file + '.hdf5')
