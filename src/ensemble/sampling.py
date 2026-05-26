
import numpy as np
import torch

#from core import device, pipe_color, show_progress, go_iter, run_pipeline, run_pairs, finalize_pipeline, laf2homo, homo2laf, apply_homo, change_patch_homo, decompose_H_other, decompose_H, compressed_pickle, decompress_pickle, qvec2rotmat, vector_norm, quaternion_matrix, affine_matrix_from_points, set_args, enable_quadtree





def pipe_union(pipe_block, unique=True, no_unmatched=False, only_matched=False, sampling_mode=None, sampling_scale=1, sampling_offset=0, overlapping_cells=False, preserve_order=False, counter=False, device=None, io_device=None, patch_matters=False):
    """
    Combines and cleans multiple sets of image matching results.

    This function is the core of the pipeline's 'Ensemble' capability. 
    It doesn't just stack lists; it performs coordinate sampling, 
    removes duplicate keypoints, handles descriptor merging, and 
    re-indexes match lists to maintain geometric consistency.

    Key Features:
    - deduplication: Merge identical keypoints from different matchers.
    - sampling: Refine coordinates using 'avg' or 'best' modes.
    - order preservation: Keeps points in a specific rank if needed.
    - geometric re-indexing: Updates 'm_idx' to point to the new, 
      consolidated keypoint indices.
    """
    if device is None: device = torch.device('cpu')
    if io_device is None: io_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not isinstance(pipe_block, list): pipe_block = [pipe_block]

    bring_desc = np.unique([pipe['desc'][0].shape[1] for pipe in pipe_block if 'desc' in pipe]).shape[0] == 1

    kp0 = []
    kH0 = []
    kr0 = []
    dd0 = []

    kp1 = []
    kH1 = []
    kr1 = []
    dd1 = []
    
    use_w = False
    for pipe_data in pipe_block:
        if 'w' in pipe_data:
            use_w = True
            break
    
    if use_w:
        w0 = []
        w1 = []
        
    counter0 = None
    counter1 = None    
        
    if counter:
        counter0 = []
        counter1 = []
        
    if preserve_order:
        rank0 = []
        rank1 = []
    
    m_idx = []
    m_val = []
    m_mask = []
    
    m0_offset = 0
    m1_offset = 0
    
    idx0 = None
    idx1 = None
        
    c_rank0 = 0
    c_rank1 = 0
    for i, pipe_data in enumerate(pipe_block):
        if 'kp' in pipe_data:
        
            kp0.append(pipe_data['kp'][0].to(device))
            kp1.append(pipe_data['kp'][1].to(device))
    
            kH0.append(pipe_data['kH'][0].to(device))
            kH1.append(pipe_data['kH'][1].to(device))

            kr0.append(pipe_data['kr'][0].to(device))
            kr1.append(pipe_data['kr'][1].to(device))
            
            if ('desc' in pipe_data) and bring_desc:
                dd0.append(pipe_data['desc'][0].to(device))
                dd1.append(pipe_data['desc'][1].to(device))            
            
            if use_w:
                w0.append(pipe_data['w'][0].to(device))
                w1.append(pipe_data['w'][1].to(device))

            if preserve_order:
                rank0.append(torch.arange(c_rank0, c_rank0 + pipe_data['kp'][0].shape[0], device=device))
                rank1.append(torch.arange(c_rank1, c_rank1 + pipe_data['kp'][1].shape[0], device=device))
                c_rank0 = c_rank0 + pipe_data['kp'][0].shape[0]
                c_rank1 = c_rank1 + pipe_data['kp'][1].shape[0]
                
                if i==0:
                    q_rank0 = pipe_data['kp'][0].shape[0]
                    q_rank1 = pipe_data['kp'][1].shape[0]

            if counter:
                counter0.append(pipe_data['k_counter'][0].to(device))
                counter1.append(pipe_data['k_counter'][1].to(device))
            
            if 'm_idx' in pipe_data:
                # print(f"m_idx shape: {pipe_data['m_idx'].shape}, m_mask shape: {pipe_data['m_mask'].shape}, m_val shape: {pipe_data['m_val'].shape}")

                # print(f"only_matched={only_matched}")

                if only_matched:
                    to_retain = torch.zeros(pipe_data['m_idx'].shape[0], device=device, dtype=torch.bool)
                    to_retain[:pipe_data['m_mask'].shape[0]] = pipe_data['m_mask'].clone().to(device)
                else:
                    to_retain = torch.full((pipe_data['m_idx'].shape[0], ), 1, device=device, dtype=torch.bool)
                          
                m_idx.append(pipe_data['m_idx'].to(device)[to_retain] + torch.tensor([m0_offset, m1_offset], device=device).unsqueeze(0))
                m_val.append(pipe_data['m_val'].to(device)[to_retain])
                m_mask.append(pipe_data['m_mask'].to(device)[to_retain[:pipe_data['m_mask'].shape[0]]])
                    
            m0_offset = m0_offset + pipe_data['kp'][0].shape[0]
            m1_offset = m1_offset + pipe_data['kp'][1].shape[0]

    if 'kp' in pipe_data:
        kp0 = torch.cat(kp0)
        kp1 = torch.cat(kp1)

        kH0 = torch.cat(kH0)
        kH1 = torch.cat(kH1)

        kr0 = torch.cat(kr0)
        kr1 = torch.cat(kr1)
        
        if use_w:
            w0 = torch.cat(w0)
            w1 = torch.cat(w1)
            
        if preserve_order:
            rank0 = torch.cat(rank0)
            rank1 = torch.cat(rank1)
            
        if counter:
            counter0 = torch.cat(counter0)
            counter1 = torch.cat(counter1)
          
        if ('desc' in pipe_data) and bring_desc:  
            dd0 = torch.cat(dd0)
            dd1 = torch.cat(dd1)
          
        if 'm_idx' in pipe_data:
            m_idx = torch.cat(m_idx)
            m_val = torch.cat(m_val)
            m_mask = torch.cat(m_mask)
            
        if bring_desc:
            if (kp0.shape[0] != dd0.shape[0]) or (kp1.shape[0] != dd1.shape[0]):
                bring_desc = False
            
    if sampling_mode is not None:
        kp0_unsampled = kp0.clone()
        kp1_unsampled = kp1.clone()
        
        kp0 = ((kp0 + sampling_offset) / sampling_scale).round() * sampling_scale - sampling_offset
        kp1 = ((kp1 + sampling_offset) / sampling_scale).round() * sampling_scale - sampling_offset
        
        if overlapping_cells:
            kp0_ = ((kp0 + sampling_offset + (sampling_scale / 2) ) / sampling_scale).round() * sampling_scale - sampling_offset - (sampling_scale / 2)
            kp1_ = ((kp1 + sampling_offset + (sampling_scale / 2) ) / sampling_scale).round() * sampling_scale - sampling_offset - (sampling_scale / 2)

            s0 = ((kp0_unsampled - kp0)**2).sum(dim=1) > ((kp0_unsampled - kp0_)**2).sum(dim=1)
            s1 = ((kp1_unsampled - kp1)**2).sum(dim=1) > ((kp1_unsampled - kp1_)**2).sum(dim=1)

            kp0[s0] = kp0_[s0]
            kp1[s1] = kp1_[s1]
                        
        if 'm_idx' in pipe_data:
            m0_idx = m_idx[:, 0]
            m1_idx = m_idx[:, 1]
            ms_val = m_val
            ms_mask = m_mask
        else:
            m0_idx = None
            m1_idx = None
            ms_val = None
            ms_mask = None
        
        kp0 = sampling(sampling_mode, kp0, kp0_unsampled, kr0, m0_idx, ms_val, ms_mask, counter=counter0)            
        kp1 = sampling(sampling_mode, kp1, kp1_unsampled, kr1, m1_idx, ms_val, ms_mask, counter=counter1)            
            
    if unique:
        if 'm_idx' in pipe_data:
           
            valid_m = (
                (m_idx[:, 0] >= 0) & (m_idx[:, 0] < kp0.shape[0]) &
                (m_idx[:, 1] >= 0) & (m_idx[:, 1] < kp1.shape[0])
            )
            m_idx = m_idx[valid_m]
            m_val = m_val[valid_m]
            m_mask = m_mask[valid_m]

            idx = torch.argsort(m_val, descending=True, stable=True)

            m_idx = m_idx[idx]
            m_val = m_val[idx]
            m_mask = m_mask[idx]

            idx = torch.argsort(m_mask.type(torch.float), descending=True, stable=True)

            m_idx = m_idx[idx]
            m_val = m_val[idx]
            m_mask = m_mask[idx]

            idx0 = torch.full((kp0.shape[0], ), m_idx.shape[0], device=device, dtype=torch.int)
            for i in range(m_idx.shape[0] - 1, -1, -1):
                idx0[m_idx[i, 0]] = i            
            idx0 = torch.argsort(idx0, stable=True)
            
            idx1 = torch.full((kp1.shape[0], ), m_idx.shape[0], device=device, dtype=torch.int)
            idx1[:] = m_idx.shape[0] + 1
            for i in range(m_idx.shape[0] - 1, -1, -1):
                idx1[m_idx[i, 1]] = i            
            idx1 = torch.argsort(idx1, stable=True)
            
        if 'kp' in pipe_data:
            if not preserve_order:
                rank0 = None
                rank1 = None
            
            if patch_matters:
                kkp0 = torch.cat((kp0, kH0.reshape(-1, 9)), dim=1)
            else:
                kkp0 = kp0.clone()
            
            idx0u, idx0r = sortrows(kkp0, idx0, rank0)
            
            if counter:
                counter_new = torch.zeros(idx0u.shape[0], device=device)
                for i in range(idx0r.shape[0]):
                    counter_new[idx0r[i]] = counter_new[idx0r[i]] + counter0[i] 
                counter0 = counter_new
            
            kp0 = kp0[idx0u]
            kH0 = kH0[idx0u]
            kr0 = kr0[idx0u]
            
            if ('desc' in pipe_data) and bring_desc:
                dd0 = dd0[idx0u]

            if patch_matters:
                kkp1 = torch.cat((kp1, kH1.reshape(-1, 9)), dim=1)
            else:
                kkp1 = kp1.clone()

            idx1u, idx1r = sortrows(kkp1, idx1, rank1)
 
            if counter:
                counter_new = torch.zeros(idx1u.shape[0], device=device)
                for i in range(idx1r.shape[0]):
                    counter_new[idx1r[i]] = counter_new[idx1r[i]] + counter1[i] 
                counter1 = counter_new
  
            kp1 = kp1[idx1u]
            kH1 = kH1[idx1u]
            kr1 = kr1[idx1u]
            
            if ('desc' in pipe_data) and bring_desc:
                dd1 = dd1[idx1u]
                        
            if use_w:
                w0 = w0[idx0u]
                w1 = w1[idx1u]
                
            if preserve_order:
                rank0 = rank0[idx0u]
                rank1 = rank1[idx1u]
                            
            if 'm_idx' in pipe_data:
                m_idx_new = torch.cat((idx0r[m_idx[:, 0]].unsqueeze(1), idx1r[m_idx[:, 1]].unsqueeze(1)), dim=1)
                idxmu, _ = sortrows(m_idx_new.clone())
                m_idx = m_idx_new[idxmu]
                m_val = m_val[idxmu]
                m_mask = m_mask[idxmu]
    
    if no_unmatched and ('m_idx' in pipe_data):
        t0 = torch.zeros(kp0.shape[0], device=device, dtype=torch.bool)
        t1 = torch.zeros(kp1.shape[0], device=device, dtype=torch.bool)
        
        if preserve_order:
            t0[rank0 < q_rank0] = True
            t1[rank1 < q_rank1] = True

        t0[m_idx[:, 0]] = True
        t1[m_idx[:, 1]] = True
        
        idx0 = t0.cumsum(dim=0) - 1
        idx1 = t1.cumsum(dim=0) - 1

        m_idx = torch.cat((idx0[m_idx[:, 0]].unsqueeze(1), idx1[m_idx[:, 1]].unsqueeze(1)), dim=1)

        kp0 = kp0[t0]
        kH0 = kH0[t0]
        kr0 = kr0[t0]
        
        if ('desc' in pipe_data) and bring_desc:
            dd0 = dd0[t0]

        kp1 = kp1[t1]
        kH1 = kH1[t1]
        kr1 = kr1[t1]
        
        if ('desc' in pipe_data) and bring_desc:
            dd1 = dd1[t1]
        
        if use_w:
            w0 = w0[t0]            
            w1 = w1[t1]            
        
        if preserve_order:
            rank0 = rank0[t0]
            rank1 = rank1[t1]
            
        if counter:
            counter0 = counter0[t0]
            counter1 = counter1[t1]
            
    if preserve_order:
        if 'kp' in pipe_data:        
            idx0 = torch.argsort(rank0)
            idr0 = torch.argsort(idx0)

            idx1 = torch.argsort(rank1)
            idr1 = torch.argsort(idx1)

            kp0 = kp0[idx0]
            kH0 = kH0[idx0]
            kr0 = kr0[idx0]

            if ('desc' in pipe_data) and bring_desc:
                dd0 = dd0[idx0]
    
            kp1 = kp1[idx1]
            kH1 = kH1[idx1]
            kr1 = kr1[idx1]
            
            if ('desc' in pipe_data) and bring_desc:
                dd1 = dd1[idx1]
                    
            if use_w:
                w0 = w0[idx0]            
                w1 = w1[idx1] 
                
            if counter:
                counter0 = counter0[idx0]
                counter1 = counter1[idx1]
                
            if 'm_idx' in pipe_data:
                m_idx = torch.cat((idr0[m_idx[:, 0]].unsqueeze(1), idr1[m_idx[:, 1]].unsqueeze(1)), dim=1)
        
    pipe_data_out = {}
                
    if 'kp' in pipe_data:
        pipe_data_out['kp'] = [kp0.to(io_device), kp1.to(io_device)]
        pipe_data_out['kH'] = [kH0.to(io_device), kH1.to(io_device)]
        pipe_data_out['kr'] = [kr0.to(io_device), kr1.to(io_device)]

        if ('desc' in pipe_data) and bring_desc:
            pipe_data_out['desc'] = [dd0.to(io_device), dd1.to(io_device)]
                    
        if use_w:
            w0[:, :2] = kp0
            w1[:, :2] = kp1
            pipe_data_out['w'] = [w0.to(io_device), w1.to(io_device)]
            
        if counter:
            pipe_data_out['k_counter'] = [counter0.to(io_device), counter1.to(io_device)]
            
        if 'm_idx' in pipe_data:
            pipe_data_out['m_idx'] = m_idx.to(io_device)
            pipe_data_out['m_val'] = m_val.to(io_device)
            pipe_data_out['m_mask'] = m_mask.to(io_device)
                
    return pipe_data_out


def sampling(sampling_mode, kp, kp_unsampled, kr, ms_idx, ms_val, ms_mask, counter=None, device=None):
    """
    Refines keypoint coordinates based on multiple observations and reliability scores.

    This function resolves 'collisions' where multiple matches point to the same 
    keypoint index. It uses sorting to group these observations and then applies 
    statistical or heuristic rules (Averaging or Best-Selection) to update 
    the final keypoint positions.

    Args:
        sampling_mode (str): The strategy to use ('raw', 'best', 'avg_all_matches', 
                             'avg_inlier_matches').
        kp (torch.Tensor): The current keypoint coordinates (to be updated).
        kp_unsampled (torch.Tensor): Original, high-precision coordinates before any rounding.
        kr (torch.Tensor): Reliability/confidence scores for the keypoints.
        ms_idx, ms_val, ms_mask: Matching indices, values, and inlier masks from RANSAC.
        counter (torch.Tensor, optional): A frequency weight for how often a point was seen.

    Returns:
        torch.Tensor: The refined and updated keypoint coordinates.
    """
    if device is None: device = torch.device('cpu')

    if (sampling_mode == 'raw') or (kp.shape[0] == 0): return kp
            
    if (sampling_mode == 'avg_all_matches'):                    
        if counter is None:
            counter = torch.full((kp.shape[0], ), 1, device=device, dtype=torch.bool)
        
    if (sampling_mode == 'avg_inlier_matches'):                
        if ms_idx is None:
            mask = torch.full((kp.shape[0], ), 1, device=device, dtype=torch.bool)
        else:            
            mask = torch.zeros(kp.shape[0], device=device, dtype=torch.bool)
            for i in torch.arange(ms_idx.shape[0]):
                mask[ms_idx[i]] = ms_mask[i]
                
        if counter is None:
            counter = torch.zeros(kp.shape[0], device=device)
            for i in torch.arange(ms_idx.shape[0]):
                if ms_mask[i]: counter[ms_idx[i]] = 1            
            
    if (sampling_mode == 'best'):
        if ms_idx is None:
            mask = torch.full((kp.shape[0], ), 1, device=device, dtype=torch.bool)  
            val = torch.full((kp.shape[0], ), 1, device=device)  
        else:
            mask = torch.zeros(kp.shape[0], device=device, dtype=torch.bool)
            for i in torch.arange(ms_idx.shape[0]):
                mask[ms_idx[i]] = ms_mask[i]
    
            val = torch.zeros(kp.shape[0], device=device)
            for i in torch.arange(ms_idx.shape[0]):
                val[ms_idx[i]] = ms_val[i]
    
    aux = kp.clone()
    idx = torch.arange(len(aux), device=device)
    for i in range(aux.shape[1] - 1, -1, -1):            
        sidx = torch.argsort(aux[:, i], stable=True)
        idx = idx[sidx]
        aux = aux[sidx]            
    
    i = 0
    j = 1
    while j < aux.shape[0]:
        if torch.all(aux[i] == aux[j]):
            j = j + 1
            continue

        if (sampling_mode == 'avg_all_matches'): 
            c_sum = counter[idx[i:j]].sum()
            kp[idx[i:j]] = (kp_unsampled[idx[i:j]] * counter[idx[i:j]].unsqueeze(-1)).sum(dim=0) / c_sum
            
        if (sampling_mode == 'avg_inlier_matches'):
            tmp = kp_unsampled[idx[i:j]]
            tmp_c = counter[idx[i:j]]
            tmp_mask = mask[idx[i:j]]
            if torch.any(tmp_mask):
                c_sum = tmp_c[tmp_mask].sum()                
                kp[idx[i:j]] = (tmp[tmp_mask] * tmp_c[tmp_mask].unsqueeze(-1)).sum(dim=0) / c_sum

        if (sampling_mode == 'best'):
            tmp_mask = torch.stack((mask[idx[i:j]], val[idx[i:j]], kr[idx[i:j]]), dim=1)
            
            max_idx = 0
            max_val = tmp_mask[0]
            for q in torch.arange(1, tmp_mask.shape[0]):
                if (max_val[0] < tmp_mask[q][0]) or ((max_val[0] == tmp_mask[q][0]) and (max_val[1] < tmp_mask[q][1])) or ((max_val[0] == tmp_mask[q][0]) and (max_val[1] == tmp_mask[q][1]) and (max_val[2] < tmp_mask[q][2])):
                    max_val = tmp_mask[q]
                    max_idx = q
            
            best = idx[i:j][max_idx]
            kp[idx[i:j]] = kp_unsampled[best]            
                        
        i = j     
        j = j + 1

    if (sampling_mode == 'avg_all_matches'):
        c_sum = counter[idx[i:j]].sum()
        kp[idx[i:j]] = (kp_unsampled[idx[i:j]] * counter[idx[i:j]].unsqueeze(-1)).sum(dim=0) / c_sum
        
    if (sampling_mode == 'avg_inlier_matches'):
        tmp = kp_unsampled[idx[i:j]]
        tmp_c = counter[idx[i:j]]
        tmp_mask = mask[idx[i:j]]
        if torch.any(tmp_mask):
            c_sum = tmp_c[tmp_mask].sum()                
            kp[idx[i:j]] = (tmp[tmp_mask] * tmp_c[tmp_mask].unsqueeze(-1)).sum(dim=0) / c_sum
    
    if (sampling_mode == 'best'):
        tmp_mask = torch.stack((mask[idx[i:j]], val[idx[i:j]], kr[idx[i:j]]), dim=1)
        
        max_idx = 0
        max_val = tmp_mask[0]
        for q in torch.arange(1, tmp_mask.shape[0]):
            if (max_val[0] < tmp_mask[q][0]) or ((max_val[0] == tmp_mask[q][0]) and (max_val[1] < tmp_mask[q][1])) or ((max_val[0] == tmp_mask[q][0]) and (max_val[1] == tmp_mask[q][1]) and (max_val[2] < tmp_mask[q][2])):
                max_val = tmp_mask[q]
                max_idx = q
        
        best = idx[i:j][max_idx]
        kp[idx[i:j]] = kp_unsampled[best]  
       
    return kp


def sortrows(kp, idx_prev=None, rank=None, device=None): 
    """
    Sorts a set of keypoints lexicographically and identifies unique points.

    This function performs a stable sort across coordinate columns (typically X and Y)
    to group identical points together. It is essential for 'cleaning' a 
    keypoint database after merging results from different matchers.

    Args:
        kp (torch.Tensor): The keypoint coordinates (N x 2).
        idx_prev (torch.Tensor, optional): Previous indices if performing 
            nested sorting.
        rank (torch.Tensor, optional): A quality score (e.g., confidence). 
            If multiple identical points exist, the one with the best 
            (lowest) rank is preserved.

    Returns:
        tuple: (idxa, idxb)
            idxa: Indices of the unique, best-ranked keypoints.
            idxb: A 'reverse map' that tells every original keypoint 
                  which unique index it now belongs to.
    """   
    if device is None: device = torch.device('cpu')

    idx = torch.arange(kp.shape[0], device=device)

    if idx_prev is not None:
        idx = idx[idx_prev]
        kp = kp[idx_prev]
        
        if rank is not None:
            rank = rank[idx_prev]
        
    for i in range(kp.shape[1] - 1, -1, -1):            
        sidx = torch.argsort(kp[:, i], stable=True)
        idx = idx[sidx]
        kp = kp[sidx]

        if rank is not None:
            rank = rank[sidx]

    idxa = torch.zeros(kp.shape[0], device=device, dtype=torch.int)
    idxb = torch.zeros(kp.shape[0], device=device, dtype=torch.int)

    k = 0
    cur = torch.zeros((0, 2), device=device)
    for i in range(kp.shape[0]):
        if (cur.shape[0] == 0) or (not torch.all(kp[i] == cur)):
            cur = kp[i]
            idxa[k] = idx[i]                                        
            k = k + 1
            
            if rank is not None:
                cur_rank = rank[i]

        if rank is not None:
            if cur_rank > rank[i]:
                cur_rank = rank[i]
                idxa[k - 1] = idx[i]
            
        idxb[idx[i]] = k - 1
            
    idxa = idxa[:k]

    return idxa, idxb


class sampling_module:
    """
    A data management module for filtering and merging keypoints and matches.

    This module handles the 'bookkeeping' of a matching pipeline. It ensures 
    that if multiple matchers or multiple image scales are used, the resulting 
    keypoints are merged logically without duplicates. It also provides 
    strategies for 'thinning' dense matches to improve downstream performance.

    Attributes:
        sampling_mode (str): The strategy for merging/filtering points 
            (e.g., 'raw', 'best', 'avg_inlier_matches').
        unique (bool): If True, removes duplicate keypoints at the same 
            pixel location.
        only_matched (bool): If True, discards any keypoints that do not 
            belong to a valid correspondence pair.
        sampling_scale (int): Used to sub-sample matches (e.g., keeping 
            every N-th match).
    """
    def __init__(self, **args):
        from core import set_args
        self.single_image = False
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = True
                
        self.args = {
            'id_more': '',
            'unique': True,
            'no_unmatched': True,
            'only_matched': True,
            'sampling_mode': 'raw', # None, raw, best, avg_inlier_matches, avg_all_matches
            'overlapping_cells': False,
            'sampling_scale': 1,
            'sampling_offset': 0,
            }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
        
        self.id_string, self.args = set_args('sampling', args, self.args)        


    def get_id(self): 
        return self.id_string
    
    
    def finalize(self):
        return


    def run(self, **args):           
        pipe_data = args

        return pipe_union(pipe_data, unique=self.args['unique'],
                          no_unmatched=self.args['no_unmatched'],
                          only_matched=self.args['only_matched'],
                          sampling_mode=self.args['sampling_mode'],
                          sampling_scale=self.args['sampling_scale'],
                          sampling_offset=self.args['sampling_offset'],
                          overlapping_cells=self.args['overlapping_cells'])    
