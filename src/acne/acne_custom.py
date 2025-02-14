from .demo_util import get_T_from_K, norm_points_with_T
from .utils import norm_points
from .network import MyNetwork

import numpy as np

def prepare_xs(xs, K1, K2, use_fundamental=0):
    """
    Prepare xs for model
    Inputs:
        xs: Nx4, Input correspondences in original image coordinates
        K: 3x3, Calibration matrix
        use_fundamental:
            0 means E case, xs is calibrated; 
            1 means F case, xs is normalized with mean, std of coordinates;
            2 means F case, xs is normalized with image size.
    Returns:
        xs: Nx4, Input correspondences in calibrated (E) or normed (F) coordinates
        T: 3x3, Transformation matrix used to normalize input in the case of F. None in the case of E.
    """
    x1, x2 = xs[:,:2], xs[:,2:4]
    if use_fundamental>0:
        # Normalize Points
        if use_fundamental == 1:
            # normal normalization
            x1, T1 = norm_points(x1)
            x2, T2 = norm_points(x2)
        elif use_fundamental == 2:
            # we used img_size normization
            T1 = get_T_from_K(K1)
            T2 = get_T_from_K(K2)
            x1 = norm_points_with_T(x1, T1)
            x2 = norm_points_with_T(x2, T2)
        else:
            raise NotImplementedError
        xs = np.concatenate([x1,x2],axis=-1).reshape(-1,4)
    else:
        # Calibrate Points with intrinsics
        x1 = (
            x1 - np.array([[K1[0,2], K1[1,2]]])
            ) / np.array([[K1[0,0], K1[1,1]]])
        x2 = (
            x2 - np.array([[K2[0,2], K2[1,2]]])
            ) / np.array([[K2[0,0], K2[1,1]]])
        xs = np.concatenate([x1,x2],axis=-1).reshape(-1,4)
        T1, T2 = None, None

    return xs, T1, T2

class NetworkTest(MyNetwork):

    def __init__(self, config, model_path):
        super(NetworkTest, self).__init__(config)
        
        # restore from model_path
        self.saver_best.restore(self.sess, model_path)
        
    def compute_E(self, xs):
        """
        Compute E/F given a set of putative correspondences. The unit weight vector 
        for each correspondenece is also given.
        Input:
            xs: BxNx4
        Output:
            out_e: Bx3x3
            w_com: BxN
            socre_local: BxN
        """
        _xs = np.array(xs).reshape(1, 1, -1, 4)
        feed_dict = {
            self.x_in: _xs,
            self.is_training: False, 
        }
        fetch = {
            "w_com": self.last_weights,
            "score_local":self.last_logit,
            "out_e": self.out_e_hat
        }
        res = self.sess.run(fetch, feed_dict=feed_dict)
        batch_size = _xs.shape[0]
        score_local = res["score_local"] # 
        w_com = res["w_com"] # Combined weights
        out_e = res["out_e"].reshape(batch_size, 3, 3)
        return out_e, w_com, score_local