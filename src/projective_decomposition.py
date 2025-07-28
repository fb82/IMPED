import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


H = torch.rand((5, 3, 3), device=device)

def decompose_H(H, ret_err=False):
    # H = torch.rand((5, 3, 3), device=device)
    
    v = H[:, -1, -1]
    V = H[:, -1, :2]
    T = H[:, :2, -1] / v.unsqueeze(-1)
    W = H[:, :2, :2] - T.unsqueeze(-1).bmm(V.unsqueeze(1))
    [R_, K_] = torch.linalg.qr(W)
    M_ = torch.eye(2, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    
    # the determinant inversion can be obtained by multiplying a row or a column by -1
    # this is done for instance by the orthogonal matrix [1 0; 0 -1] (reflection matrix)
    # notice that [1 0; 0 -1]*[1 0; 0 -1]=eye(2);
    
    # det sign
    t = K_.det().sign() < 0
    # this change the sign of det
    K_[t] = torch.tensor([[1., 0.], [0., -1.]], device=device) @ K_[t] 
    R_[t] = R_[t] @ torch.tensor([[1., 0.], [0., -1.]], device=device)
    
    # det sign    
    t = R_.det().sign() < 0    
    # this change the sign of det
    R_[t] = torch.tensor([[1., 0.], [0., -1.]], device=device) @ R_[t]
    # this is the inverse to nullify the total effect
    M_[t] = M_[t] @ torch.tensor([[1., 0.], [0., -1.]], device=device)
    
    s = R_.bmm(K_).det().abs() ** 0.5
    K = K_ / (K_.det().abs() ** 0.5).unsqueeze(-1).unsqueeze(-1)
    R = R_ / (R_.det().abs() ** 0.5).unsqueeze(-1).unsqueeze(-1)
    M = M_
    
    # projective
    H_p = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_p[:, -1, :2] = V
    H_p[:, -1, -1] = v
     
    # affine
    H_a = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_a[:, :2, :2] = K
    
    # rotation
    H_r = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_r[:, :2, :2] = R
    
    # reflection
    H_m = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_m[:, :2, :2] = M
    
    # scale
    H_s = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_s[:, 0, 0] = s
    H_s[:, 1, 1] = s
    
    # translation
    H_t = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_t[:, :2, -1] = T
    
    if ret_err:    
        err = (H - H_t.bmm(H_s.bmm(H_m.bmm(H_r.bmm(H_a.bmm(H_p)))))).abs().sum(dim=(1, 2))
    else:
        err = None
        
    return H_t, H_s, H_m, H_r, H_a, H_p, err


def decompose_H_other(H, ret_err=False):
    # H = torch.rand((5, 3, 3), device=device)
    
    A_ = H[:, :2, :2]
    a_ = H[:, -1, :2]
    t_ = H[:, :2, -1]
    a = H[:, -1, -1]
    
    # RQ decomposition
    P = torch.tensor([[0., 1.], [1., 0.]], device=device)
    R_, K_ = torch.linalg.qr(A_.permute(0, 2, 1) @ P)
    K_ = P @ K_.permute(0, 2, 1) @ P
    R_ = P @ R_.permute(0, 2, 1)
        
    M_ = torch.eye(2, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)

    # the determinant inversion can be obtained by multiplying a row or a column by -1
    # this is done for instance by the orthogonal matrix [1 0; 0 -1] (reflection matrix)
    # notice that [1 0; 0 -1]*[1 0; 0 -1]=eye(2);

    # det sign
    t = K_.det().sign() < 0
    # this change the sign of det
    K_[t] =  K_[t] @ torch.tensor([[1., 0.], [0., -1.]], device=device)
    R_[t] = torch.tensor([[1., 0.], [0., -1.]], device=device) @ R_[t]

    # det sign    
    t = R_.det().sign() < 0    
    # this change the sign of det
    R_[t] = torch.tensor([[1., 0.], [0., -1.]], device=device) @ R_[t]
    # this is the inverse to nullify the total effect
    M_[t] = M_[t] @ torch.tensor([[1., 0.], [0., -1.]], device=device)

    s = R_.bmm(K_).det().abs() ** 0.5
    K = K_ / (K_.det().abs() ** 0.5).unsqueeze(-1).unsqueeze(-1)
    R = R_ / (R_.det().abs() ** 0.5).unsqueeze(-1).unsqueeze(-1)
    M = M_

    V = torch.linalg.inv(A_.permute(0, 2, 1)).bmm(a_.unsqueeze(-1))
    T = torch.linalg.inv(K).bmm(t_.unsqueeze(-1))
    v = a - V.permute(0, 2, 1).bmm(t_.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    # projective
    H_p = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_p[:, -1, :2] = V.squeeze(-1)
    H_p[:, -1, -1] = v
     
    # affine
    H_a = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_a[:, :2, :2] = K
    
    # rotation
    H_r = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_r[:, :2, :2] = R
    
    # reflection
    H_m = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_m[:, :2, :2] = M
    
    # scale
    H_s = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_s[:, 0, 0] = s
    H_s[:, 1, 1] = s
    
    # translation
    H_t = torch.eye(3, device=device).unsqueeze(0).repeat(H.shape[0], 1, 1)
    H_t[:, :2, -1] = T.squeeze(-1)
    
    if ret_err:    
        err = (H - H_p.bmm(H_a.bmm(H_t.bmm(H_s.bmm(H_m.bmm(H_r)))))).abs().sum(dim=(1, 2))
    else:
        err = None
        
    return H_p, H_a, H_t, H_s, H_m, H_r, err

H_t, H_s, H_m, H_r, H_a, H_p, err = decompose_H(H, ret_err=True)
H_p_, H_a_, H_t_, H_s_, H_m_, H_r_, err_ = decompose_H_other(H, ret_err=True)