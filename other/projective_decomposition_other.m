function [H_p,H_a,H_t,H_s,H_m,H_r]=projective_decomposition_other(H)
% H=H_p*H_a*H_t*H_s*H_m*H_r

A_=H(1:2,1:2);
a_=H(3,1:2)';
t_=H(1:2,3);
a=H(3,3);

% RQ decomposition
P=[0 1; 1 0];
[R_,K_]=qr(A_'*P);
K_=P*K_'*P;
R_=P*R_';

M_=eye(2);
% the determinant inversion can be obtained by multiplying a row or a column by -1
% this is done for instance by the orthogonal matrix [1 0; 0 -1] (reflection matrix)
% notice that [1 0; 0 -1]*[1 0; 0 -1]=eye(2);
if sign(det(K_))<0
    K_=K_*[1 0; 0 -1]; % this change the sign of det
    R_=[1 0; 0 -1]*R_; % this is the inverse to nullify the total effect
end
if sign(det(R_))<0
    R_=[1 0; 0 -1]*R_; % this change the sign of det
    M_=M_*[1 0; 0 -1]; % this is the inverse to nullify the total effect
end
s=sqrt(abs(det(K_*R_)));
K=K_/sqrt(abs(det(K_)));
R=R_/sqrt(abs(det(R_)));
M=M_;

V=inv(A_')*a_;
T=inv(K)*t_;
v=a-V'*t_;

H_p=[1 0 0; 0 1 0; V' v];        % projective
H_a=[K [0 0]'; 0 0 1];           % affine
H_m=[M [0 0]'; 0 0 1];           % reflection
H_r=[R [0 0]'; 0 0 1];           % rotation
H_s=[s 0 0; 0 s 0; 0 0 1];       % scale
H_t=[1 0 T(1); 0 1 T(2); 0 0 1]; % translation
