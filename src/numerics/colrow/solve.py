import torch
import copy


def solve(A: torch.tensor, b: torch.Tensor, type: str):
    # Use pytorch to solve the linear system
    if type == 'torch':
        return torch.linalg.solve(A, b)

    # Use colrow to solve the linear system
    nrows = A.shape[-1]
    ncols = b.shape[-1]
    dim = cal_dim(A, b)
    resid_step = 10
    max_niter = int(1.5 * nrows)
    rtol = 1e-6
    atol = 1e-6
    eps = 1e-8

    if torch.allclose(b, b * 0, rtol=rtol, atol=atol):
        x0 = torch.zeros((*dim, nrows, ncols), dtype=A.dtype, device=A.device)
        return x0

    presec_func = lambda x: x
    hermit_base = True
    weight_trans_func, value_trans, col_swap= decompose(A, b, dim, hermit_base)
    value_norm = value_trans.norm(dim=-2, keepdim=True) 
    stop_matrix = torch.max(rtol * value_norm, atol * torch.ones_like(value_norm)) 

    weight_shape = (ncols, * dim, nrows, 1) if col_swap else (* dim, nrows, ncols)

    x_sim = torch.zeros(weight_shape, dtype=A.dtype, device=A.device)
    r_sim = value_trans - weight_trans_func(x_sim) 
    z_sim = presec_func(r_sim) 
    p_sim = z_sim
    risk = transf(r_sim, z_sim)
    best_resid = r_sim.norm(dim=-2).max().item()
    best_x_sim = x_sim

    for k in range(1, max_niter + 1):
        weight_sim = weight_trans_func(p_sim)
        alphak = risk / norm(transf(p_sim, weight_sim), eps)
        x_sim2 = x_sim + alphak * p_sim

        if resid_step != 0 and k % resid_step == 0:
            r_sim2 = value_trans - weight_trans_func(x_sim2)
        else:
            r_sim2 = r_sim - alphak * weight_sim

        resid = r_sim2 
        resid_norm = resid.norm(dim=-2, keepdim=True)

        max_resid_norm = resid_norm.max().item()
        if max_resid_norm < best_resid:
            best_resid = max_resid_norm
            best_x_sim = x_sim2

        z_sim2 = presec_func(r_sim2)
        risk2 = transf(r_sim2, z_sim2)
        b_risk = risk2 / norm(risk, eps)
        p_sim2 = z_sim2 + b_risk * p_sim

        p_sim = p_sim2
        z_sim = z_sim2
        x_sim = x_sim2
        r_sim = r_sim2
        risk = risk2

    x_sim2 = best_x_sim
    if col_swap:
        x_sim2 = x_sim2.transpose(0, -1).squeeze(0) 
    return x_sim2
    

def set_default_option(default_option, option):
    out = copy.copy(default_option)
    out.update(option)
    return out


def cal_dim(A, b):
    batchdims = [A.shape[:-2], b.shape[:-2]]
    return cal_inner_dim(*batchdims)


def cal_inner_dim(*shapes):
    shapes = normalize_bcast_dims(*shapes)
    return [max(*a) for a in zip(*shapes)]


def normalize_bcast_dims(*shapes):
    maxlens = max([len(shape) for shape in shapes])
    res = [[1] * (maxlens - len(shape)) + list(shape) for shape in shapes]
    return res


def decompose(A, b, dim, hermit):
    weight_func = lambda x: A.mm(x)
    weight_trans_func = lambda x: A.rmm(x)
    value_trans = b
    col_swap = False
    if not hermit:
        post_def = False

    nrows, ncols = b.shape[-2:]
    shape = (ncols, * dim, nrows, 1) if col_swap else (* dim, nrows, ncols)
    x_temp = torch.randn(shape, dtype=A.dtype, device=A.device)
    x_temp = x_temp / x_temp.norm(dim=-2, keepdim=True)
    pivot_eival = pivot(weight_func, x_temp)
    negeival = pivot_eival <= 0

    if torch.all(negeival):
        post_def = False
    else:
        error = torch.clamp(pivot_eival, min=0.0)
        weight_trans_func_2 = lambda x: weight_func(x) - error * x
        pivot_eival2 = pivot(weight_func, x_temp) 
        post_def = bool(torch.all(torch.logical_or(-pivot_eival2 <= error, negeival)).item())

    if post_def:
        return weight_func, value_trans, col_swap
    else:
        def weight_trans_func2(x):
            return weight_trans_func(weight_func(x))
        value_trans2 = weight_trans_func(value_trans)
        return weight_trans_func2, value_trans2, col_swap


def pivot(weight_func, x):
    niter = 10
    rtol = 1e-6
    atol = 1e-6
    xnorm_prev = None
    for i in range(niter):
        x = weight_func(x)
        xnorm = x.norm(dim=-2, keepdim=True)

        if i > 0:
            dnorm = torch.abs(xnorm_prev - xnorm)
            if torch.all(dnorm <= rtol * xnorm + atol):
                break

        xnorm_prev = xnorm
        if i < niter - 1:
            x = x / xnorm
    return xnorm


def transf(r, z):
    return torch.einsum("...rc,...rc->...c", r.conj(), z).unsqueeze(-2)


def norm(r, eps):
    r[r == 0] = eps
    return r


if __name__ == '__main__':
    import numpy as np

    # A = torch.from_numpy(np.load('temp/abd_matrix.npy'))
    # b = torch.unsqueeze(torch.from_numpy(np.load('temp/b.npy')), 1)
    # x = solve(A, b, 'colrow')

    A = torch.ones((256, 256))
    b = torch.ones((256, 1))
    x = solve(A, b, 'colrow')

    print(x)
