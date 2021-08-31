import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pgd import pgd
from sklearn import linear_model


def binary_search(cond, Xs, ys, epsilons, anchor_masks, targeted, targets,
                  upper=np.inf, lower=0.0, tol=0.00001, max_steps=100, upscale_mult=10.0,
                  downscale_mult=10.0, lower_limit=1e-8, upper_limit=1e+3):
    """batch binary search for attack approximation of robustness
    """
    batch_size = len(Xs)
    upper_not_found = np.full(batch_size, True)
    lower_not_found = np.full(batch_size, True)
    upper = np.full(batch_size, upper)
    lower = np.full(batch_size, lower)
    
    Xs_pgd = torch.zeros_like(Xs)  # placeholder for perturbed images
    
    step = 0
    run_idx = np.arange(batch_size)
    while len(run_idx) != 0:
        if targeted:
            cur_successes, cur_Xs_pgd = cond(Xs[run_idx], ys[run_idx], epsilons[run_idx], anchor_masks[run_idx],
                                             targeted, targets[run_idx])
        else:
            cur_successes, cur_Xs_pgd = cond(Xs[run_idx], ys[run_idx], epsilons[run_idx], anchor_masks[run_idx],
                                             targeted, None) 
        for cur_i, run_i in enumerate(run_idx):
            if cur_successes[cur_i]: # success at current value (attack success)
                Xs_pgd[run_i] = cur_Xs_pgd[cur_i]
                if lower_not_found[run_i]: # always success, we have not found the true lower bound
                    upper[run_i] = epsilons[run_i]
                    epsilons[run_i] /= downscale_mult  # downscale search range, try to find an pseudo lower bound
                else:
                    upper[run_i] = epsilons[run_i]
                    epsilons[run_i] = 0.5 * (lower[run_i] + upper[run_i])
                upper_not_found[run_i] = False # when initial is a success we always have a valid upper bound
            else:
                if upper_not_found[run_i]:
                    lower[run_i] = epsilons[run_i]
                    epsilons[run_i] *= upscale_mult
                else:
                    lower[run_i] = epsilons[run_i]
                    epsilons[run_i] = 0.5 * (lower[run_i] + upper[run_i]) # when initial is a failure we have a pseudo lower bound
                lower_not_found[run_i] = False
        step += 1
        not_meet_tolerance = (upper - lower) > tol  # not meet tolerance
        not_exceed_limit = np.logical_and((epsilons < upper_limit), (epsilons > lower_limit)) # not exceed limit
        run_cond = np.logical_and(not_meet_tolerance, not_exceed_limit)
        run_idx = np.arange(batch_size)[run_cond] 
        if step >= max_steps:
            break

    return upper, Xs_pgd


def uniform_sampling(X, num_samples, cur_anchor=None):
    """direct uniform sampling
    """
    if cur_anchor is not None:
        # sample from the rest of un-anchored features
        samples = np.repeat(cur_anchor.reshape(1, -1), num_samples, axis=0)
        samples_flatten = np.random.choice([0, 1], size=int(num_samples * (28*28 - cur_anchor.sum())), replace=True, p=[0.8, 0.2])
        samples[samples!=1] = samples_flatten
    else:
        samples = np.random.choice([0,1], size=(num_samples, X.shape[1], X.shape[2], X.shape[3]), replace=True, p=[0.8, 0.2])
    return samples, samples


def iterative_sampling(X, num_samples, cur_anchor, num_new_anchor):
    samples = np.repeat(cur_anchor.reshape(1, -1), num_samples, axis=0)
    samples_flatten = np.zeros((num_samples, int(28*28-cur_anchor.sum())))
    samples_flatten[:, :num_new_anchor] = 1
    samples_flatten = np.apply_along_axis(np.random.permutation, 1, samples_flatten)
    samples[samples == 0] = samples_flatten.reshape(-1)
    Zs = samples_flatten
    return samples, Zs


def evaluate_robustness(Xs, ys, model, targeted, targets, norm, anchors, eps_start, step_size, niters, device,
                        reverse_anchor=False, batch_size=10000):
    """evaluating robustness for different anchors for a given X"""
    assert Xs.shape == (1, 1, 28, 28)
    assert ys.shape == (1,)
    if targeted:
        assert targets.shape == (1,)
    
    total = range(len(anchors))
    num_batch = len(total) // batch_size
    if len(total) % batch_size != 0:
        num_batch += 1
    batches = [total[i*batch_size:(i+1)*batch_size] for i in range(num_batch)]
    robust_ub = np.full(len(total), np.inf)
    for batch in batches:
        Xs_repeat = Xs.repeat(len(batch), 1, 1, 1)
        ys_repeat = ys.repeat(len(batch))
        eps_repeat = np.full(len(batch), eps_start)
        if targeted:
            targets_repeat = targets.repeat(len(batch))
        cur_anchors = torch.BoolTensor(anchors[batch]).to(device)
        if reverse_anchor:
            cur_anchors = ~cur_anchors
        def binary_search_cond(Xs, ys, epsilons, anchor_masks, targeted, targets):
            epsilons = torch.FloatTensor(epsilons).to(device)
            successes, Xs_pgd = pgd(Xs, ys, model, epsilons, norm, niters, step_size, anchor_masks, targeted, targets=targets,
                                    box_min=0., box_max=1., verbose=False, multi_start=False, device=device)
            return successes, Xs_pgd
        if targeted:
            robust_ub[batch], _ = binary_search(binary_search_cond, Xs_repeat, ys_repeat, eps_repeat, cur_anchors, targeted, targets_repeat, tol=0.001, max_steps=25)
        else:
            robust_ub[batch], _ = binary_search(binary_search_cond, Xs_repeat, ys_repeat, eps_repeat, cur_anchors, targeted, None, tol=0.001, max_steps=25)
    
    return robust_ub


def greedy(X, y, model, targeted, target, norm, eps_start, step_size, niters, device, reverse_anchor, percentage=0.3, writer=None):
    anchor_map = np.zeros(28 * 28)
    X_np = X.cpu().numpy()
    cur_anchor = np.zeros(28 * 28)
    cur_val = 1000  # for greedy ranking
    while cur_anchor.sum() < 784 * percentage:
        n_sample = int(784 - cur_anchor.sum())
        Zs = np.eye(n_sample)
        anchor_samples = np.repeat(cur_anchor.reshape(1, -1), n_sample, axis=0)
        anchor_samples[anchor_samples != 1] = Zs.reshape(-1)
        anchor_samples = anchor_samples.reshape(n_sample, 1, 28, 28)

        robust_ub = evaluate_robustness(X, y, model, targeted, target, norm, anchor_samples, eps_start,
                                        step_size, niters, device, reverse_anchor=reverse_anchor)
        robust_ub[robust_ub == np.inf] = np.linalg.norm(X_np - np.zeros_like(X_np))

        ks = np.ones(n_sample)
        feature_robust = kernel_regression(Zs, ks, robust_ub)
        if reverse_anchor:
            max_idx = feature_robust.argsort()[:int(784*0.05)]
        else:
            max_idx = feature_robust.argsort()[-int(784*0.05):]  # select four features at a single greedy step (speed up)
        max_idx = np.where(cur_anchor != 1)[0][max_idx]
        cur_anchor[max_idx] = 1
        anchor_map[max_idx] = cur_val
        cur_val -= 2

    anchor_map = anchor_map.reshape(1, 1, 28, 28)
    return anchor_map


def one_step_banzhaf(X, y, model, targeted, target, norm, eps_start, step_size, niters, device, reverse_anchor, writer=None):
    n_sample = 10000

    # anchor_samples = uniform_sampling(X, num_samples)
    X_np = X.cpu().numpy()
    Xs = np.repeat(X_np.reshape(1, -1), n_sample, axis=0)
    Zs = np.apply_along_axis(shap_sampling, 1, Xs, p=0.5)
    anchor_samples = Zs.reshape(n_sample, 1, 28, 28)
    
    robust_ub = evaluate_robustness(X, y, model, targeted, target, norm, anchor_samples, eps_start,
                                    step_size, niters, device, reverse_anchor=reverse_anchor)
    robust_ub[robust_ub == np.inf] = np.linalg.norm(X_np - np.zeros_like(X_np))

    ks = np.ones(n_sample)
    feature_robust = kernel_regression(Zs, ks, robust_ub)
    
    if reverse_anchor:
        feature_robust = -feature_robust
        
    if writer is None:
        return feature_robust
    else:
        raise NotImplementedError()
        fig, name = vis(X, feature_robust, np.zeros_like(X), 'ls_regression', None, tensorboard=True, rescale=False)
        writer.add_figure(name, fig)


def forward_batch(model, input, batchsize):
    inputsize = input.shape[0]
    for count in range((inputsize - 1) // batchsize + 1):
        end = min(inputsize, (count + 1) * batchsize)
        if count == 0:
            tempinput = input[count * batchsize:end]
            out = model(tempinput.cuda())
            out = out.data.cpu().numpy()
        else:
            tempinput = input[count * batchsize:end]
            temp = model(tempinput.cuda()).data
            out = np.concatenate([out, temp.cpu().numpy()], axis=0)
    return out


def shap(X, label, pdt, model, n_sample):
    X = X.cpu().numpy()
    Xs = np.repeat(X.reshape(1, -1), n_sample, axis=0)
    Xs_img = Xs.reshape(n_sample, 1, 28, 28)

    Zs = np.apply_along_axis(sample_shap_Z, 1, Xs)
    Zs_real = np.copy(Zs)
    Zs_real[Zs == 1] = Xs[Zs == 1]
    Zs_real_img = Zs_real.reshape(n_sample, 1, 28, 28)
    Zs_img = Variable(torch.tensor(Xs_img - Zs_real_img), requires_grad=False).float()
    out = forward_batch(model, Zs_img, 5000)
    ys = out[:, label]

    ys = pdt.data.cpu().numpy() - ys
    ks = np.apply_along_axis(shap_kernel, 1, Zs, X=X.reshape(-1))

    expl = kernel_regression(Zs, ks, ys)

    return expl


def kernel_regression(Is, ks, ys):
    """
    *Inputs:
        I: sample of perturbation of interest, shape = (n_sample, n_feature)
        K: kernel weight
    *Return:
        expl: explanation minimizing the weighted least square
    """
    n_sample, n_feature = Is.shape
    IIk = np.matmul(np.matmul(Is.transpose(), np.diag(ks)), Is)
    Iyk = np.matmul(np.matmul(Is.transpose(), np.diag(ks)), ys)
    expl = np.matmul(np.linalg.pinv(IIk), Iyk)
    return expl


def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


def shap_sampling(X, p=0.5):
    nz_ind = np.nonzero(X)[0]
    nz_ind = np.arange(X.shape[0])
    num_nz = len(nz_ind)
    bb = 0
    while bb == 0 or bb == num_nz:
        aa = np.random.rand(num_nz)
        bb = np.sum(aa > p)
    sample_ind = np.where(aa > p)
    Z = np.zeros(len(X))
    Z[nz_ind[sample_ind]] = 1
    return Z


def sample_shap_Z(X):
    nz_ind = np.nonzero(X)[0]
    nz_ind = np.arange(X.shape[0])
    num_nz = len(nz_ind)
    bb = 0
    while bb == 0 or bb == num_nz:
        aa = np.random.rand(num_nz)
        bb = np.sum(aa > 0.5)
    sample_ind = np.where(aa > 0.5)
    Z = np.zeros(len(X))
    Z[nz_ind[sample_ind]] = 1

    return Z


def shap_kernel(Z, X):
    M = X.shape[0]
    z_ = np.count_nonzero(Z)
    return (M-1) * 1.0 / (z_ * (M - 1 - z_) * nCr(M - 1, z_))


def greedy_as(X, y, model, targeted, target, norm, eps_start, step_size, niters, device, reverse_anchor, percentage=0.5):
    print('Computing explanation by Greedy-AS...')
    anchor_map = np.zeros(28 * 28)
    X_np = X.cpu().numpy()
    cur_anchor = np.zeros(28 * 28)
    cur_val = 1000.  # for greedy ranking
    
    while cur_anchor.sum() < 784 * percentage:
        print('number of relevant features selected:', cur_anchor.sum())
        n_sample = 5000
        anchor_samples, Zs = uniform_sampling(X_np, n_sample, cur_anchor)
        anchor_samples = anchor_samples.reshape(n_sample, 1, 28, 28)
        robust_ub = evaluate_robustness(X, y, model, targeted, target, norm, anchor_samples, eps_start,
                                        step_size, niters, device, reverse_anchor=reverse_anchor)
        robust_ub[robust_ub == np.inf] = np.linalg.norm(X_np - np.zeros_like(X_np))

        ks = np.ones(n_sample)
        feature_robust = kernel_regression(Zs, ks, robust_ub)
        if reverse_anchor:
            max_idx = feature_robust[cur_anchor != 1].argsort()[:int(784*0.05)]
        else:
            max_idx = feature_robust[cur_anchor != 1].argsort()[-int(784*0.05):]
        max_idx = np.where(cur_anchor != 1)[0][max_idx]
        cur_anchor[max_idx] = 1
        anchor_map[max_idx] = cur_val
        cur_val -= 2

    anchor_map = anchor_map.reshape(1, 1, 28, 28)
    return anchor_map


def plot_eval_curve(X, y, model, target, norm, eps_start, step_size, niters, writer, anchor, method, reverse_anchor):
    plot_step = 0
    for i in range(5, 50, 5):
        threshold = np.percentile(anchor, 100-i)
        cur_anchor = np.copy(anchor)
        cur_anchor[anchor <= threshold] = 0
        cur_anchor[anchor > threshold] = 1
        if reverse_anchor:
            cur_anchor = np.ones_like(cur_anchor) - cur_anchor
        def binary_search_cond(current_epsilons, cur_anchor_masks):
            success, X_pgd, margin = pgd(None, X.repeat(len(current_epsilons), 1, 1, 1), y.repeat(len(current_epsilons)), 
                        [model], current_epsilons, norm, niters=niters, step_size=step_size, alpha=[1],
                        anchor_idx=cur_anchor_masks, target=target.repeat(len(current_epsilons)), multi_start=True, multi_success=False)
            return success, X_pgd, margin
        robust_ub, X_pgd, _ = binary_search(binary_search_cond, np.full((1, 1, 28, 28), eps_start), cur_anchor, tol=0.001, max_steps=25)
        robust_ub = robust_ub[0]
        print('robustness:', robust_ub)
        fig, name = vis(X, cur_anchor, X_pgd, method, robust_ub, tensorboard=True, plot_curve=True) 
        writer.add_scalar(name+'_robust', robust_ub, plot_step)
        writer.add_figure(name+'_vis', fig, plot_step)
        plot_step += 1


def empty_expl(Xs):
    anchor_maps = np.zeros_like(Xs)
    return anchor_maps


def random_expl(Xs):
    anchor_maps = np.random.rand(*Xs.shape)
    return anchor_maps


def saliency_expl(expl_method, Xs, ys, model):
    """currently only suppor one example at a time"""
    print('Computing explanation by {}...'.format(expl_method))

    assert Xs.shape[0] == 1
    assert ys.shape[0] == 1

    if expl_method == 'SG':
        sg_r = 0.2
        sg_N = 500
        given_expl = 'Grad'
    else:
        sg_r = None
        sg_N = None
        given_expl = None

    anchor_maps, _ = get_explanation_pdt(Xs, model, ys, expl_method, sg_r=sg_r, sg_N=sg_N, given_expl=given_expl, binary_I=False)
    anchor_maps = np.abs(anchor_maps)
    return anchor_maps


def get_explanation_pdt(image, model, label, exp, sg_r=None, sg_N=None, given_expl=None, binary_I=False):
    label = label[0]
    image_v = Variable(image, requires_grad=True)
    model.zero_grad()
    out = model(image_v)
    pdtr = out[:, label]
    pdt = torch.sum(out[:, label])

    if exp == 'Grad':
        pdt.backward()
        grad = image_v.grad
        expl = grad.data.cpu().numpy()
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'LOO':        
        mask = torch.eye(784).view(784, 1, 28, 28).bool()
        image_loo = image.repeat(784, 1, 1, 1)
        image_loo[mask] = 0
        image_loo = Variable(image_loo, requires_grad=True)
        pred = model(image_loo)[:, label]
        expl = pred - pdtr.item()
        expl = expl.data.cpu().numpy()
    elif exp == 'SHAP':
        expl = shap(image.cpu(), label, pdt, model, 20000)
    elif exp == 'IG':
        for i in range(10):
            image_v = Variable(image * i/10, requires_grad=True)
            model.zero_grad()
            out = model(image_v)
            pdt = torch.sum(out[:, label])
            pdt.backward()
            grad = image_v.grad
            if i == 0:
                expl = grad.data.cpu().numpy() / 10
            else:
                expl += grad.data.cpu().numpy() / 10
        if binary_I:
            expl = expl * image.cpu().numpy()
    else:
        raise NotImplementedError('Explanation method not supported.')

    return expl, pdtr
