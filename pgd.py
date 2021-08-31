import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as F
import numpy as np


def sample_eps_L2(num_dim, epsilon, N, anchor_masks, device):
    if num_dim == 0:
        return torch.zeros_like(anchor_masks).float()
    normal_deviates = np.random.normal(size=(N, num_dim))
    total_dim = normal_deviates.shape[1]
    radius = np.linalg.norm(normal_deviates, axis=1, keepdims=True)
    normal_deviates = normal_deviates * (np.random.rand(N, total_dim) ** (1.0 / total_dim))
    points = normal_deviates * (epsilon) / radius
    samples = torch.zeros_like(anchor_masks).float()
    samples[~anchor_masks] = torch.FloatTensor(points).to(device).view(-1)
    return samples


def pgd(Xs, ys, model, epsilons, norm, niters, step_size, anchor_masks, targeted, targets=None,
        box_min=0., box_max=1., verbose=False, multi_start=False, device=None): 
    if multi_start: # currently only support only example at a time
        assert Xs.shape == (1, 1, 28, 28)
        assert ys.shape == (1,)
        assert epsilons.shape == (1,)
        num_start = 1000
        anchor_masks = anchor_masks.repeat(num_start, 1, 1, 1)
        epsilons = epsilons.repeat(num_start)
        num_free_dim = 28*28 - anchor_masks[0].sum()
        if norm == 2:
            samples = sample_eps_L2(num_free_dim, epsilons[0].item(), num_start, anchor_masks, device)
            Xs_pgd = Variable(Xs.data + samples, requires_grad=True)
            Xs_pgd.data = torch.clamp(Xs_pgd.data, 0.0, 1.0)
        else:
            raise ValueError('norm not supported')

        if targeted:
            targets = targets.repeat(num_start).view(-1, 1)
        else:
            ys = ys.repeat(num_start).view(-1, 1)
    else:
        Xs_pgd = Variable(Xs.data, requires_grad=True)

        if targeted:
            targets = targets.view(-1, 1)
        else:
            ys = ys.view(-1, 1)


    for i in range(niters):
        outputs = model(Xs_pgd)
        if targeted:
            loss = (torch.gather(outputs, 1, targets)).sum() 
        else:
            loss = -((torch.gather(outputs, 1, ys)).sum())
    
        loss.backward()
    
        cur_grad = Xs_pgd.grad.data
        
        eta = cur_grad
        eta[anchor_masks] = 0. 
        zero_mask = eta.view(eta.shape[0], -1).norm(norm, dim=-1) == 0.
        eta = (eta * step_size /
               eta.view(eta.shape[0], -1).norm(norm, dim=-1).view(-1, 1, 1, 1))
        eta[zero_mask] = 0.

        Xs_pgd = Variable(Xs_pgd.data + eta, requires_grad=True)
        Xs_pgd.data = torch.clamp(Xs_pgd.data, 0.0, 1.0) 
    
        eta = Xs_pgd.data - Xs.data
        
        mask = eta.view(eta.shape[0], -1).norm(norm, dim=1) <= epsilons
        scaling_factor = eta.view(eta.shape[0], -1).norm(norm, dim=1)
        scaling_factor[mask] = epsilons[mask]
        eta *= (epsilons / scaling_factor).view(-1, 1, 1, 1)

        Xs_pgd.data = Xs.data + eta

    ys_pred_pgd = model(Xs_pgd)
    if targeted:
        successes = ys_pred_pgd.data.max(1)[1] == targets.data.view(-1)
    else:
        successes = ys_pred_pgd.data.max(1)[1] != ys.data.view(-1)
    
    # print('attack success on {} of {}'.format(successes.sum().item(), len(Xs_pgd)))

    if multi_start:
        success = successes.any(dim=0, keepdim=True)
        success_Xs_pgd = Xs_pgd.data[successes][[0]] if success.item() else Xs_pgd.data[[0]]
        return success, success_Xs_pgd
    else:
        return successes, Xs_pgd.data
    
            
    
    