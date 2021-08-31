import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np


def tv_norm(input, tv_beta):
    img = input[0, 0, :, :]
    row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    return row_grad + col_grad

def bbmp(X, y, model, device):
    print('Computing explanation by BBMP...')
    tv_beta = 3
    learning_rate = 0.1
    max_iterations = 1000
    l1_coeff = 10
    tv_coeff = 0.01

    img = X.cpu().numpy().reshape(28, 28)
    blurred_img = cv2.GaussianBlur(img, (11, 11), 5)

    mask_init = np.ones((28, 28), dtype = np.float32)
    mask = torch.Tensor(mask_init).to(device).view(1, 1, 28, 28)
    mask.requires_grad = True

    optimizer = torch.optim.Adam([mask], lr=learning_rate)
    img = torch.Tensor(img).to(device).view(1, 1, 28, 28)
    blurred_img = torch.Tensor(blurred_img).to(device).view(1, 1, 28, 28)
    category = y.item()

    for i in range(max_iterations):
        perturbated_input = img.mul(mask) + \
                            blurred_img.mul(1-mask)

        noise = torch.randn(1, 1, 28, 28).to(device)
        
        perturbated_input = perturbated_input + noise

        outputs = model(perturbated_input)
        loss = l1_coeff*torch.mean(mask) + \
                tv_coeff*tv_norm(mask, tv_beta) + outputs[0, category]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)
    
    return mask.data.cpu().numpy()