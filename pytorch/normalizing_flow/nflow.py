import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.distributions import multivariate_normal

# Fix seed for reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)


parser = argparse.ArgumentParser()
# action
parser.add_argument('--train', action='store_true', help='Train a flow model.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate a flow.')
parser.add_argument('--plot', action='store_true', help='Plot a flow and target density.')
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--output_dir', default='.', help='Path to output folder.')
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
# target potential
parser.add_argument('--target_potential', choices=['u_z0', 'u_z5', 'u_z1', 'u_z2', 'u_z3', 'u_z4'], help='Which potential function to approximate.')
# flow params
parser.add_argument('--base_sigma', type=float, default=4, help='Std of the base isotropic 0-mean Gaussian distribution.')
parser.add_argument('--learn_base', default=False, action='store_true', help='Whether to learn a mu-sigma affine transform of the base distribution.')
parser.add_argument('--flow_length', type=int, default=2, help='Length of the flow.')
# training params
parser.add_argument('--init_sigma', type=float, default=1, help='Initialization std for the trainable flow parameters.')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--start_step', type=int, default=0, help='Starting step (if resuming training will be overwrite from filename).')
parser.add_argument('--n_steps', type=int, default=1000000, help='Optimization steps.')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay.')
parser.add_argument('--beta', type=float, default=1, help='Multiplier for the target potential loss.')
parser.add_argument('--seed', type=int, default=2, help='Random seed.')


#
# Planar Flow
#

class PlanarFlow(nn.Module):

    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.h = lambda x: torch.tanh(x)
        self.h_prime = lambda x: 1 - torch.tanh(x) ** 2

        # Initialize the parameters with normal distribution
        # If init_sigma is too small, flow cannot generate the bimodal distribution.
        self.init_sigma = 1
        self.weights = nn.Parameter(torch.randn(1, dim).normal_(0, self.init_sigma))
        self.u = nn.Parameter(torch.randn(1, dim).normal_(0, self.init_sigma))
        self.bias = nn.Parameter(torch.randn(1).fill_(0))


    def forward(self, x):

        z, log_q = x

        # Modification of u to have inverse transformation w^T * u >= -1
        inner_wu = (self.weights @ self.u.t()).squeeze()
        u_hat = self.u + ( (-1 + torch.log1p(inner_wu.exp())) - inner_wu) * (self.weights / (self.weights @ self.weights.t()))

        z = z + u_hat * self.h( z @ self.weights.t() + self.bias )
        psi = self.h_prime (z @ self.weights.t() + self.bias) * self.weights
        det_jacob = torch.abs(1 + u_hat @ psi.t())
        log_q -= torch.log(det_jacob + 1e-7).squeeze()

        return z, log_q



#
# Normalizing flow
#

class NormalizingFlow(nn.Module):

    def __init__(self, dim, K):

        super(NormalizingFlow, self).__init__()

        self.dim = dim
        self.K = K
        self.flows = torch.nn.ModuleList()
        for k in range(K):
            self.flows.append(PlanarFlow(dim))

    def forward(self, z, log_q):
        # Flow calculation
        for flow in self.flows:
            z, log_q = flow((z, log_q))

        return z, log_q


#
# Loss function
#

def calc_loss(z_k, log_q_k, target_density):
    log_p = torch.log(target_density.calc_prob_torch(z_k)+1e-7)
    loss = torch.mean(log_q_k - log_p).squeeze()
    return loss


# Model Training

def train_flow(flow,
    lr_decay = 0.999,
    log_interval = 2000,
    iterations = 20000,
    batch_size = 100, ):

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
    optimizer = torch.optim.RMSprop(nflow.parameters(), lr=1e-5, momentum=0.9, alpha=0.90, eps=1e-6)

    # anneal rate for free energy
    # temp = lambda i: min(1, 0.01 + i/10000)

    running_loss = 0
    for i in range(1, iterations + 1):

        scheduler.step()

        # Sample z_0, log_q_0
        z_0 = standard_normal_2d.sample([batch_size])
        log_q_0 = standard_normal_2d.log_prob(z_0)

        z_k, log_q_k = flow(z_0, log_q_0)
        optimizer.zero_grad()
        loss = calc_loss(z_k, log_q_k, target_density=traget_density)
        loss.backward()
        optimizer.step()

        # print learing process
        running_loss += loss.item()
        if i % log_interval == 0:
            print('[%5d] loss: %.3f' %
                  (i, running_loss / log_interval))
            running_loss = 0.0