import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Variable
from torch import autograd
from copy import deepcopy


class Net(nn.Module):
    def __init__(self, args, dropout):
        super(Net, self).__init__()
        #### SELF ARGS ####
        self.dropout = dropout

        #### MODEL PARAMS ####
        self.fc1 = nn.Linear(784, 400)
        self.fc1_drop = nn.Dropout(0.5) if dropout else nn.Dropout(0)
        self.fc2 = nn.Linear(400, 400)
        self.fc2_drop = nn.Dropout(0.5) if dropout else nn.Dropout(0)
        self.fc3 = nn.Linear(400, 400)
        self.fc3_drop = nn.Dropout(0.5) if dropout else nn.Dropout(0)
        self.fc_final = nn.Linear(400, 10)

    def forward(self, x):
        # Flatten input
        x = x.view(-1, 784)
        # Keep it for dropout

        #FIRST FC
        previous = x
        x_relu = F.relu(self.fc1(x))

        x = self.fc1_drop(x_relu)

        #SECOND FC
        previous = x
        x_relu = F.relu(self.fc2(x))

        x = self.fc2_drop(x_relu)

        #THIRD FC
        previous = x
        x_relu = F.relu(self.fc3(x))

        x = self.fc3_drop(x_relu)

        x = self.fc_final(x)

        return x

    def estimate_fisher(self, dataset, sample_size, batch_size=64):
        # Get loglikelihoods from data
        data_loader = dataset
        loglikelihoods = []

        for x, y in data_loader:
            x = x.view(batch_size, -1)
            x = Variable(x).cuda() if self._is_on_cuda() else Variable(x)
            y = Variable(y).cuda() if self._is_on_cuda() else Variable(y)

            loglikelihoods.append(F.log_softmax(self(x), dim=1)[range(batch_size), y.data])

            if len(loglikelihoods) >= sample_size // batch_size:
                break

        loglikelihood = torch.cat(loglikelihoods).mean(0)
        loglikelihood_grads = autograd.grad(loglikelihood, self.parameters())

        parameter_names = [
            n.replace('.', '__') for n, p in self.named_parameters()
        ]

        return {n: g**2 for n, g in zip(parameter_names, loglikelihood_grads)}

    def consolidate(self, fisher):
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            self.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
            self.register_buffer('{}_estimated_fisher'
                                 .format(n), fisher[n].data.clone())

    def ewc_loss(self, lamda, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(self, '{}_estimated_mean'.format(n))
                fisher = getattr(self, '{}_estimated_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p-mean)**2).sum())
            return (lamda/2)*sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
