import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as V
from torch import autograd
import numpy as np


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        #### SELF ARGS ####
        self.dropout = args.dropout
        # Model optimizer
        self.optimizer = None

        #### MODEL PARAMS ####
        self.fc1 = nn.Linear(784, 400)
        self.fc1_drop = nn.Dropout(0.5) if self.dropout else nn.Dropout(0)
        self.fc2 = nn.Linear(400, 400)
        self.fc2_drop = nn.Dropout(0.5) if self.dropout else nn.Dropout(0)
        # self.fc3 = nn.Linear(400, 400)
        # self.fc3_drop = nn.Dropout(0.5) if self.dropout else nn.Dropout(0)
        self.fc_final = nn.Linear(400, 10)

        # Init Matrix which will get Fisher Matrix
        self.Fisher = {}

        # Self Params
        self.params = [param for param in self.parameters()]


    def forward(self, x):
        # Flatten input
        x = x.view(-1, 784)
        # Keep it for dropout

        # FIRST FC
        x_relu = F.relu(self.fc1(x))
        x = self.fc1_drop(x_relu)

        # SECOND FC
        x_relu = F.relu(self.fc2(x))
        x = self.fc2_drop(x_relu)

        # # THIRD FC
        # x_relu = F.relu(self.fc3(x))
        # x = self.fc3_drop(x_relu)

        # Classification
        x = self.fc_final(x)

        return x

    def estimate_fisher(self, dataset, sample_size, batch_size=64):
        # Get loglikelihoods from data
        self.F_accum = []
        for v, _ in enumerate(self.params):
            self.F_accum.append(np.zeros(list(self.params[v].size())))
        data_loader = dataset
        loglikelihoods = []

        for x, y in data_loader:
            #print(x.size(), y.size())
            x = x.view(batch_size, -1)
            x = V(x).cuda() if self._is_on_cuda() else V(x)
            y = V(y).cuda() if self._is_on_cuda() else V(y)

            loglikelihoods.append(F.log_softmax(self(x), dim=1)[range(batch_size), y.data])

            if len(loglikelihoods) >= sample_size // batch_size:
                break

            #loglikelihood = torch.cat(loglikelihoods).mean(0)
            loglikelihood = torch.cat(loglikelihoods).mean(0)
            loglikelihood_grads = autograd.grad(loglikelihood, self.parameters(),retain_graph=True)
            #print("FINISHED GRADING", len(loglikelihood_grads))
            for v in range(len(self.F_accum)):
                #print(len(self.F_accum))
                torch.add(torch.Tensor((self.F_accum[v])), torch.pow(loglikelihood_grads[v], 2).data)

        for v in range(len(self.F_accum)):
            self.F_accum[v] /= sample_size

        parameter_names = [
            n.replace('.', '__') for n, p in self.named_parameters()
        ]
        #print("RETURNING", len(parameter_names))

        return {n: g for n, g in zip(parameter_names, self.F_accum)}

    def consolidate(self, fisher):
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            self.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
            #print(dir(fisher[n].data))
            self.register_buffer('{}_estimated_fisher'
                                 .format(n), fisher[n].data)

    def ewc_loss(self, lamda, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(self, '{}_estimated_mean'.format(n))
                fisher = getattr(self, '{}_estimated_fisher'.format(n))
                # wrap mean and fisher in Vs.
                mean = V(mean)
                fisher = V(fisher.data)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p-mean)**2).sum())
            return (lamda/2)*sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                V(torch.zeros(1)).cuda() if cuda else
                V(torch.zeros(1))
            )

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
