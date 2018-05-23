from __future__ import print_function
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import get_datasets
from utils import test_it
from model import Net
from arguments import get_args
# Training settings


def train(model, epoch, train_loader, optimizer=None, args=None):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target).long()
        optimizer.zero_grad()
        output = model(data)

        objective_loss = F.cross_entropy(output, target)

        # Manual
        ewc_loss = 0
        if(args.ewc and not(args.dropout)):
            ewc_loss = model.ewc_loss(1, cuda=torch.cuda.is_available())
        loss = objective_loss + ewc_loss

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def run(train_datasets, test_datasets, args=None):
    # Number of samples used for estimating fisher #
    fisher_estimation_sample_size = 1024

    # Define Model
    model = Net(args)
    if torch.cuda.is_available():
        model.cuda()

    # Global optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-05)

    for task, train_dataset in enumerate(train_datasets):
        ''' Evaluate Current Net '''
        test_it(model, 0, test_datasets, args, task)

        for epoch in range(1, args.epochs + 1):
            train(model, epoch, train_dataset, optimizer, args)

            ''' Evaluate Current Net '''
            test_it(model, epoch, test_datasets, args, task)

        if args.ewc:
            # Get fisher inf of parameters and consolidate it in the net
            model.consolidate(model.estimate_fisher(
                train_dataset, fisher_estimation_sample_size))


def main():
    args = get_args()

    '''
        Set fixed seed
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Permutations to get different inputs
    permutations = [np.random.permutation(28**2) for k in range(3)]

    # Getting Datasets Tasks (A, B, C), to propagate through the model
    train_datasets, test_datasets = get_datasets(permutations, args)

    args.ewc = True
    args.dropout = False
    print("RUNNING EWC ONE")
    run(train_datasets, test_datasets, args)


    args.ewc = False
    args.dropout = True
    print("RUNNING DROPOUT ONE")
    run(train_datasets, test_datasets)

    args.dropout = False
    print("RUNNING NONE ONE")
    run(train_datasets, test_datasets)


if __name__ == "__main__":
    main()
