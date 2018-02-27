from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from utils import saveLog
from myDataLoader import getDataLoader
# Training settings

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.99, metavar='M',
                    help='SGD momentum (default: 0.99)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dropout', action='store_true', default=False,
                    help='Activates dropout training!')
parser.add_argument('--origin',type=str , default='mnist',
                    help='Kind of training!')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


# Permutations to get different inputs
permutations = [ np.random.permutation(28**2) for k in range(3) ]

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

        x = self.fc_final(x)

        return F.log_softmax(x, dim=1)


def train(model, epoch, train_loader, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target).long()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(model, epoch, test_loader, test_task, args, continuous):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target).long()
        #print(target)
        output = model(data)

        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        #print(correct)

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))
    if args.dropout == True:
        drop_way = "Dropout"
    else:
        drop_way = "None"
    saveLog(test_loss, test_acc, correct, drop_way, args, epoch, test_task, continuous)


def run(train_datasets, test_datasets):

    model = Net(args, args.dropout)

    if torch.cuda.is_available():
        model.cuda()


    for task, train_dataset in enumerate(train_datasets, 1):
        ## EVALUATING MULTIPLE TASKS | task != test_task is to evaluate if is a continual
        ## run or first one ##
        [test(model, 0, test_dataset, test_task, args, task != test_task)
        if test_task <= task else None
        for test_task, test_dataset in enumerate(test_datasets, 1)]

        for epoch in range(1, args.epochs + 1):
            train(model, epoch, train_dataset, args)

            ## EVALUATING MULTIPLE TASKS | task != test_task is to evaluate if is a continual
            ## run or first one ##
            [test(model, epoch, test_dataset, test_task, args, task != test_task)
            if test_task <= task else None
            for test_task, test_dataset in enumerate(test_datasets, 1)]

def main():
    # Getting Datasets Tasks (A, B, C), to propagate through the model
    train_datasets = [
        getDataLoader(args.origin, train=True, permutation=p, args=args) for p in permutations
    ]

    test_datasets = [
        getDataLoader(args.origin, train=False, permutation=p, args=args) for p in permutations
    ]

    args.dropout = True
    print("RUNNING DROPOUT ONE")
    run(train_datasets, test_datasets)

    args.dropout = False
    print("RUNNING NONE ONE")
    run(train_datasets, test_datasets)

if __name__ == "__main__":
    main()
