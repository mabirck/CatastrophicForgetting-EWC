import os
import csv
import torch
import torch.optim as optim
from torch.autograd import Variable as V
import torch.functional as F
from myDataLoader import getDataLoader

def saveLog(test_loss, test_acc, correct, dropout, args, epoch, test_task, continual=False):
    print(test_task, continual)
    path = './log/'
    #path += "_".join([args.arc, str(args.epochs), args.filter_reg, str(args.phi), 'seed', str(args.seed), 'depth', str(args.depth), args.intra_extra])
    path+= dropout+'_MNIST_TASK_'+str(test_task)+'_'+str(args.seed)
    path = path+'.csv'
    if epoch == 0 and os.path.isfile(path) and not continual: os.remove(path)
    assert not(os.path.isfile(path) == True and epoch ==0 and not continual), "That can't be right. This file should not be here!!!!"
    fields = ['epoch', epoch, 'test_loss', test_loss, 'test_acc', test_acc, 'correct', correct]
    with open(path, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or V) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, V) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return V(y_one_hot) if isinstance(y, V) else y_one_hot

def test(model, epoch, test_loader, test_task, args, continuous):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = V(data, volatile=True), V(target).long()
        #print(target)
        output = model(data)

        test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
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
    elif args.ewc == True:
        drop_way = "EWC"
    else:
        drop_way = "None"
    saveLog(test_loss, test_acc, correct, drop_way, args, epoch, test_task, continuous)


def test_it(model, epoch, test_datasets, args, task):
    # EVALUATING MULTIPLE TASKS #
    # task != test_task is to evaluate if is a continual #
    # run or first one #
    [
        test(model, epoch, test_dataset, test_task, args, task != test_task)
            if test_task <= task else None
                for test_task, test_dataset in enumerate(test_datasets, 1)
    ]

def get_datasets(permutations, args):
    train_datasets = [
        getDataLoader(train=True, permutation=p, args=args) for p in permutations
    ]

    test_datasets = [
        getDataLoader(train=False, permutation=p, args=args) for p in permutations
    ]

    return train_datasets, test_datasets
