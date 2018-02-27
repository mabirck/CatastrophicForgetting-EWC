import os, csv
import torch
from torch.autograd import Variable

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
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot
