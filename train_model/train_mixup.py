from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
sys.path.append('..')
from models import *
from utils import Logger
from random import shuffle


# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR model Train')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float,metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=30, metavar='N',help='how many batches to wait before logging training status')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--resume', choices=['vallina', 'finetune'], default='vallina')
args = parser.parse_args()

def label_smoothing(one_hot, factor):
    return one_hot * factor + (one_hot - 1.) * ((factor-1) / float(10 - 1))

def attack(model, img, label, eps=8/255, iters=10, step=2/255):
    adv = img.clone().detach()
    adv.requires_grad = True

    for j in range(iters):
        out_adv = model(adv.clone())
        loss = F.cross_entropy(out_adv, label)
        loss.backward()

        noise = adv.grad
        adv.data = adv.data + step * noise.sign()

        adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
        adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()

    x = adv
    x_nat = img  # 256,3,32,32
    y = torch.eye(10)[label].cuda()  # 256,10
    perturb = (x - x_nat) * 2.0
    adversarial_vertex = x_nat + perturb
    adversarial_vertex.clamp_(0.0, 1.0)    # 256,3,32,32
    y_nat = label_smoothing(y, 1.0)   # 256,10 one-hot label
    y_vertex = label_smoothing(y, 0.1)   # 256,10 smooth label
    x_weight = np.random.beta(1.0, 1.0, [x.shape[0], 1, 1, 1])  # 256,1,1,1
    y_weight = np.reshape(x_weight, [-1, 1])     # 256,1
    x_weight = torch.from_numpy(x_weight).float().cuda()
    y_weight = torch.from_numpy(y_weight).float().cuda()
    x = x_nat * x_weight + adversarial_vertex * (1 - x_weight)  # 256,3,32,32
    y = y_nat * y_weight + y_vertex * (1 - y_weight)    # 256,10

    return [x, y]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args.cuda = not args.no_cuda and torch.cuda.is_available()

sys.stdout = Logger(os.path.join(args.save_dir, 'resnet50_mixup' + '.txt'))

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.Pad(4),
                         transforms.RandomCrop(32),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                     ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.Compose([transforms.ToTensor(),])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


model = ResNet50()
if args.cuda:
    model.cuda()
if args.resume=='finetune':
    print('Load checkpoint...')
    filename = '../checkpoint/resnet50_vallina.pth'
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)

def SoftCrossEntropy(inputs, target, reduction='average'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    elif reduction == 'sum':
        loss = torch.sum(torch.mul(log_likelihood, target))
    else:
        raise NotImplementedError
    return loss

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        model.eval()
        x_batch_adv, y_batch_adv = attack(model, data, target)
        data = torch.cat((data, x_batch_adv), 0)
        target = torch.eye(10)[target].cuda()
        target = torch.cat((target, y_batch_adv), 0)

        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = SoftCrossEntropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    torch.save(model.state_dict(), '../checkpoint/resnet50_mixup.pth')


def inference():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * float(correct) / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    if epoch in [args.epochs * 0.5, args.epochs * 0.75, args.epochs * 0.875]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    inference()