from __future__ import print_function
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
sys.path.append('..')
from models import Feat2_ResNet50
from utils import Logger
from loss import Proximity, Con_Proximity


# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR model Train')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--lr_model', type=float, default=0.01, help="learning rate for CE Loss")
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_prox', type=float, default=0.5, help="learning rate for Proximity Loss")  # as per paper
parser.add_argument('--weight-prox', type=float, default=1, help="weight for Proximity Loss")  # as per paper
parser.add_argument('--lr_conprox', type=float, default=0.0001,help="learning rate for Con-Proximity Loss")  # as per paper
parser.add_argument('--weight-conprox', type=float, default=0.0001,help="weight for Con-Proximity Loss")
parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=30, metavar='N',help='how many batches to wait before logging training status')
parser.add_argument('--save-dir', type=str, default='log')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args.cuda = not args.no_cuda and torch.cuda.is_available()

sys.stdout = Logger(os.path.join(args.save_dir, 'resnet50_pcl.txt'))

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

num_classes = 10
use_gpu = args.cuda
model = Feat2_ResNet50()
if args.cuda:
    model.cuda()
filename = '../checkpoint/resnet50_vallina.pth'
checkpoint = torch.load(filename)
model.load_state_dict(checkpoint)
criterion_xent = nn.CrossEntropyLoss()
criterion_prox_512 = Proximity(num_classes=num_classes, feat_dim=2048, use_gpu=use_gpu)
criterion_prox_256 = Proximity(num_classes=num_classes, feat_dim=1024, use_gpu=use_gpu)

criterion_conprox_512 = Con_Proximity(num_classes=num_classes, feat_dim=2048, use_gpu=use_gpu)
criterion_conprox_256 = Con_Proximity(num_classes=num_classes, feat_dim=1024, use_gpu=use_gpu)

optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=args.weight_decay, momentum=args.momentum)

optimizer_prox_512 = torch.optim.SGD(criterion_prox_512.parameters(), lr=args.lr_prox)
optimizer_prox_256 = torch.optim.SGD(criterion_prox_256.parameters(), lr=args.lr_prox)

optimizer_conprox_512 = torch.optim.SGD(criterion_conprox_512.parameters(), lr=args.lr_conprox)
optimizer_conprox_256 = torch.optim.SGD(criterion_conprox_256.parameters(), lr=args.lr_conprox)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        feats1024, feats2048, outputs = model(data)
        loss_xent = criterion_xent(outputs, target)

        loss_prox_512 = criterion_prox_512(feats2048, target)
        loss_prox_256 = criterion_prox_256(feats1024, target)

        loss_conprox_512 = criterion_conprox_512(feats2048, target)
        loss_conprox_256 = criterion_conprox_256(feats1024, target)

        loss_prox_512 *= args.weight_prox
        loss_prox_256 *= args.weight_prox

        loss_conprox_512 *= args.weight_conprox
        loss_conprox_256 *= args.weight_conprox

        loss = loss_xent + loss_prox_512 + loss_prox_256 - loss_conprox_512 - loss_conprox_256  # total loss
        optimizer_model.zero_grad()

        optimizer_prox_512.zero_grad()
        optimizer_prox_256.zero_grad()

        optimizer_conprox_512.zero_grad()
        optimizer_conprox_256.zero_grad()

        loss.backward()
        optimizer_model.step()

        for param in criterion_prox_512.parameters():
            param.grad.data *= (1. / args.weight_prox)
        optimizer_prox_512.step()

        for param in criterion_prox_256.parameters():
            param.grad.data *= (1. / args.weight_prox)
        optimizer_prox_256.step()

        for param in criterion_conprox_512.parameters():
            param.grad.data *= (1. / args.weight_conprox)
        optimizer_conprox_512.step()

        for param in criterion_conprox_256.parameters():
            param.grad.data *= (1. / args.weight_conprox)
        optimizer_conprox_256.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tLoss_xce: {:.6f}\tLoss_prox256: {:.6f}'
                  '\tLoss_prox512: {:.6f}\tLoss_conprox256: {:.6f}\tLoss_conprox512: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item(),loss_xent.item(),loss_prox_256.item(),loss_prox_512.item(),
                       loss_conprox_256.item(),loss_conprox_512.item()))

    torch.save(model.state_dict(), '../checkpoint/resnet50_pcl.pth')


def inference():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            _, _, output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * float(correct) / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
        for param_group in optimizer_model.param_groups:
            param_group['lr'] *= 0.1
        for param_group in optimizer_prox_256.param_groups:
            param_group['lr'] *= 0.1
        for param_group in optimizer_conprox_256.param_groups:
            param_group['lr'] *= 0.1
        for param_group in optimizer_prox_512.param_groups:
            param_group['lr'] *= 0.1
        for param_group in optimizer_conprox_512.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    inference()