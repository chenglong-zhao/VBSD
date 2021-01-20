from __future__ import print_function
import argparse
import os
import torch
import sys
sys.path.append('..')
from models import ResNet50, ResNet34
from black_box_algo import nes, nattack, simba_single, subspace_attack
from data_loader import clean_loader_cifar, adv_loader_data
from robust_inference import robust_inference

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Attack')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--data-dir', type=str, default='data', metavar='N',help='')
parser.add_argument('--model-checkpoint', type=str, default='../checkpoint/resnet50_vallina.pth', metavar='N')
parser.add_argument('--gpu-id', type=str, default='1')
args = parser.parse_args()

print("Evaluating...:", args.model_checkpoint)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

def craft_adv_samples(data_loader, model, args, attack_method):
    adv_samples = []
    target_tensor = []
    succ_list = []
    q_list = []
    model.eval()
    for bi, batch in enumerate(data_loader):
        if bi==155:
            break
        inputs, targets = batch
        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        if attack_method == 'nes':
            x_adv, succ, q = nes(model, inputs, targets, targeted=False, alpha=0.007, sigma=3e-3, eps=0.03,
                        sample_batch_size=16, maxIter=125)
        elif attack_method == 'nattack':
            x_adv, succ, q = nattack(model, inputs, targets, targeted=False, npop=16, sigma=0.1, alpha=0.007,
                        epsi=0.03, n_cls=10, n_channel=3)
        elif attack_method == 'simba':
            x_adv, succ, q = simba_single(model, inputs, targets, num_iters=2000, epsilon=0.03)
        elif attack_method == 'subspace':
            model_p = ResNet34()
            model_p.cuda()
            model_p.load_state_dict(torch.load('../checkpoint/resnet34_vallina.pth'))
            model_p.eval()
            x_adv, succ, q = subspace_attack(model, model_p, inputs, targets, alpha=0.001, sigma=1e-3,
                        maxQuery=2000, tau=1.0, delta=0.1, epsilon=0.03, eta_g=0.1, eta=0.007)
        else:
            raise NotImplementedError
        adv_samples.append(x_adv)
        target_tensor.append(targets)
        succ_list.append(succ)
        q_list.append(q)

    return torch.cat(adv_samples, 0), torch.cat(target_tensor, 0), succ_list


def main():

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    clean_loader = clean_loader_cifar(args)
    model = ResNet50()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        model.cuda()
    model.load_state_dict(torch.load(args.model_checkpoint))

    robust_inference(model, clean_loader, args, note='natural')

    for attack_method in ['nes','nattack','simba','subspace']:
        adv_samples, targets, succ_list = craft_adv_samples(clean_loader, model, args, attack_method)
        print('Attack success:', sum(succ_list)/len(succ_list))
        if args.cuda:
            adv_samples = adv_samples.cpu()
            targets = targets.cpu()
        adv_loader = adv_loader_data(args, adv_samples, targets)

        robust_inference(model, adv_loader, args, note=attack_method)

if __name__ == '__main__':
    main()
