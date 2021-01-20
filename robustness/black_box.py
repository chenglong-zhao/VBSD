from __future__ import print_function
import argparse
import os
import torch
import sys
sys.path.append('..')
#from models.vgg import VGG
from models import ResNet50
from adv_attack import fgsm, pgd, mim, cw
from data_loader import clean_loader_cifar, adv_loader_data
from robust_inference import robust_inference


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Attack')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--data-dir', type=str, default='data', metavar='N',help='')
parser.add_argument('--checkpoint-dir', type=str, default='../checkpoint', metavar='N')
parser.add_argument('--eps', type=float, default=0.03)
parser.add_argument('--eps-fgsm', type=float, default=0.03)
parser.add_argument('--gpu-id', type=str, default='0')
args = parser.parse_args()

print("Evaluating...:", args.checkpoint_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

def craft_adv_samples(data_loader, model, args, attack_method):
    adv_samples = []
    target_tensor = []
    L2_list = []
    model.eval()
    for bi, batch in enumerate(data_loader):
        inputs, targets = batch
        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        if attack_method == 'fgsm':
            crafted, l2 = fgsm(inputs, targets, model, eps=args.eps_fgsm)
        elif attack_method == 'pgd10':
            crafted, l2 = pgd(inputs, targets, model, eps=args.eps, iters=10)
        elif attack_method == 'pgd20':
            crafted, l2 = pgd(inputs, targets, model, eps=args.eps, iters=20)
        elif attack_method == 'pgd50':
            crafted, l2 = pgd(inputs, targets, model, eps=args.eps, iters=50)
        elif attack_method == 'mim':
            crafted, l2 = mim(inputs, targets, model, eps=args.eps)
        elif attack_method == 'cw20':
            crafted, l2 = cw(inputs, targets, model)
        else:
            raise NotImplementedError
        adv_samples.append(crafted)
        target_tensor.append(targets)
        L2_list.append(l2)

    return torch.cat(adv_samples, 0), torch.cat(target_tensor, 0), sum(L2_list)/len(L2_list)


def main():

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    clean_loader = clean_loader_cifar(args)

    model_attack = ResNet50()
    model_target = ResNet50()

    model_list = [os.path.join(args.checkpoint_dir, m) for m in os.listdir(args.checkpoint_dir)]

    for ckpt_attack in model_list:
        for ckpt_target in model_list:
            if ckpt_attack == ckpt_target:
                continue

            print('attack model:', ckpt_attack)
            print('target model:', ckpt_target)

            if args.cuda:
                torch.cuda.manual_seed(args.seed)
                model_attack.cuda()
                model_target.cuda()
            model_attack.load_state_dict(torch.load(ckpt_attack))
            model_target.load_state_dict(torch.load(ckpt_target))

            robust_inference(model_attack, clean_loader, args, note='natural attack')
            robust_inference(model_target, clean_loader, args, note='natural target')

            for attack_method in ['fgsm','pgd10']:
                adv_samples, targets, l2_mean = craft_adv_samples(clean_loader, model_attack, args, attack_method)
                if args.cuda:
                    adv_samples = adv_samples.cpu()
                    targets = targets.cpu()
                adv_loader = adv_loader_data(args, adv_samples, targets)

                robust_inference(model_target, adv_loader, args, note='black-box:'+attack_method)

if __name__ == '__main__':
    main()
