import torch
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import cv2


def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (np.log((1 + x) / (1 - x))) * 0.5


def nattack(model, images, labels, targeted=True, npop=32, sigma=0.1, alpha=0.02, epsi=0.05, n_cls=10, n_channel=3):
    # npop = 200
    # sigma = 0.1
    # alpha = 0.008
    assert len(images.shape) == 4

    boxmin = 0
    boxmax = 1
    boxplus = (boxmin + boxmax) / 2.
    boxmul = (boxmax - boxmin) / 2.
    # epsi = 0.031
    epsilon = 1e-30

    success = False

    inputs = images.cpu().numpy()
    targets = labels.cpu().numpy()

    modify = np.random.randn(1, n_channel, 32, 32) * 0.001

    for runstep in range(125):
        Nsample = np.random.randn(npop, n_channel, 32, 32)
        modify_try = modify.repeat(npop, 0) + sigma * Nsample

        temp = []
        for x in modify_try:
            # temp.append(cv2.resize(x.transpose(1, 2, 0), dsize=(224, 224), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1))
            temp.append(x)
        modify_try = np.array(temp)

        newimg = torch_arctanh((inputs - boxplus) / boxmul)
        inputimg = np.tanh(newimg + modify_try) * boxmul + boxplus  # 200,3,32,32
        if runstep % 2 == 0:

            temp = []
            for x in modify:
                # temp.append(cv2.resize(x.transpose(1, 2, 0), dsize=(224, 224), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1))
                temp.append(x)
            modify_test = np.array(temp)

            realinputimg = np.tanh(newimg + modify_test) * boxmul + boxplus
            realdist = realinputimg - (np.tanh(newimg) * boxmul + boxplus)
            realclipdist = np.clip(realdist, -epsi, epsi)
            realclipinput = realclipdist + (np.tanh(newimg) * boxmul + boxplus)
            l2real = np.sum((realclipinput - (np.tanh(newimg) * boxmul + boxplus)) ** 2) ** 0.5
            # print('epoch:' + str(runstep) + '  l2real: ' + str(l2real.max()))

            realclipinput = np.asarray(realclipinput, dtype='float32')
            input_var = autograd.Variable(torch.from_numpy(realclipinput).cuda(), volatile=True)  # 1,3,32,32

            y = model(input_var)
            outputsreal = F.softmax(y)
            outputsreal = outputsreal.data.cpu().numpy()

            if targeted:
                if (np.argmax(outputsreal) == targets) and (np.abs(realclipdist).max() <= epsi):
                    success = True
                    break
            else:
                if (np.argmax(outputsreal) != targets) and (np.abs(realclipdist).max() <= epsi):
                    success = True
                    break

        dist = inputimg - (np.tanh(newimg) * boxmul + boxplus)
        clipdist = np.clip(dist, -epsi, epsi)
        clipinput = (clipdist + (np.tanh(newimg) * boxmul + boxplus)).reshape(npop, n_channel, 32, 32)
        target_onehot = np.zeros((1, n_cls))
        target_onehot[0][targets] = 1.

        clipinput = np.asarray(clipinput, dtype='float32')
        clipinput = autograd.Variable(torch.from_numpy(clipinput).cuda(), volatile=True)  # 200,3,32,32
        y = model(clipinput)
        outputs = F.softmax(y)
        outputs = outputs.data.cpu().numpy()

        target_onehot = target_onehot.repeat(npop, 0)

        real = np.log((target_onehot * outputs).sum(1) + 1e-30)
        other = np.log(((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0] + 1e-30)
        if targeted:
            loss1 = np.clip(other - real, 0., 1000)
        else:
            loss1 = np.clip(real - other, 0., 1000)
        Reward = 0.5 * loss1
        if ~targeted:
            Reward = -Reward
        A = (Reward - np.mean(Reward)) / (np.std(Reward) + 1e-7)  # 200,
        modify = modify + (alpha / (npop * sigma)) * (
            (np.dot(Nsample.reshape(npop, -1).T, A)).reshape(n_channel, 32, 32))

    images = images.cuda()

    l1dist = torch.norm(input_var - images, p=1)
    l2dist = torch.norm(input_var - images, p=2)
    l8dist = torch.max(input_var - images)
    # print(l8dist)
    # return input_var.detach(), success, runstep * npop, l1dist.detach(), l2dist.detach(), l8dist.detach()

    return input_var.detach(), success, runstep * npop