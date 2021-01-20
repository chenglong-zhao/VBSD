import torch
import torch.nn.functional as F
from torch.autograd import Variable

def where(cond, x, y):
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

def nes(model, images, labels, targeted=False, alpha=0.007, sigma=3e-3, eps=0.03,
               sample_batch_size=16,maxIter=625):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_lr_schedule = False
    cost_cache_length = 20
    lr_drop = 2
    momentum = 0.9
    min_lr = 1e-7

    if use_lr_schedule:
        last_reduce_lr = 0
        cost_cache = []

    images_ori = images.clone().squeeze(0).to(device)
    labels_ori = labels.clone().to(device)

    images = images.to(device)
    labels = labels.to(device)
    labels = labels.repeat(sample_batch_size)  # 32,

    prev_grad = None
    cur_grad = None

    success = False

    iters=0
    for i in range(maxIter):
        iters+=1
        # gradient estimation
        images = images.squeeze(0)  # 3,299,299

        noise_pos = torch.randn((sample_batch_size // 2,) + images.shape).to(device)
        noise = torch.cat((noise_pos, -noise_pos), 0)

        eval_points = images + sigma * noise  # 32,3,299,299

        logits = model(eval_points)  # 32,1000
        losses = torch.nn.CrossEntropyLoss(reduction='none')(logits, labels)  # 32,
        losses_tiled = losses.reshape(-1, 1, 1, 1).repeat(1, *images.shape)  # 32,3,299,299
        grad_estimate = torch.mean(losses_tiled * noise, 0) / sigma  # 3,299,299

        outputs = model(images.unsqueeze(0))
        _, curpre = torch.max(outputs.data, 1)

        # finish iteration condition
        if targeted:
            isFinish = torch.equal(curpre, labels_ori)
        else:
            isFinish = not torch.equal(curpre, labels_ori)
        if isFinish:
            success = True
            #print('Early stop at epoch %d.' % i)
            break

        if i % 20 == 0:
            pass
            #print('epoch %d loss: %f' % (i, losses.mean()))

        # lr_schedule
        if use_lr_schedule:
            if targeted:
                cost_cache.append(losses.mean().detach())
            else:
                cost_cache.append(-losses.mean().detach())
            cost_cache = cost_cache[-cost_cache_length:]
            last_reduce_lr += 1
            if cost_cache[-1] > cost_cache[0] and len(cost_cache) == cost_cache_length and last_reduce_lr > 10:
                if alpha > min_lr:
                    # print('[log] Annealing learning rate.')
                    alpha = max(alpha / lr_drop, min_lr)
                    last_reduce_lr = 0

        if not targeted:
            grad_estimate = -grad_estimate

        # momentum
        prev_grad = grad_estimate.detach() if cur_grad is None else cur_grad
        cur_grad = grad_estimate.detach()
        cur_grad = momentum * prev_grad + (1 - momentum) * cur_grad

        images = images - alpha * cur_grad.sign()

        images = where(images > images_ori + eps, images_ori + eps, images)
        images = where(images < images_ori - eps, images_ori - eps, images)
        images = torch.clamp(images, 0, 1)

    l1dist = torch.norm(images_ori - images, p=1)
    l2dist = torch.norm(images_ori - images, p=2)
    l8dist = torch.max(images_ori - images)

    return images.unsqueeze(0).detach(), success, iters * sample_batch_size
