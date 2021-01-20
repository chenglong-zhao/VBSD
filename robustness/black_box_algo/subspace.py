import torch.nn.functional as F
import torch
from torch.autograd import Variable
from tqdm import tqdm


def subspace_attack(model, model_prior, images, labels, alpha=0.001, sigma=1e-3,
                    maxQuery=3000, tau=1.0, delta=0.1, epsilon=0.05, eta_g=0.1, eta=1 / 255):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    success = False
    criterion = F.cross_entropy

    images = images.to(device)

    regmin = images - epsilon
    regmax = images + epsilon
    regmin = regmin.to(device)
    regmax = regmax.to(device)

    x_adv = images.clone().to(device)
    y = labels.to(device)
    g = torch.zeros_like(images).to(device)

    for i in range(maxQuery // 2):
        x_adv.requires_grad_(True)
        model_prior.zero_grad()
        model.zero_grad()
        output = model_prior(x_adv)

        loss = criterion(output, y)
        loss.backward()
        u = x_adv.grad
        # u=torch.randn(u.shape).to(device)

        with torch.no_grad():
            g_plus = g + tau * u
            g_minus = g - tau * u
            g_plus_prime = g_plus / g_plus.norm()
            g_minus_prime = g_minus / g_minus.norm()

            x_plus = x_adv + delta * g_plus_prime
            x_minus = x_adv + delta * g_minus_prime

            query_minus = model(x_minus)
            query_plus = model(x_plus)
            # query_minus = F.softmax(query_minus)
            # query_plus = F.softmax(query_plus)
            # print(query_minus, query_plus)
            delta_t = ((criterion(query_plus, y) - criterion(query_minus, y)) / (tau * epsilon)) * u
            # print(criterion(query_plus, y), criterion(query_minus, y))

            g += eta_g * delta_t

            # print(g.min(), g.max())
            x_adv += eta * torch.sign(g)

            x_adv = torch.max(x_adv, regmin)
            x_adv = torch.min(x_adv, regmax)
            x_adv = torch.clamp(x_adv, 0, 1)

            outputs = model(x_adv)
            _, curpre = torch.max(outputs.data, 1)
            isFinish = not torch.equal(curpre, y)
            # print(curpre, y)
            # isFinish = curpre.item() != y.item()

            if isFinish:
                success = True
                #print('Early stop at epoch %d.' % i)
                break

    l1dist = torch.norm(images - x_adv, p=1)
    l2dist = torch.norm(images - x_adv, p=2)
    l8dist = torch.max(images - x_adv)
    # return x_adv, success, (i+1) * 2, l1dist, l2dist, l8dist
    return x_adv, success, (i + 1) * 2