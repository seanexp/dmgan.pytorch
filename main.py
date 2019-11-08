from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.utils as vutils

from torch.utils.tensorboard import SummaryWriter

import numpy as np

from models import (
        MNISTGenerator, MNISTDiscriminator,
        ChairsGenerator, ChairsDiscriminator,
        )


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['mnist', 'chairs'], help='which data to use')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
# parser.add_argument('--netG', default='', help="path to netG (to continue training)")
# parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--load_from', default='', help='pth file name of netG and netD (to continue training)')
parser.add_argument('--dis_step', type=int, default=5, help='number of dis step per one gen step')
parser.add_argument('--lambda', dest='lambda_q', type=float, default=1., help='coefficient for variational mutual information')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--log_every', type=int, default=10, help='image save interval')
parser.add_argument('--save_every', type=int, default=50, help='model save interval')

args = parser.parse_args()
print(args)

from datetime import datetime
timestamp = datetime.now().strftime('%b%d_%H-%M-%S')

os.makedirs(os.path.join(args.outf, args.dataset, timestamp, 'images'), exist_ok=True)
os.makedirs(os.path.join(args.outf, args.dataset, 'models'), exist_ok=True)

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

if args.load_from != '':
    netG_fname = 'netG_' + args.load_from + '.pth'
    netD_fname = 'netD_' + args.load_from + '.pth'
    netG_path = os.path.join(args.outf, args.dataset, 'models', netG_fname)
    netD_path = os.path.join(args.outf, args.dataset, 'models', netD_fname)

if args.dataset == 'mnist':
    dataset = dset.MNIST(
                    root='datasets/mnist', download=True,
                    transform=T.Compose([
                           T.ToTensor(),
                           T.Normalize((0.5,), (0.5,)),
                       ])
                   )
    nz = 8
    n_gen = 10
    n_epoch = 25 * args.dis_step

    netG = MNISTGenerator(nz, n_gen).to(device)
    netD = MNISTDiscriminator(n_gen).to(device)

elif args.dataset == 'chairs':

    container = np.load('/home/sjjung/data/chairs/chairs_half_64.npz')
    img_data = container['img']
    label_data = container['label']

    transform = T.Compose([
                        T.ToTensor(),
                        T.Normalize((0.5,), (0.5,)),
                        ])

    img_tensor = torch.stack([transform(i) for i in img_data])
    label_tensor = torch.Tensor(label_data)

    dataset = torch.utils.data.TensorDataset(img_tensor, label_tensor)

    nz = 10
    n_gen = 20
    n_epoch = 1000 * args.dis_step

    netG = ChairsGenerator(nz, n_gen).to(device)
    netD = ChairsDiscriminator(n_gen).to(device)

else:
    raise NotImplementedError

if args.load_from != '':
    netG.load_state_dict(torch.load(netG_path))
    netD.load_state_dict(torch.load(netD_path))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=int(args.workers), pin_memory=args.cuda)


data_per_gen = args.batch_size // n_gen
fixed_noise = torch.randn(n_gen, nz, device=device).repeat(1, data_per_gen).view(-1, nz)
fixed_gidx = torch.arange(n_gen).repeat(data_per_gen)

log_dir = os.path.join(
    'runs', f"{timestamp}_{args.dataset}_lambda={args.lambda_q}")

writer = SummaryWriter(log_dir)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam([
                        {'params': netG.parameters()},
                        {'params': netD.latent.parameters()},
                        {'params': netD.posterior.parameters()},
                        ], lr=args.lr, betas=(args.beta1, 0.999))

i_step = 0
for i_epoch in range(1, n_epoch+1):
    for i, (real, _) in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize E_{x ~ p_x} [D(x)] - E_{z ~ p_z} [D(G(z))]
        ###########################
        batch_size = real.size(0)

        noise = torch.randn(batch_size, nz, device=device)
        g_idx = torch.multinomial(torch.ones(n_gen), batch_size, replacement=True)
        fake = netG(noise, g_idx)

        # train with real
        netD.zero_grad()

        real = real.to(device)
        output, _ = netD(real)
        real_score = output.mean()

        # train with fake
        output, _ = netD(fake.detach())
        fake_score = output.mean()
        errD = - real_score + fake_score
        errD.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize E_{z ~ p_z} [D(G(z))]
        ###########################
        if i_step % args.dis_step == 0:
            optimizerG.zero_grad()
            output, posterior = netD(fake)
            fake_score = output.mean()

            errQ = F.cross_entropy(posterior, g_idx.to(device))

            errG = - fake_score + args.lambda_q * errQ
            errG.backward()
            optimizerG.step()

        writer.add_scalar('dis_loss', errD, i_step)
        writer.add_scalar('Q loss', errQ, i_step)
        i_step += 1

    if i_epoch % args.log_every == 0:
        vutils.save_image(real,
                f'{args.outf}/{args.dataset}/{timestamp}/images/real_samples.png', nrow=n_gen,
                normalize=True)
        fake = netG(fixed_noise, fixed_gidx)
        vutils.save_image(fake.detach(),
                f'{args.outf}/{args.dataset}/{timestamp}/images/fake_samples_epoch_{i_epoch}.png', nrow=n_gen,
                normalize=True)

    if i_epoch % args.save_every == 0 or i_epoch == n_epoch:
        # do checkpointing
        torch.save(netG.state_dict(), f'{args.outf}/{args.dataset}/models/netG_{timestamp}_epoch_{i_epoch}.pth')
        torch.save(netD.state_dict(), f'{args.outf}/{args.dataset}/models/netD_{timestamp}_epoch_{i_epoch}.pth')
