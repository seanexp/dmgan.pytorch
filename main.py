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
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models import Generator, Discriminator, Encoder


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['mnist', 'chairs'], help='which data to use')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--nz', type=int, default=8, help='size of the latent z vector')
parser.add_argument('--ng', dest='n_gen', type=int, default=10, help='size of the latent z vector')
parser.add_argument('--n_epoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netQ', default='', help="path to netQ (to continue training)")
parser.add_argument('--lambda', dest='lambda_q', type=float, default=1., help='coefficient for variational mutual information')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
print(args)

os.makedirs(os.path.join(args.outf, 'images'), exist_ok=True)
os.makedirs(os.path.join(args.outf, 'models'), exist_ok=True)

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if args.dataset == 'mnist':
    dataset = dset.MNIST(root='datasets/mnist', download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,)),
                       ]))
elif args.dataset == 'chairs':

    container = np.load('/home/sjjung/data/chairs/chairs_half_64.npz')
    img_data = container['img']
    label_data = container['label']

    transform = T.ToTensor()

    img_tensor = torch.stack([transform(i) for i in img_data])
    label_tensor = torch.Tensor(label_data)

    dataset = torch.utils.data.TensorDataset(img_tensor, label_tensor)
else:
    raise NotImplementedError

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=int(args.workers), pin_memory=args.cuda)

device = torch.device("cuda:0" if args.cuda else "cpu")


nz = args.nz
n_gen = args.n_gen

netG = Generator(nz, n_gen).to(device)
if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))
print(netG)

netD = Discriminator().to(device)
if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))
print(netD)

netQ = Encoder(n_gen).to(device)
if args.netQ != '':
    netQ.load_state_dict(torch.load(args.netQ))
print(netQ)

data_per_gen = args.batch_size // n_gen
fixed_noise = torch.randn(n_gen, nz, device=device).repeat(1, data_per_gen).view(-1, nz)
fixed_gidx = torch.arange(n_gen).repeat(data_per_gen)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam([
                        {'params': netG.parameters()},
                        {'params': netQ.parameters()}
                        ], lr=args.lr, betas=(args.beta1, 0.999))

from datetime import datetime
timestamp = datetime.now().strftime('%b%d_$H-%M-%S')

for i_epoch in range(1, args.n_epoch+1):
    for i, (real, _) in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize E_{x ~ p_x} [D(x)] - E_{z ~ p_z} [D(G(z))]
        ###########################
        batch_size = real.size(0)
        real = real.to(device)

        for _ in range(3):
            # train with real
            netD.zero_grad()

            output = netD(real)
            real_score = output.mean()

            # train with fake
            noise = torch.randn(batch_size, nz, device=device)
            g_idx = torch.multinomial(torch.ones(n_gen), replacement=True)
            fake = netG(noise, g_idx)

            output = netD(fake.detach())
            fake_score = output.mean()
            errD = - real_score + fake_score
            errD.backward()
            optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizerG.zero_grad()
        output = netD(fake)
        fake_score = output.mean()

        errQ = F.cross_entropy(netQ(fake), g_idx)

        errG = - fake_score + args.lambda_q * errQ
        errG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f'
              % (i_epoch, args.n_epoch, i, len(dataloader),
                 errD.item(), errG.item(), real_score, fake_score))

    if i_epoch % 5 == 0:
        vutils.save_image(real,
                f'{args.outf}/images/real_samples.png', nrow=n_gen,
                normalize=True)
        fake = netG(fixed_noise, fixed_gidx)
        vutils.save_image(fake.detach(),
                f'{args.outf}/images/fake_samples_epoch_{i_epoch}.png', nrow=n_gen,
                normalize=True)

        # do checkpointing
        torch.save(netG.state_dict(), f'{args.outf}/models/netG_{timestamp}_epoch_{i_epoch}_{args.dataset}.pth')
        torch.save(netD.state_dict(), f'{args.outf}/models/netD_{timestamp}_epoch_{i_epoch}_{args.dataset}.pth')
