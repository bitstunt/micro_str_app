from skimage import segmentation
from PIL import Image
import cv2
import os
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F

## Generate Superpixels
def generate_superpixels(image, scale, sigma):
    seg_map = segmentation.felzenszwalb(image, scale=scale, sigma=sigma)
    boundaries = segmentation.mark_boundaries(image, seg_map)
    return boundaries, seg_map

## Convolution Methods
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.bat= nn.BatchNorm2d(out_ch)


    def forward(self, x):
        x = self.conv(x)
        x=self.bat(x)
        return x

## Denormalizing Image
def denormalizeimage(images, mean=(0., 0., 0.), std=(1., 1., 1.)):
    images = images.cpu().numpy()
    # N*C*H*W to N*H*W*C
    images = images.transpose((0,2,3,1))
    images *= std
    images += mean
    images *= 255.0
    # N*H*W*C to N*C*H*W
    images = images.transpose((0,3,1,2))
    return torch.tensor(images)

## Deep Clustering Subnetwork
class DCS(nn.Module):
    def __init__(self, c, k, stage_num=3):
        super(DCS, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k)  #
        mu.normal_(0, math.sqrt(2. / k))  #
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 =nn.Conv2d(c, c, 1, bias=False)

        ####iteration
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)  # b * n * c
                z = torch.bmm(x_t, mu)  # b * n * k
                z = F.softmax(z, dim=2)  # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)
                mu = self._l2norm(mu, dim=1)

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = x.view(b, c, h, w)  # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn #
        x = F.relu(x, inplace=True)

        return x, mu

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

## Full Network
class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()
        self.inc = inconv(inp_dim, 64)
        self.down1 = down(64, 128)
        self.up4 = up(192, 128)
        self.dcs0 = DCS(128, 32, 3)
        self.outc = outconv(128, mod_dim2)
        self.dcs1 = DCS(mod_dim2, 32, 3)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up4(x2, x1)
        x = self.outc(x)
        x, mu = self.dcs1(x)
        return x


## Arguments
class Args(object):
    train_epoch = 300 ## training iteration T ##
    mod_dim1 = 64  #
    mod_dim2 = 6 #
    gpu_id =0
    min_label_num = 2  # if the label number small than it, break loop
    max_label_num = 256  # if the label number small than it, start to show result image.


## Run
def run(np_image, seg_map):
    image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    args = Args()
    torch.manual_seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # choose GPU:0

    softmax = nn.Softmax(dim=1)

    '''segmentation map'''
    show = segmentation.mark_boundaries(image, seg_map)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)]

    '''train init'''
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    '''loading image to tensor'''
    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    tensor = torch.from_numpy(tensor).to(device)

    '''setting up model'''
    model = MyNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)

    '''train loop'''
    model.train()
    for batch_idx in range(args.train_epoch):
        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)[0]
        output1 = output
        output1 = output1[np.newaxis, :, :, :]
        output2 = output[0:1, :, :]
        croppings = (output2 > 0).float()

        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy()

        '''refine'''
        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]

        '''backward'''
        target = torch.from_numpy(im_target)
        target = target.to(device)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        '''generating segmented mask'''
        un_label = np.unique(im_target)
        if len(un_label) <= args.min_label_num:
            break

    height, width, __ = image.shape
    label = im_target.reshape((height, width))

    return label