import argparse
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor  # To share lru_cache
from datetime import datetime
from os.path import join

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloader import OmniStereoDataset
from dataloader.custom_transforms import Resize, ToTensor, Normalize
from models import OmniMVS
from models import SphericalSweeping
from utils import InvDepthConverter, evaluation_metrics

parser = argparse.ArgumentParser(description='Training for OmniMVS',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('root_dir', metavar='DATA_DIR', help='path to dataset')
parser.add_argument('-t', '--train-list', default='./dataloader/omnithings_train.txt',
                    type=str, help='Text file includes filenames for training')
parser.add_argument('-v', '--val-list', default='./dataloader/omnithings_val.txt',
                    type=str, help='Text file includes filenames for validation')
parser.add_argument('--epochs', default=21, type=int, metavar='N', help='total epochs')
parser.add_argument('--pretrained', default=None, metavar='PATH',
                    help='path to pre-trained model')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--min_depth', type=float, default=0.55, help='minimum depth in m')
parser.add_argument('--fov', type=float, default=220, help='field of view of the camera in degree')
if False:
    # Paper setting
    parser.add_argument('--ndisp', type=int, default=192, metavar='N', help='number of disparity')
    parser.add_argument('--input_width', type=int, default=800, metavar='N', help='input image width')
    parser.add_argument('--input_height', type=int, default=768, metavar='N', help='input image height')
    parser.add_argument('--output_width', type=int, default=640, metavar='N', help='output depth width')
    parser.add_argument('--output_height', type=int, default=320, metavar='N', help='output depth height')
else:
    # Light weight
    parser.add_argument('--ndisp', type=int, default=64, metavar='N', help='number of disparity')
    parser.add_argument('--input_width', type=int, default=500, metavar='N', help='input image width')
    parser.add_argument('--input_height', type=int, default=480, metavar='N', help='input image height')
    parser.add_argument('--output_width', type=int, default=512, metavar='N', help='output depth width')
    parser.add_argument('--output_height', type=int, default=256, metavar='N', help='output depth height')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd')
parser.add_argument('--arch', default='omni_small', type=str, help='architecture name for log folder')
parser.add_argument('--log-interval', type=int, default=5, metavar='N', help='tensorboard log interval')


def main():
    args = parser.parse_args()
    print('Arguments:')
    print(json.dumps(vars(args), indent=1))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    if device.type != 'cpu':
        cudnn.benchmark = True
    print("device:", device)

    ###############################
    # Setup model
    sweep = SphericalSweeping(args.root_dir, h=args.output_height, w=args.output_width, fov=args.fov)
    model = OmniMVS(sweep, args.ndisp, args.min_depth, h=args.output_height, w=args.output_width)
    model = model.to(device)

    # cache
    num_cam = 4
    pool = ThreadPoolExecutor(5)
    futures = []
    for i in range(num_cam):
        for d in model.depths[::2]:
            futures.append(pool.submit(sweep.get_grid, i, d))

    # Setup solver
    print('=> setting optimizer')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
    print('=> setting scheduler')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2 * args.epochs // 3, gamma=0.1)

    start_epoch = 0
    # Load pretrained model
    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        param_check = {
            'ndisp': model.ndisp,
            'min_depth': model.min_depth,
            'output_width': model.w,
            'output_height': model.h,
        }
        for key, val in param_check.items():
            if not checkpoint[key] == val:
                print(f'Error! Key:{key} is not the same as the checkpoints')

        print("=> using pre-trained weights")
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Resume training from epoch {}".format(start_epoch))

    # Setup solver
    timestamp = datetime.now().strftime("%m%d-%H%M")
    log_folder = join('checkpoints', f'{args.arch}_{timestamp}')
    print(f'=> create log folder: {log_folder}')
    os.makedirs(log_folder, exist_ok=True)
    with open(join(log_folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=1)
    writer = SummaryWriter(log_dir=log_folder)
    writer.add_text('args', json.dumps(vars(args), indent=1))

    # Setup dataloader
    image_size = (args.input_width, args.input_height)
    depth_size = (args.output_width, args.output_height)
    train_transform = transforms.Compose([Resize(image_size, depth_size), ToTensor(), Normalize()])
    trainset = OmniStereoDataset(args.root_dir, args.train_list, transform=train_transform, fov=args.fov)
    val_transform = transforms.Compose([Resize(image_size, depth_size), ToTensor(), Normalize()])
    valset = OmniStereoDataset(args.root_dir, args.val_list, transform=val_transform, fov=args.fov)
    print(f'{len(trainset)} samples for training.')
    print(f'{len(valset)} samples for validation.')
    train_loader = DataLoader(trainset, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(valset, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    print('=> wait for a while until all tasks in pool are finished')
    pool.shutdown()
    print('=> Done!')

    ###############################
    # Start training
    ###############################
    print("Start training")
    for epoch in range(start_epoch, args.epochs):
        # train
        ave_loss = train(args, model, train_loader, optimizer, writer, epoch, device)
        print(f"Epoch:{epoch}/{args.epochs}, Train Loss average:{ave_loss:.4f}")

        # save data here
        save_data = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'ave_loss': ave_loss,
            'ndisp': model.ndisp,
            'min_depth': model.min_depth,
            'output_width': model.w,
            'output_height': model.h,
        }
        # save images
        torch.save(save_data, join(log_folder, f'checkpoints_{epoch}.pth'))

        # validation
        ave_loss = validation(args, model, val_loader, writer, epoch, device)
        print(f"Epoch:{epoch}/{args.epochs}, Val Loss average:{ave_loss:.4f}")
        scheduler.step()

    writer.close()
    print('Finish training')


def train(args, model, train_loader, optimizer, writer, epoch, device):
    invd_0 = model.inv_depths[0]
    invd_max = model.inv_depths[-1]
    converter = InvDepthConverter(args.ndisp, invd_0, invd_max)

    losses = []
    model.train()
    pbar = tqdm(train_loader)
    for idx, batch in enumerate(pbar):
        # to cuda
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        pred = model(batch)

        gt_idepth = batch['idepth']
        # Loss function
        gt_invd_idx = converter.invdepth_to_index(gt_idepth)
        loss = nn.L1Loss()(pred, gt_invd_idx)
        losses.append(loss.item())

        # update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        display = OrderedDict(epoch=f"{epoch:>2}", loss=f"{losses[-1]:.4f}")
        pbar.set_postfix(display)

        # tensorboard log
        niter = epoch * len(train_loader) + idx
        if idx % args.log_interval == 0:
            writer.add_scalar('train/loss', loss.item(), niter)
        if idx % (200 * args.log_interval) == 0:
            imgs = []
            for cam in model.cam_list:
                imgs.append(0.5 * batch[cam][0] + 0.5)
            img_grid = make_grid(imgs, nrow=2, padding=5, pad_value=1)
            writer.add_image('train/fisheye', img_grid, niter)
            writer.add_image('train/pred', pred / model.ndisp, niter)
            writer.add_image('train/gt', gt_invd_idx / model.ndisp, niter)

    # End of one epoch
    ave_loss = sum(losses) / len(losses)
    writer.add_scalar('train/loss_ave', ave_loss, epoch)

    return ave_loss


def validation(args, model, val_loader, writer, epoch, device):
    invd_0 = model.inv_depths[0]
    invd_max = model.inv_depths[-1]
    converter = InvDepthConverter(args.ndisp, invd_0, invd_max)

    preds = []
    gts = []
    losses = []
    model.eval()
    pbar = tqdm(val_loader)
    for idx, batch in enumerate(pbar):
        with torch.no_grad():
            # to cuda
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            pred = model(batch)

            gt_idepth = batch['idepth']
            # Loss function
            gt_invd_idx = converter.invdepth_to_index(gt_idepth)
            loss = nn.L1Loss()(pred, gt_invd_idx)
            losses.append(loss.item())
            # save for evaluation
            preds.append(pred.cpu())
            gts.append(gt_invd_idx.cpu())

        # update progress bar
        display = OrderedDict(epoch=f"{epoch:>2}", loss=f"{losses[-1]:.4f}")
        pbar.set_postfix(display)

        # tensorboard log
        niter = epoch * len(val_loader) + idx
        if idx % args.log_interval == 0:
            writer.add_scalar('val/loss', loss.item(), niter)
        if idx % 200 * args.log_interval == 0:
            imgs = []
            for cam in model.cam_list:
                imgs.append(0.5 * batch[cam][0] + 0.5)
            img_grid = make_grid(imgs, nrow=2, padding=5, pad_value=1)
            writer.add_image('val/fisheye', img_grid, niter)
            writer.add_image('val/pred', pred / model.ndisp, niter)
            writer.add_image('val/gt', gt_invd_idx / model.ndisp, niter)

    preds = torch.cat(preds)
    gts = torch.cat(gts)
    errors, error_names = evaluation_metrics(preds, gts, args.ndisp)
    for name, val in zip(error_names, errors):
        writer.add_scalar(f'val_metrics/{name}', val, epoch)
    print("Evaluation metrics: ")
    print("{:>8}, {:>8}, {:>8}, {:>8}, {:>8}".format(*error_names))
    print("{:8.4f}, {:8.4f}, {:8.4f}, {:8.4f}, {:8.4f}".format(*errors))
    # End of one epoch
    ave_loss = sum(losses) / len(losses)
    writer.add_scalar('val/loss_ave', ave_loss, epoch)

    return ave_loss


if __name__ == '__main__':
    main()
