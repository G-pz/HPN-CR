import os
import argparse
import torch
import torch.nn as nn
from utils.check_point_rw import save_checkpoint
from torch.utils.data import DataLoader
from model.hpn import hpn_cr
from loss import *
from dataloader import *
import numpy as np
from utils.common import AverageMeter
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(100)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', default=6, type=int, help='number of workers')
    parser.add_argument('--lr', default=0.0001, type=float, help='experiment setting')
    parser.add_argument('--optimizer', default='adamw', type=str, help='GPUs used for training')
    parser.add_argument('--batch_size', default=32, type=int, help='')
    parser.add_argument('--backup_dir', default='./backup', type=str, help='')
    parser.add_argument('--star_epoch', default=0, type=int, help='')
    parser.add_argument('--total_epoch', default=15, type=int, help='')
    parser.add_argument('--checkpoint_interval', default=1, type=int, help='')
    parser.add_argument('--weight_path', default=None, type=str, help='./backup/weight_10.pth')
    parser.add_argument('--cuda_num', default='0', type=str, help='')

    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--input_data_folder', type=str, default='./SEN12MS-CR/train/')
    parser.add_argument('--data_list_filepath', type=str, default='./SEN12MS-CR/train/data.csv')
    parser.add_argument('--is_test', type=bool, default=False)

    args = parser.parse_args()
    return args


def train(train_loader, network, criterion, optimizer, epoch_idx):
    losses = AverageMeter()
    torch.cuda.empty_cache()
    network.train()

    idx_iter = 0
    for batch in train_loader:
        cloudy_img = batch['cloudy_data'].cuda()
        s1_img = batch['s1_data'].cuda()
        target_img = batch['target'].cuda()
        # source = batch['source'].cuda()
        output = network(cloudy_img, s1_img)
        loss = criterion.forward(output, target_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        idx_iter += 1
        print('epoch: {} Iter: {} Loss: {} avg: {:.6f}'.format(epoch_idx, idx_iter, loss.item(), losses.avg))
    fd = open('loss.txt', 'a')
    fd.write(str(epoch_idx) + ': ' + str(losses.avg) + '\n')
    fd.close()


if __name__ == '__main__':
    print('---------------------------start---------------------------')
    args = arg_parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num

    network = hpn_cr()
    if args.weight_path is not None:
        network.load_state_dict(torch.load(args.weight_path), strict=True)
    network = nn.DataParallel(network).cuda()

    criterion = sl1_ssim_sam_loss().cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    train_filelist, _, _ = get_train_val_test_filelists(args.data_list_filepath)
    train_data = AlignedDataset(args, train_filelist)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    for epoch_idx in range(args.star_epoch, args.star_epoch + args.total_epoch):
        train(train_loader, network, criterion, optimizer, epoch_idx)
        scheduler.step()
        if epoch_idx % args.checkpoint_interval == 0:
            save_checkpoint(network, args.backup_dir, f"weight_%d.pth" % (epoch_idx + 1))
