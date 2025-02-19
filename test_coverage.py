from torch.utils.data import DataLoader
from utils.common import AverageMeter
from utils.metric import *
from model.hpn import hpn_cr
from dataloader import *
from tqdm import tqdm
import warnings
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', default=6, type=int, help='number of workers')
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--batch_size', default=1, type=int, help='')
    parser.add_argument('--input_data_folder', type=str, default='/home/gpz/data/SEN12MS-CR/test/data')
    parser.add_argument('--data_list_filepath', type=str, default='/home/gpz/data/SEN12MS-CR/test/data/data.csv')
    parser.add_argument('--weight_path', type=str, default='./backup/weight.pth')
    parser.add_argument('--is_use_cloudmask', type=bool, default=False)
    parser.add_argument('--is_test', type=bool, default=False)
    args = parser.parse_args()
    return args

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def eval(eval_loader, network):

    PSNR_2 = AverageMeter()
    SSIM_2 = AverageMeter()
    SAM_2 = AverageMeter()
    MAE_2 = AverageMeter()

    PSNR_4 = AverageMeter()
    SSIM_4 = AverageMeter()
    SAM_4 = AverageMeter()
    MAE_4 = AverageMeter()

    PSNR_6 = AverageMeter()
    SSIM_6 = AverageMeter()
    SAM_6 = AverageMeter()
    MAE_6 = AverageMeter()

    PSNR_8 = AverageMeter()
    SSIM_8 = AverageMeter()
    SAM_8 = AverageMeter()
    MAE_8 = AverageMeter()

    PSNR_10 = AverageMeter()
    SSIM_10 = AverageMeter()
    SAM_10 = AverageMeter()
    MAE_10 = AverageMeter()

    for batch in tqdm(test_loader, desc='Evaluating', unit="batch" ):
        optical_img = batch['cloudy_data'].cuda()
        target_img = batch['target'].cuda()
        s1_img = batch['s1_data'].cuda()
        source = batch['source'].cuda()
        cloud_coverage = batch['cloud_coverage']


        output = network(optical_img, s1_img).clamp_(0, 1)
        PSNR_val = Psnr(target_img, output)
        SSIM_val = Ssim(target_img, output)
        SAM_val = Sam(target_img, output)
        MAE_val = Mae(target_img, output)

        if cloud_coverage <= 0.2:
            PSNR_2.update(PSNR_val)
            SSIM_2.update(SSIM_val)
            SAM_2.update(SAM_val)
            MAE_2.update(MAE_val)
        elif 0.2 < cloud_coverage <= 0.4:
            PSNR_4.update(PSNR_val)
            SSIM_4.update(SSIM_val)
            SAM_4.update(SAM_val)
            MAE_4.update(MAE_val)
        elif 0.4 < cloud_coverage <= 0.6:
            PSNR_6.update(PSNR_val)
            SSIM_6.update(SSIM_val)
            SAM_6.update(SAM_val)
            MAE_6.update(MAE_val)
        elif 0.6 < cloud_coverage <= 0.8:
            PSNR_8.update(PSNR_val)
            SSIM_8.update(SSIM_val)
            SAM_8.update(SAM_val)
            MAE_8.update(MAE_val)
        elif 0.8 < cloud_coverage <= 1:
            PSNR_10.update(PSNR_val)
            SSIM_10.update(SSIM_val)
            SAM_10.update(SAM_val)
            MAE_10.update(MAE_val)

    print('PSNR_2: %f\n'
          'PSNR_4: %f\n'
          'PSNR_6: %f\n'
          'PSNR_8: %f\n'
          'PSNR_10: %f' % (PSNR_2.avg, PSNR_4.avg, PSNR_6.avg, PSNR_8.avg, PSNR_10.avg))
    print('SSIM_2: %f\n'
          'SSIM_4: %f\n'
          'SSIM_6: %f\n'
          'SSIM_8: %f\n'
          'SSIM_10: %f' % (SSIM_2.avg, SSIM_4.avg, SSIM_6.avg, SSIM_8.avg, SSIM_10.avg))
    print('MAE_2: %f\n'
          'MAE_4: %f\n'
          'MAE_6: %f\n'
          'MAE_8: %f\n'
          'MAE_10: %f' % (MAE_2.avg, MAE_4.avg, MAE_6.avg, MAE_8.avg, MAE_10.avg))
    print('SAM_2: %f\n'
          'SAM_4: %f\n'
          'SAM_6: %f\n'
          'SAM_8: %f\n'
          'SAM_10: %f' % (SAM_2.avg, SAM_4.avg, SAM_6.avg, SAM_8.avg, SAM_10.avg))

if __name__ == '__main__':
    args = arg_parse()

    weight_path = args.weight_path
    network = hpn_cr().cuda()
    network.load_state_dict(torch.load(weight_path))
    network.eval()
    for _, param in network.named_parameters():
        param.requires_grad = False
    _, _, test_filelist = get_train_val_test_filelists(args.data_list_filepath)
    test_data = AlignedDataset(args, test_filelist)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    eval(test_loader, network)
