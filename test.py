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
    parser.add_argument('--input_data_folder', type=str, default='./SEN12MS-CR/test/')
    parser.add_argument('--data_list_filepath', type=str, default='./SEN12MS-CR/test/data.csv')
    parser.add_argument('--weight_path', type=str, default='./backup/weight.pth')
    parser.add_argument('--is_use_cloudmask', type=bool, default=False)
    parser.add_argument('--is_test', type=bool, default=False)
    args = parser.parse_args()
    return args

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def eval(eval_loader, network):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    SAM = AverageMeter()
    MAE = AverageMeter()
    
    for batch in tqdm(eval_loader, desc='Evaluating', unit="batch"):
        source_img = batch['cloudy_data'].cuda()
        target_img = batch['target'].cuda()
        s1_img = batch['s1_data'].cuda()
        source = batch['source'].cuda()
        idx_img = batch['file_name']

        output = network(source_img, s1_img).clamp_(0, 1)
        PSNR_val = Psnr(target_img, output)
        SSIM_val = Ssim(target_img, output)
        MAE_val = Mae(target_img, output)
        SAM_val = Sam(target_img, output)

        SAM.update(SAM_val)
        PSNR.update(PSNR_val)
        SSIM.update(SSIM_val)
        MAE.update(MAE_val)
        
    print('PSNR: %f\n'
          'SSIM: %f\n'
          'SAM: %f\n'
          'MAE: %f' % (PSNR.avg, SSIM.avg, SAM.avg, MAE.avg))
    fd = open('metric.txt', 'a')
    fd.write('  PSNR.avg:' + str(PSNR.avg) + '  SSIM.avg:' + str(SSIM.avg) + '  SAM.avg:' + str(
        SAM.avg) + '  MAE.avg:' + str(MAE.avg) + '\n')
    fd.close()


if __name__ == '__main__':
    args = arg_parse()
    _, _, test_filelist = get_train_val_test_filelists(args.data_list_filepath)
    eval_data = AlignedDataset(args, test_filelist)
    eval_loader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    weight_path = args.weight_path 
    network = hpn_cr().cuda()
    network.load_state_dict(torch.load(weight_path))
    network.eval()
    for _, param in network.named_parameters():
        param.requires_grad = False
    eval(eval_loader, network)
