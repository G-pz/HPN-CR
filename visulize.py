import os
import torch
import argparse
from torch.utils.data import DataLoader
from utils import chw_to_hwc, write_rslt
from model.hpn_5_4branches import *
from dataloader import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', default=6, type=int, help='number of workers')
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--batch_size', default=1, type=int, help='')
    parser.add_argument('--input_data_folder', type=str, default='/home/gpz/data_visulize/')
    parser.add_argument('--data_list_filepath', type=str, default='/home/gpz/data_visulize/data.csv')
    parser.add_argument('--is_use_cloudmask', type=bool, default=False)
    parser.add_argument('--is_test', type=bool, default=False)
    args = parser.parse_args()
    return args

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test(test_loader, network, result_dir):
    torch.cuda.empty_cache()
    network.eval()
    for batch in test_loader:
        source_img = batch['cloudy_data'].cuda()
        sar_data = batch['s1_data'].cuda()
        source = batch['source'].cuda()
        idx_img = batch['file_name']
        output = network(source_img, sar_data).cpu()
        for i in range(0, args.batch_size):
            filename = idx_img[i]
            print(filename)
            out = output[i].detach().numpy()
            write_rslt(os.path.join(result_dir, filename), out)




if __name__ == '__main__':
    img_dir = ''
    result_dir = './'
    args = arg_parse()

    weight_path = './'
    network = mbn_cr().cuda()
    network.load_state_dict(torch.load(weight_path))

    _, _, test_filelist = get_train_val_test_filelists(args.data_list_filepath)
    test_data = AlignedDataset(args, test_filelist)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    test(test_loader, network, result_dir)
