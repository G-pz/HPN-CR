import numpy as np
import skimage.io as skio


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def write_img(filename, img, cloud_data, sar_vh_data, sar_vv_data, target_img):
    img = np.round((img.copy() * 255.0)).astype('uint8')
    target_img = np.round((target_img.copy() * 255.0)).astype('uint8')
    img = np.concatenate((img, cloud_data, sar_vh_data, sar_vv_data, target_img), axis=1)
    skio.imsave(filename, img)


def write_rslt(filename, img):
    img = np.round((img.copy() * 10000.0)).astype('uint16')
    skio.imsave(filename, img)

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0]).copy()
