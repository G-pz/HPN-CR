import torch
import torch.nn as nn

def save_feature_to_img(features, name, method='cv2', channel=None, output_dir=None, maxmin=None):
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import os

    if output_dir is None:
        output_dir = ''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # for i in range(len(features)):
    if isinstance(features, list) or isinstance(features, tuple):
        for i in range(3):
            features_ = features[i]
            for j in range(features_.shape[0]):
                upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
                features_ = upsample(features_)

                feature = features_[j, :, :, :]
                if channel is None:
                    feature = torch.sum(feature, 0)
                else:
                    feature = feature[channel, :, :]
                feature = feature.detach().cpu().numpy()  

                dist_dir = os.path.join(output_dir, 'hpn')
                if not os.path.exists(dist_dir):
                    os.mkdir(dist_dir)

                if method == 'cv2':
                    if maxmin is not None:
                        img = (feature - maxmin[1]) / (maxmin[0] - maxmin[1] + 1e-5) * 255
                    else:
                        img = (feature - np.amin(feature)) / (
                                    np.amax(feature) - np.amin(feature) + 1e-5) * 255  
                    img = img.astype(np.uint8)
                    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    # plt.imshow(feature)
                    plt.axis('off')
                    cv2.imwrite(os.path.join(dist_dir, name + str(i) + '.jpg'), img)

                elif method == 'matshow':
                    plt.matshow(feature, interpolation='nearest')
                    plt.colorbar()
                    plt.axis('off')

                    plt.savefig(os.path.join(dist_dir, name + str(i) + '.png'))
                    plt.close()
                else:
                    NotImplementedError()

    else:
        for j in range(features.shape[0]):
            upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            features = upsample(features)

            feature = features[j, :, :, :]
            if channel is None:
                feature = torch.sum(feature, 0)
            else:
                feature = feature[channel, :, :]
            feature = feature.detach().cpu().numpy()  

            dist_dir = os.path.join(output_dir, 'hpn')
            if not os.path.exists(dist_dir):
                os.mkdir(dist_dir)

            if method == 'cv2':
                if maxmin is not None:
                    img = (feature - maxmin[1]) / (maxmin[0] - maxmin[1] + 1e-5) * 255
                else:
                    img = (feature - np.amin(feature)) / (
                                np.amax(feature) - np.amin(feature) + 1e-5) * 255  # 注意要防止分母为0！
                img = img.astype(np.uint8)
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                # plt.imshow(feature)
                plt.axis('off')
                cv2.imwrite(os.path.join(dist_dir, name + '.jpg'), img)

            elif method == 'matshow':
                plt.matshow(feature, interpolation='nearest')
                plt.colorbar()
                plt.axis('off')

                plt.savefig(os.path.join(dist_dir, name + '.png'))
                plt.close()
            else:
                NotImplementedError()


def feature_map_norm(x):
    # input (B, C, H, W)
    bs, c, h, w = x.shape
    x = x.view(bs, -1, h * w)
    x_mean = torch.mean(x, dim=2, keepdim=True)
    x_std = torch.std(x, dim=2, keepdim=True)
    x = (x - x_mean) / x_std
    x = x.view(bs, c, h, w)
    return x