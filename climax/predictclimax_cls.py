import os

from os import path, makedirs, listdir
import sys
import numpy as np

np.random.seed(1)
import random

random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn
from climax import ClimaXLegacy

from torch.autograd import Variable

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from zoo.models import Res34_Unet_Double

from utility import *

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

test_dir = "/Users/adityaranjan/Documents/ncsa-hack/images"
pred_folder = "/Users/adityaranjan/Documents/ncsa-hack/pred34_cls_"
models_folder = "/Users/adityaranjan/Documents/ncsa-hack/model"

if __name__ == "__main__":
    t0 = timeit.default_timer()

    seed = int(sys.argv[1])
    # vis_dev = sys.argv[2]

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev

    # pred_folder = 'res34cls2_{}_tuned'.format(seed)
    makedirs(pred_folder, exist_ok=True)

    # cudnn.benchmark = True

    models = []

    snap_to_load = "climax_cls_cce_0_0_best.pth"
    variables = ["R", "G", "B"]
    # model = ClimaXLegacy(img_size=[512,512],patch_size=16,default_vars=variables,pretrained='5.625deg.ckpt')
    # model = ClimaXLegacy(img_size=[512, 512], patch_size=16, default_vars=variables,
    #                      pretrained='/Users/adityaranjan/Documents/ncsa-hack/model/climax_cls_cce_0_0_best.pth',
    #                      upsampling_steps=[{"step_scale_factor": 2, "new_channel_dim": 1024, "feature_dim": 2048},
    #                                        {"step_scale_factor": 2, "new_channel_dim": 512, "feature_dim": 1024},
    #                                        {"step_scale_factor": 2, "new_channel_dim": 256, "feature_dim": 512},
    #                                        {"step_scale_factor": 2, "new_channel_dim": 128, "feature_dim": 256}, ],
    #                      feature_extractor_type="res-net", out_dim=1)

    model = ClimaXLegacy(
        img_size=[256, 256],
        patch_size=16,
        default_vars=variables,
        pretrained="/Users/adityaranjan/Documents/ncsa-hack/model/climax_cls_cce_0_0_best.pth",
        out_dim=5,
        upsampling_steps=[
            {"step_scale_factor": 2, "new_channel_dim": 1024, "feature_dim": 2048},
            {"step_scale_factor": 2, "new_channel_dim": 512, "feature_dim": 1024},
            {"step_scale_factor": 2, "new_channel_dim": 256, "feature_dim": 512},
            {"step_scale_factor": 2, "new_channel_dim": 128, "feature_dim": 256},
        ],
        feature_extractor_type="res-net",
        double=True,
    )

    model = nn.DataParallel(model)  # .cuda()
    print("=> loading checkpoint '{}'".format(snap_to_load))
    checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location="cpu")
    loaded_dict = checkpoint["state_dict"]
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    print(
        "loaded checkpoint '{}' (epoch {}, best_score {})".format(
            snap_to_load, checkpoint["epoch"], checkpoint["best_score"]
        )
    )
    model.eval()
    models.append(model)

    with torch.no_grad():
        for f in tqdm(sorted(listdir(test_dir))):
            if "_pre_" in f:
                fn = path.join(test_dir, f)

                img = cv2.imread(fn, cv2.IMREAD_COLOR)
                img2 = cv2.imread(fn.replace("_pre_", "_post_"), cv2.IMREAD_COLOR)

                img = np.concatenate([img, img2], axis=2)
                img = preprocess_inputs(img)

                inp = []
                inp.append(img)
                inp.append(img[::-1, ...])
                inp.append(img[:, ::-1, ...])
                inp.append(img[::-1, ::-1, ...])
                inp = np.asarray(inp, dtype="float")
                inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
                inp = Variable(inp)  # .cuda()

                pred = []
                for model in models:
                    msk = model(inp)
                    msk = torch.sigmoid(msk)
                    msk = msk.cpu().numpy()

                    pred.append(msk[0, ...])
                    pred.append(msk[1, :, ::-1, :])
                    pred.append(msk[2, :, :, ::-1])
                    pred.append(msk[3, :, ::-1, ::-1])
                # for i in range(len(msk[0])):
                #     np.savetxt(path.join(pred_folder, f'{f.replace(".png", f"_pred_channel{i}.txt")}'),
                #                msk[i], fmt='%.6f')

                print(msk[..., 0])
                pred_full = np.asarray(pred).mean(axis=0)
                msk = pred_full * 255
                # msk = msk.astype('uint8').transpose(1, 2, 0)
                threshold_value = 127
                binary_mask = np.where(msk > threshold_value, 255, 0).astype(np.uint8)
                masked_pred_full = np.where(binary_mask, pred_full, 0)

                # Calculate damage severity only over relevant areas
                relevant_area = np.sum(binary_mask)  # Total number of relevant pixels

                if relevant_area > 0:
                    damage_severity = (
                        np.sum(masked_pred_full) / relevant_area * 100
                    )  # Percentage of damaged area over relevant pixels
                else:
                    damage_severity = 0  # Avoid division by zero

                threshold = 0.5
                # damage_severity = np.mean(pred_full > threshold) * 100

                print("damage_severity", damage_severity)
                # for i in range(pred_full.shape[0]):
                #     np.savetxt(path.join(pred_folder, f'{f.replace(".png", f"_pred_full_channel{i}.txt")}'),
                #                pred_full[i], fmt='%.6f')
                msk = pred_full * 255
                msk = msk.astype("uint8").transpose(1, 2, 0)
                print("size 1", msk[..., :3].shape)
                print("size 2", msk[..., 2:].shape)
                cv2.imwrite(
                    path.join(
                        pred_folder, "{0}.png".format(f.replace(".png", "_part1.png"))
                    ),
                    msk[..., :3],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )
                cv2.imwrite(
                    path.join(
                        pred_folder, "{0}.png".format(f.replace(".png", "_part2.png"))
                    ),
                    msk[..., 2:],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )

    elapsed = timeit.default_timer() - t0
    print("Time: {:.3f} min".format(elapsed / 60))
