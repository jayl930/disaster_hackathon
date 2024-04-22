import os

from os import path, makedirs, listdir
import sys
import numpy as np

np.random.seed(1)
import random

random.seed(1)
from climax import ClimaXLegacy

import torch
from torch import nn
from torch.backends import cudnn

from torch.autograd import Variable

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from zoo.models import Res34_Unet_Loc

from utility import *

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

test_dir = "/Users/jaylee/Documents/work/hackathon/data"
pred_folder = "/Users/jaylee/Documents/work/hackathon/pred34_loc_"
models_folder = "/Users/jaylee/Documents/work/hackathon/model"

if __name__ == "__main__":
    t0 = timeit.default_timer()

    makedirs(pred_folder, exist_ok=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    models = []

    for seed in ["0"]:  # , 1, 2]:
        # snap_to_load = 'res34_loc_{}_1_best'.format(seed)
        snap_to_load = "climax_loc_0_0_best.pth"
        variables = ["R", "G", "B"]
        # model = ClimaXLegacy(img_size=[512,512],patch_size=16,default_vars=variables,pretrained='5.625deg.ckpt')
        model = ClimaXLegacy(
            img_size=[512, 512],
            patch_size=16,
            default_vars=variables,
            pretrained="/Users/adityaranjan/Documents/ncsa-hack/model/climax_loc_0_0_best.pth",
            upsampling_steps=[
                {"step_scale_factor": 2, "new_channel_dim": 1024, "feature_dim": 2048},
                {"step_scale_factor": 2, "new_channel_dim": 512, "feature_dim": 1024},
                {"step_scale_factor": 2, "new_channel_dim": 256, "feature_dim": 512},
                {"step_scale_factor": 2, "new_channel_dim": 128, "feature_dim": 256},
            ],
            feature_extractor_type="res-net",
            out_dim=1,
        )
        # model = Res34_Unet_Loc() #.cuda()
        model = nn.DataParallel(model)  # .cuda()
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(
            path.join(models_folder, snap_to_load), map_location="cpu"
        )
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
                # if '_post_' in f:
                fn = path.join(test_dir, f)

                img = cv2.imread(fn, cv2.IMREAD_COLOR)
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

                pred_full = np.asarray(pred).mean(axis=0)
                print("ored shape", pred_full.shape)
                msk = pred_full * 255
                msk = msk.astype("uint8").transpose(1, 2, 0)
                cv2.imwrite(
                    path.join(
                        pred_folder, "{0}.png".format(f.replace(".png", "_part1.png"))
                    ),
                    msk[..., 0],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )

    elapsed = timeit.default_timer() - t0
    print("Time: {:.3f} min".format(elapsed / 60))
