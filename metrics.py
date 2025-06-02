#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    """读取渲染图像和真实图像，返回张量列表和图像名称"""
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        # 转换为张量并移动到GPU
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):
    """评估模型渲染质量，计算SSIM、PSNR和LPIPS指标"""
    full_dict = {}  # 存储整体评估结果
    per_view_dict = {}  # 存储每张视图的评估结果

    for scene_dir in model_paths:
        try:
            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                # 计算三种质量指标
                ssims = []
                psnrs = []
                lpipss = []
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))  # 结构相似性
                    psnrs.append(psnr(renders[idx], gts[idx]))  # 峰值信噪比
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))  # 感知相似性

                # 保存结果到JSON文件
                with open(scene_dir + "/results.json", 'w') as fp:
                    json.dump(full_dict[scene_dir], fp, indent=True)
                with open(scene_dir + "/per_view.json", 'w') as fp:
                    json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
