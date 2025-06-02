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

import os
import logging
from argparse import ArgumentParser
import shutil

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
# 设置命令行参数
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')  # 禁用GPU加速
parser.add_argument("--skip_matching", action='store_true')  # 跳过特征匹配步骤
parser.add_argument("--source_path", "-s", required=True, type=str)  # 输入数据路径
parser.add_argument("--camera", default="OPENCV", type=str)  # 相机模型
parser.add_argument("--colmap_executable", default="", type=str)  # COLMAP可执行文件路径
parser.add_argument("--resize", action="store_true")  # 是否调整图像大小
parser.add_argument("--magick_executable", default="", type=str)  # ImageMagick可执行文件路径
args = parser.parse_args()

# 配置COLMAP和ImageMagick命令
colmap_command = '"{}"'.format(args.colmap_executable) if args.colmap_executable else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if args.magick_executable else "magick"
use_gpu = 1 if not args.no_gpu else 0  # GPU使用标志

if not args.skip_matching:
    # 特征提取流程
    feat_extracton_cmd = colmap_command + " feature_extractor " \
                                          "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # 特征匹配流程
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # 光束法平差(Bundle Adjustment)
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --output_path " + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

# 图像去畸变处理
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

# 整理稀疏重建结果
files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
for file in files:
    if file == '0':
        continue
    shutil.move(os.path.join(args.source_path, "sparse", file),
                os.path.join(args.source_path, "sparse", "0", file))

# 图像大小调整
if args.resize:
    # 创建多分辨率图像目录
    for res in ["_2", "_4", "_8"]:
        os.makedirs(args.source_path + f"/images{res}", exist_ok=True)

    # 处理每张图像
    for file in os.listdir(args.source_path + "/images"):
        for res, scale in [("_2", "50%"), ("_4", "25%"), ("_8", "12.5%")]:
            dest = os.path.join(args.source_path, f"images{res}", file)
            shutil.copy2(os.path.join(args.source_path, "images", file), dest)
            exit_code = os.system(f"{magick_command} mogrify -resize {scale} {dest}")
            if exit_code != 0:
                logging.error(f"{scale} resize failed with code {exit_code}. Exiting.")
                exit(exit_code)

print("Done.")
