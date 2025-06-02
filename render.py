import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True  # 检查是否支持稀疏Adam优化器
except:
    SPARSE_ADAM_AVAILABLE = False

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    """渲染一组视图并保存结果"""
    # 设置渲染结果和真实图像的保存路径
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    # 创建输出目录
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # 遍历所有视图进行渲染
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 执行渲染
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]  # 获取真实图像

        # 如果使用训练测试曝光分离，只取图像的后半部分
        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        # 保存渲染结果和真实图像
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, separate_sh: bool):
    """渲染训练集和测试集"""
    with torch.no_grad():  # 禁用梯度计算
        # 初始化高斯模型和场景
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # 设置背景颜色
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 渲染训练集和测试集
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)  # 模型参数
    pipeline = PipelineParams(parser)  # 渲染管线参数
    parser.add_argument("--iteration", default=-1, type=int)  # 指定迭代次数
    parser.add_argument("--skip_train", action="store_true")  # 跳过训练集
    parser.add_argument("--skip_test", action="store_true")  # 跳过测试集
    parser.add_argument("--quiet", action="store_true")  # 静默模式
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # 初始化系统状态(随机数生成器)
    safe_state(args.quiet)

    # 执行渲染
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)