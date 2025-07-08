import numpy as np
import shap
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.segmentation import slic

from data.load_dataset import load_cifar
from model.load_model import load_model

def KernelShap_img():
    # 加载模型和数据
    model = load_model("../../../cfz_train/utils/ResNet.py",
                       "ResNet18",
                       "../../cfz_train/utils/resnet18_cifar10.pt")
    dataloader = load_cifar()
    model.eval()

    # 取出要解释的5张图像
    images, labels = next(iter(dataloader))
    images = images[:5]
    labels = labels[:5]

    # 转换为numpy格式(5, H, W, C)
    imgs_orig = images.permute(0, 2, 3, 1).cpu().numpy()
    print("标签是:", labels.tolist())

    # 对每张图片进行超像素分割
    segments_list = [slic(img_orig, n_segments=50, compactness=10, sigma=1) for img_orig in imgs_orig]

    # 遮挡超像素块，生成masked图像
    def mask_image(zs, segmentation, image, background=None):
        if background is None:
            background = image.mean((0, 1))
        out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
        for i in range(zs.shape[0]):
            out[i, :, :, :] = image
            for j in range(zs.shape[1]):
                if zs[i, j] == 0:
                    out[i][segmentation == j, :] = background
        return out

    # 用模型预测masked图像
    def f(z, segmentation, image):
        imgs = mask_image(z, segmentation, image, background=None)
        # NHWC -> NCHW
        imgs = imgs.transpose((0, 3, 1, 2))
        imgs_tensor = torch.tensor(imgs, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(imgs_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    shap_values_list = []
    top_preds_list = []

    # 计算每张图的shap值
    for i in range(5):
        # 全部遮挡背景
        background = np.zeros((1, 50))
        explainer = shap.KernelExplainer(lambda z: f(z, segments_list[i], imgs_orig[i]), background)
        shap_values = explainer.shap_values(np.ones((1, 50)), nsamples=1000)
        shap_values_list.append(shap_values)

        # 获取模型预测
        img_tensor_batch = images[i].unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            pred = model(img_tensor_batch)
            probs = torch.nn.functional.softmax(pred, dim=1).cpu().numpy()
        top_preds = np.argsort(-probs[0])
        top_preds_list.append(top_preds)

    # 自定义颜色映射：红色负影响，绿色正影响
    colors = []
    for l in np.linspace(1, 0, 100):
        colors.append((245 / 255, 39 / 255, 87 / 255, l))
    for l in np.linspace(0, 1, 100):
        colors.append((24 / 255, 196 / 255, 93 / 255, l))
    cm = LinearSegmentedColormap.from_list("shap", colors)

    # 将每个超像素的SHAP值映射会原图像
    def fill_segmentation(values, segmentation):
        out = np.zeros(segmentation.shape)
        for i in range(len(values)):
            out[segmentation == i] = values[i]
        return out

    # 可视化原图和Top3类别的SHAP值
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(16, 20),gridspec_kw={'bottom': 0.1})
    for i in range(5):
        img_orig = imgs_orig[i]
        segments_slic = segments_list[i]
        shap_values = shap_values_list[i]
        top_preds = top_preds_list[i]

        max_val = np.max(np.abs(shap_values[0]))

        axes[i, 0].imshow(img_orig)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Original Image\nLabel: {labels[i].item()}")

        from PIL import Image
        img_pil = Image.fromarray((img_orig * 255).astype(np.uint8)).convert('LA')

        # top3类别
        inds = top_preds[:3]
        for j, c in enumerate(inds):
            m = fill_segmentation(shap_values[0][:, c], segments_slic)
            axes[i, j + 1].imshow(img_pil, alpha=0.15)
            im = axes[i, j + 1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
            axes[i, j + 1].set_title(f"Class {c}")
            axes[i, j + 1].axis('off')

    cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value",
                      orientation="horizontal", aspect=60, pad=0.08)
    cb.outline.set_visible(False)
    plt.tight_layout()
    plt.show()
