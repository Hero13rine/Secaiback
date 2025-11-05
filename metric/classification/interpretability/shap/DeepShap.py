import shap
import torch

from data.load_dataset import load_cifar
from model.load_model import load_model

import matplotlib
matplotlib.use('Agg')  # 使用无GUI的后端
import matplotlib.pyplot as plt

def DeepShap():
    # 加载模型和数据
    model=load_model("../../../cfz_train/utils/ResNet.py",
               "ResNet18",
               "../../cfz_train/utils/resnet18_cifar10.pt")
    dataloader = load_cifar()

    # 取出200张数据作为background
    background_list = []
    num_required = 200
    it = iter(dataloader)
    while len(background_list) * dataloader.batch_size < num_required:
        images, _ = next(it)
        background_list.append(images)
    background = torch.cat(background_list, dim=0)[:num_required].to(torch.float32)
    # 取出5张图像作为要解释的图像
    test_images, _ = next(it)
    test_images = test_images[:5].to(torch.float32)

    # 保留中间层的激活值
    def replace_relu_inplace(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.ReLU) and child.inplace:
                setattr(module, name, torch.nn.ReLU(inplace=False))
            else:
                replace_relu_inplace(child)
    replace_relu_inplace(model)

    model.eval()
    explainer = shap.DeepExplainer(model, background)
    shap_values, indexes = explainer.shap_values(test_images, ranked_outputs=3, check_additivity=False)

    # 把shap_values转换为list，并转换图像格式 (N, C, H, W) -> (N, H, W, C)
    shap_values = [shap_values[..., i].transpose(0, 2, 3, 1) for i in range(shap_values.shape[-1])]
    images = test_images.permute(0, 2, 3, 1).numpy()

    labels = [[f"Class: {label}" for label in sample] for sample in indexes.tolist()]
    shap.image_plot(shap_values, images, labels=labels)
    plt.gcf().savefig("./result/deepshap_output.png", dpi=300, bbox_inches="tight")