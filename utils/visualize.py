import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def denormalize(image, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)):
    # 反归一化处理
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image = image * np.array(std) + np.array(mean)
    return np.clip(image, 0, 1)


def plot_samples(clean, adv, labels, n=5):
    plt.figure(figsize=(15, 3))
    for i in range(n):
        # 原始样本
        plt.subplot(2, n, i + 1)
        img_clean = denormalize(clean[i])
        plt.imshow(img_clean)
        plt.title(f"Clean: {labels[i]}")
        plt.axis('off')

        # 对抗样本
        plt.subplot(2, n, n + i + 1)
        img_adv = denormalize(adv[i])
        plt.imshow(img_adv)
        plt.title("Adversarial")
        plt.axis('off')
    plt.tight_layout()
    plt.show()