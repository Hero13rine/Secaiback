import numpy as np
from matplotlib.colors import Colormap
from typing import Literal, cast

from shap import kmeans, Explanation
from shap.plots import colors
import matplotlib.pyplot as plt

def image_plot_no_orig_nobar(
        shap_values: Explanation | np.ndarray | list[np.ndarray],
        pixel_values: np.ndarray | None = None,
        labels: list[str] | np.ndarray | None = None,
        true_labels: list | None = None,
        width: int | None = 20,
        aspect: float | None = 0.2,
        hspace: float | Literal["auto"] | None = 0.2,
        labelpad: float | None = None,
        cmap: str | Colormap | None = colors.red_transparent_blue,
        vmax: float | None = None,
        show: bool | None = True,
):
    # 处理shap_values和pixel_values
    if isinstance(shap_values, Explanation):
        shap_exp: Explanation = shap_values
        if len(shap_exp.output_dims) == 1:
            shap_values = cast("list[np.ndarray]", [shap_exp.values[..., i] for i in range(shap_exp.values.shape[-1])])
        elif len(shap_exp.output_dims) == 0:
            shap_values = cast("list[np.ndarray]", [shap_exp.values])
        else:
            raise Exception("Number of outputs needs to have support added!! (probably a simple fix)")
        if pixel_values is None:
            pixel_values = cast("np.ndarray", shap_exp.data)
        if labels is None:
            labels = cast("list[str]", shap_exp.output_names)
    else:
        assert isinstance(pixel_values, np.ndarray), (
            "pixel_values必须是numpy数组或提供Explanation对象!"
        )

    if not isinstance(shap_values, list):
        shap_values = cast("list[np.ndarray]", [shap_values])

    if len(shap_values[0].shape) == 3:
        shap_values = [v.reshape(1, *v.shape) for v in shap_values]
        pixel_values = pixel_values.reshape(1, *pixel_values.shape)

    if labels is not None:
        if isinstance(labels, list):
            labels = np.array(labels).reshape(1, -1)

    label_kwargs = {} if labelpad is None else {"pad": labelpad}

    # 只绘制shap图，列数为shap_values数量
    fig_size = np.array([3 * len(shap_values), 2.5 * (pixel_values.shape[0] + 1)])
    if fig_size[0] > width:
        fig_size *= width / fig_size[0]

    fig, axes = plt.subplots(nrows=pixel_values.shape[0], ncols=len(shap_values), figsize=fig_size, squeeze=False)

    for row in range(pixel_values.shape[0]):
        x_curr = pixel_values[row].copy()

        # 生成灰度图背景
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])

        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = 0.2989 * x_curr[:, :, 0] + 0.5870 * x_curr[:, :, 1] + 0.1140 * x_curr[:, :, 2]
        elif len(x_curr.shape) == 3:
            x_curr_gray = x_curr.mean(2)
        else:
            x_curr_gray = x_curr

        # 计算最大绝对值（vmax）
        if len(shap_values[0][row].shape) == 2:
            abs_vals = np.stack([np.abs(shap_values[i][row]) for i in range(len(shap_values))], 0).flatten()
        else:
            abs_vals = np.stack([np.abs(shap_values[i][row].sum(-1)) for i in range(len(shap_values))], 0).flatten()

        max_val = np.nanpercentile(abs_vals, 99.9) if vmax is None else vmax

        for i in range(len(shap_values)):
            if labels is not None and row == 0:
                axes[row, i].set_title(labels[row, i], **label_kwargs)
            sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)
            axes[row, i].imshow(
                x_curr_gray, cmap=plt.get_cmap("gray"), alpha=0.15,
                extent=(-1, sv.shape[1], sv.shape[0], -1)
            )
            axes[row, i].imshow(sv, cmap=cmap, vmin=-max_val, vmax=max_val)
            axes[row, i].axis("off")

    if hspace == "auto":
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=hspace)

    # 不绘制颜色条
    if show:
        plt.show()