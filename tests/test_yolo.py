"""
测试yolo的评估流程 - 完整修正版本

修正重点：
1. 确保模型进入评估模式 (model.eval())。
2. 修复 YOLO 真实标签 (N x 6) 到评估工具字典格式的转换逻辑，特别处理 batch_id 分割。
3. 确保模型和数据在正确的设备上 (GPU/CPU)。
"""

import torch

# --- 假设导入的依赖模块（需要确保这些模块在您的环境中存在） ---
# 请根据您的实际文件结构确保以下导入路径正确：
from estimator import EstimatorFactory
from utils.load_config import load_config
from metric.object_detection.basic.detection import cal_object_detection
from model.load_yolo_model import load_yolo_model
from data.load_yolo_dataset import load_dior # 您的 DIOR 数据加载器
from utils.yolo import convert_yolo_loader_to_dict_format

YOLOV7_UTILS_AVAILABLE = True

def main():
    # 0、定义关键参数
    evaluation_path = "../config/user/model_pytorch_det_yolo.yaml"
    
    # 1.加载配置文件
    user_config = load_config(evaluation_path)
    model_instantiation_config = user_config["model"]["instantiation"]
    model_estimator_config = user_config["model"]["estimator"]
    evaluation_config = user_config["evaluation"]
    print("进度: 配置文件已加载完毕")

    # 2.初始化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_yolo_model(
        weight_path=model_instantiation_config["weight_path"],
        model_path=model_instantiation_config.get("model_path"),
        device=device
    )
    # --- 关键修正：设置模型为评估模式并移动到设备 ---
    model.eval() 
    model.to(device)
    print("进度: 模型初始化完成")

    # 3.获取优化器和损失函数 (此处保持不变)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)
    loss = None

    # 4.生成估计器
    estimator = EstimatorFactory.create(
        model=model,
        loss=loss,
        optimizer=optimizer,
        config=model_estimator_config
    )
    # 注意：模型已经在 load_yolo_model 时移动到正确的设备上了
    print("进度: 估计器已生成")

    # 5.加载数据
    data_yaml_path = "/wkm/data/dior/dior.yaml"
    if "data" in user_config and "data_yaml_path" in user_config["data"]:
        data_yaml_path = user_config["data"]["data_yaml_path"]
    
    yolo_loader = load_dior(
        data_yaml_path=data_yaml_path,
        split="val",
        batch_size=8, # 恢复 batch_size，但如果之前有 DataLoader 线程错误，请改为 1 或 0
        img_size=640,
        augment=False 
    )
    print("进度: 数据集已加载")

    # 6.根据传入的评测类型进行评测
    print("开始执行检测流程测试...")
    test_loader = convert_yolo_loader_to_dict_format(yolo_loader)
    
    # 执行评估
    cal_object_detection(estimator, test_loader, evaluation_config)
    print("检测流程测试完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
