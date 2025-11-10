# 任务分配后端评测对接技术文档

## 1. 总体架构

SecAI 评测后端在 Kubernetes Pod 中运行。每个 Pod 通过环境变量和主机名获取评测上下文，并按照统一流程执行：加载配置 → 初始化模型与估计器 → 按评测类型执行指标计算 → 将日志、结果、状态回写给上一层后端。该流程在 `eva_start.py` 的 `main` 函数中集中实现，适用于所有分类模型评测入口。【F:eva_start.py†L29-L89】

## 2. Pod 启动时的上下文约定

### 2.1 Pod 命名解析
- Pod 主机名 `HOSTNAME` 使用 `用户ID-模型ID-评测维度` 的格式，例如 `1242343443-1880539772613976065-basic`。
- 程序在启动时通过 `os.getenv('HOSTNAME')` 获取主机名，并拆分出 `user_id`、`model_id` 与 `evaluation_type`（basic、robustness、fairness 等）。【F:eva_start.py†L31-L38】

### 2.2 评测配置路径
- 所有评测 Pod 读取固定路径 `/app/userData/modelData/evaluationConfigs/evaluationConfig.yaml`。
- 配置文件会被解析为 `model`、`estimator`、`evaluation` 三个顶级键；具体字段示例见 `config/user/model_pytorch_cls.yaml`。【F:eva_start.py†L39-L45】【F:config/user/model_pytorch_cls.yaml†L1-L22】

### 2.3 必须注入的环境变量
所有与上一层后端通信所需的信息通过环境变量注入，`ResultSender` 会在模块加载时读取：【F:utils/SecAISender.py†L7-L17】

| 变量名 | 作用 |
| --- | --- |
| `modelId` | 当前模型唯一标识，用于写回结果 |
| `containerName` | Pod 名称，日志追踪时回传 |
| `evaluateDimension` | 评测维度（如 classification） |
| `evaluateMetric` | 评测指标集名称（如 robustness） |
| `evaluationType` | 评测类型，和主机名中的 `evaluation_type` 对齐 |
| `logUrl` | 写日志的 HTTP 接口地址 |
| `resultUrl` | 写评测结果的 HTTP 接口前缀（`/status` 共用此前缀） |
| `resultColumn` | 数据库结果列标识，用于后端解析 |
| `resultPath` |（可选）用于生成图像类文件的落盘路径，安全指标模块会检查 |

> **注意**：`ResultSender` 在导入时立即读取环境变量，因此容器启动前必须完成所有变量的注入。

## 3. 评测流程详解

### 3.1 配置加载与模型初始化
1. `load_config` 读取 YAML 并返回 Python 字典。【F:eva_start.py†L41-L45】【F:method/load_config.py†L4-L8】
2. `model.instantiation` 段提供模型文件路径、权重与构造参数；`model.estimator` 段提供 ART 估计器配置。
3. `load_model` 动态导入模型并加载权重，随后使用 `EstimatorFactory.create` 结合损失函数、优化器组装估计器对象。【F:eva_start.py†L48-L64】

### 3.2 数据加载
- 默认入口调用 `/app/systemData/database_code/load_dataset.load_data`，该函数应返回测试数据 `DataLoader`。在本地示例中，`data/load_dataset.py` 展示了 CIFAR-10/MNIST 的实现，可作为实现参考。【F:eva_start.py†L66-L68】【F:data/load_dataset.py†L1-L49】

### 3.3 评测类型到指标模块的映射
`evaluation_type` 决定调用的指标集合：【F:eva_start.py†L70-L87】
- `basic` → `metric/classification/basic/basic.py::cal_basic`
- `robustness` → `metric/classification/robustness/evaluate_robustness.py::evaluation_robustness`
- `interpretability` → `metric/classification/interpretability/shap/GradientShap.py::GradientShap`
- `safety` → `metric/classification/safety/membershipinference/evaluate_mia.py::evaluate_mia`
- `generalization` → `metric/classification/generalization/generalization.py::evaluate_generalization`
- `fairness` → `metric/classification/fairness/fairness_metrics.py::calculate_fairness_metrics`

各模块内部都会在关键阶段写日志，并在成功/失败时发送状态，确保上一层后端能够实时追踪进度。

## 4. 评测结果采集与回传

### 4.1 日志写入
- 使用 `ResultSender.send_log(message_key, message_value)` 将阶段性进度或错误推送到 `logUrl`。
- 请求体包含模型、容器和指标上下文，便于后端进行多维度检索。【F:utils/SecAISender.py†L45-L65】

### 4.2 结果写入
- 使用 `ResultSender.send_result(key, value)`，参数以键值对的形式传入，可一次传多个键值对。
- 方法内部将字典序列化后 POST 到 `resultUrl`，请求体包含 `modelId`、`resultColumn` 以及结果字典。【F:utils/SecAISender.py†L19-L43】
- 所有数值会被转换为字符串；若传入字典，将以 JSON 字符串形式发送。

### 4.3 状态写入
- 评测结束后调用 `ResultSender.send_status(status)` 将最终状态写回 `resultUrl/status`，`status` 取值为“成功”或“失败”。【F:utils/SecAISender.py†L67-L86】
- 建议在每个评测模块的 `try/except` 结构中分别发送成功/失败状态，示例可参考基础指标实现。【F:metric/classification/basic/basic.py†L7-L72】

## 5. 各指标模块的回传要点

### 5.1 基础性能 (`basic`)
- 采集 `accuracy`、`precision`、`recall`、`f1score` 及逐类统计，并发送混淆矩阵。
- 通过 `classification_report` 衍生逐类指标，确保结果以 JSON 字符串发送。【F:metric/classification/basic/basic.py†L26-L65】

### 5.2 鲁棒性 (`robustness`)
- 评估流程分为对抗攻击（多 `eps` FGSM）与常规扰动两部分。
- 对抗攻击：`evaluate_robustness_adv_all` 会针对每个 `eps` 发送 `advacc`, `adverr`, `actc`, `acac` 等指标，并附加平均值键 `*_avg`。【F:metric/classification/robustness/evaluate_robustness.py†L21-L118】【F:metric/classification/robustness/evaluate_robustness.py†L137-L169】
- 扰动攻击：循环所有扰动算子与严重度，将累计指标通过 `ResultSender` 写回，例如 `mCE`、`RmCE`。【F:metric/classification/robustness/evaluate_robustness.py†L190-L252】
- 模块在异常时会发送失败状态与错误日志，以便后端感知异常。【F:metric/classification/robustness/evaluate_robustness.py†L21-L37】

### 5.3 安全性 (`safety` / 成员推理)
- 需要同时加载训练与测试数据，用于生成成员推理攻击样本。
- 结果包含 `auc`、`attack_average_precision`、`tpr_at_fpr`、`threshold`、`actual_fpr` 等；若配置包含 `roc_curve`，会根据 `resultPath` 写出图像并返回文件路径。【F:metric/classification/safety/membershipinference/evaluate_mia.py†L61-L206】

### 5.4 泛化性 (`generalization`)
- 计算 MSP、预测熵、Rademacher 复杂度等指标并写回，同时输出信息级日志。【F:metric/classification/generalization/generalization.py†L16-L88】

### 5.5 可解释性 (`interpretability`)
- `GradientShap` 会输出 SHAP 值并将可视化结果保存到共享存储路径，然后通过 `send_result` 和 `send_status` 回写。【F:metric/classification/interpretability/shap/GradientShap.py†L18-L85】

### 5.6 公平性 (`fairness`)
- 依赖外部传入的敏感属性函数，计算 SPD、DIR、EOD、AOD、Consistency 等指标，全部以单独键写回。【F:metric/classification/fairness/fairness_metrics.py†L3-L76】

## 6. 与上一层后端的协作建议

1. **统一上下文**：确保任务分配服务下发的 `evaluationConfig.yaml` 与环境变量一致，例如 `evaluation_type` 与配置 `evaluation` 字段匹配，否则 Pod 会加载到错误的指标集。
2. **接口幂等性**：`ResultSender` 可能在异常重试中重复发送，上一层接口需保证幂等，建议按 `modelId`+`resultColumn` 进行去重。
3. **日志落盘**：若需持久化 Pod 内部日志，可在 `send_log` 的接收端附加时间戳与序列号；模块已经对关键阶段提供日志调用。
4. **状态同步**：Pod 在成功与失败时都会调用 `send_status`，任务分配服务应监听该接口以更新任务生命周期。
5. **文件输出**：对于需要回传文件的指标（如 SHAP 可视化、MIA ROC 曲线），需提前挂载共享存储并在环境变量中提供 `resultPath`，以便后端据此读取实体文件。

## 7. 调试与扩展

- 在本地调试时，可设置 `logUrl`/`resultUrl` 为 mock 服务（例如 HTTP server），并通过环境变量覆盖默认值。
- 新增评测类型时，需在 `eva_start.main` 添加分支，并在新模块中遵循 `ResultSender` 的日志、结果、状态写回规范。
- 若需要支持多模型或多数据集，只需在配置文件中调整 `model` 与 `evaluation` 字段，无需改动入口逻辑。

通过遵循上述约定，任务分配后端可以稳定地获取 Pod 产生的评测信息，并将结果统一回写到核心调度平台。
