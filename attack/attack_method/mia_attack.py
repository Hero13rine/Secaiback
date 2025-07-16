from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from attack.attack_factory import AttackFactory  # 导入你的工厂类


@AttackFactory.register("mia")  # 注册为"mia"攻击类型，与配置文件method对应
class MIAAttack:
    """成员推理攻击实现类，适配AttackFactory工厂"""

    def __init__(
            self,
            estimator,  # 目标模型的估计器（由工厂传入）
            attack_model_type: str = "rf",  # 攻击模型类型
            targeted_fpr: float = 0.01,  # 目标FPR值
            batch_size: int = 16,  # 批处理大小（可选参数）
            input_type: str = "prediction",
            scaler_type: str = "standard",
            nn_model_epochs: int = 100,
            nn_model_learning_rate: float = 0.0001,
    ):
        # 初始化ART的黑盒成员推理攻击核心
        self.attack = MembershipInferenceBlackBox(
            estimator=estimator,
            input_type=input_type,
            attack_model_type=attack_model_type,
            scaler_type=scaler_type,
            nn_model_epochs=nn_model_epochs,
            nn_model_batch_size=batch_size,
            nn_model_learning_rate=nn_model_learning_rate,
        )
        self.targeted_fpr = targeted_fpr  # 用于后续评测的目标FPR
        self.is_trained = False  # 标记攻击模型是否已训练

    def train(self, x_member, y_member, x_non_member, y_non_member):
        """训练攻击模型（使用成员和非成员样本）"""
        self.attack.fit(
            x=x_member,  # 成员样本（目标模型训练过的数据）
            y=y_member,  # 成员样本标签
            test_x=x_non_member,  # 非成员样本（目标模型未训练过的数据）
            test_y=y_non_member  # 非成员样本标签
        )
        self.is_trained = True
        return self

    def infer(self, x, y=None, **kwargs):
        """执行成员推理攻击，返回预测结果（概率或标签）"""
        if not self.is_trained:
            raise RuntimeError("攻击模型未训练，请先调用train方法")
        return self.attack.infer(x=x, y=y, **kwargs)
