import inspect
import os

import yaml


class EstimatorFactory:
    _registry = {}  # 存储注册的估计器类
    _param_schemas = {}  # 存储参数模板

    @classmethod
    def load_schema(cls, framework: str, task: str):
        # 获取当前脚本所在的绝对路径（假设当前脚本在项目内的任意位置）
        current_script_path = os.path.abspath(__file__)
        # 项目根目录：假设根目录是当前脚本的上两级目录（根据实际目录结构调整，比如 `os.path.dirname(os.path.dirname(...))`）
        project_root = os.path.dirname(os.path.dirname(current_script_path))
        """加载参数校验模板"""
        schema_path = project_root + f"/config/estimator/{framework}/{framework}_{task}.yaml"
        with open(schema_path) as f:
            schema = yaml.safe_load(f)
            key = f"{framework}_{task}"
            cls._param_schemas[key] = schema

    @classmethod
    def create(cls, model, loss, optimizer, config: dict):
        """
        创建估计器的智能入口
        model: 用户模型实例
        loss: 损失函数对象
        optimizer: 优化器实例
        config: 包含框架配置的字典，必须含framework/task字段
        """
        # 1. 防御性检查基础配置
        framework = config["framework"]
        task = config["task"]
        if not framework or not task:
            raise ValueError("config必须包含framework和task字段")

        # 2.动态加载参数模板
        schema_key = f"{framework}_{task}"
        if schema_key not in cls._param_schemas:
            cls.load_schema(framework, task)
        schema = cls._param_schemas[schema_key]

        # 3. 参数校验
        required_params = schema['parameters']['required']
        optional_params = schema['parameters']['optional']

        # 4. 检查缺失的必要参数
        params_config = config["parameters"]
        if required_params is not None:
            missing = [p for p in required_params if p not in params_config]
            if missing:
                raise ValueError(f"Missing required parameters: {missing}")

        # 5. 合并可选参数默认值
        final_params = {**optional_params, ** params_config}

        # 6. 获取具体实现类
        impl_class = cls._registry.get(schema_key)
        if not impl_class:
            raise ValueError(f"No implementation for {schema_key}")

        # 7. 过滤无效参数
        sig = inspect.signature(impl_class.__init__)
        valid_params = {k: v for k, v in final_params.items() if k in sig.parameters}

        # 8. 实例化并返回
        return impl_class(model, optimizer, loss, **valid_params)

    @classmethod
    def register(cls, framework: str, task: str):
        """带参数模板的装饰器"""

        def decorator(impl_class):
            key = f"{framework}_{task}"
            cls._registry[key] = impl_class
            return impl_class

        return decorator
