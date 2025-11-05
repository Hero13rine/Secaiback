import inspect
import os

import yaml


class AttackFactory:
    _registry = {}
    _param_schemas = {}  # 存储参数模板

    @classmethod
    def load_schema(cls, method: str):
        # 获取当前脚本所在的绝对路径（假设当前脚本在项目内的任意位置）
        current_script_path = os.path.abspath(__file__)
        # 项目根目录：假设根目录是当前脚本的上两级目录（根据实际目录结构调整，比如 `os.path.dirname(os.path.dirname(...))`）
        project_root = os.path.dirname(os.path.dirname(current_script_path))
        """加载参数校验模板"""
        schema_path = project_root + f"/config/attack/{method}.yaml"
        with open(schema_path, encoding='utf-8') as f:
            schema = yaml.safe_load(f)
            key = f"{method}"
            cls._param_schemas[key] = schema

    @classmethod
    def create(cls, estimator, config: dict):

        # 调试
        print("Registered attacks:", AttackFactory._registry.keys())

        # 1. 防御性检查基础配置
        if not config["method"]:
            raise ValueError("config必须包含attack-method字段")
        method = config["method"]

        # 2.动态加载参数模板
        schema_key = f"{method}"
        if schema_key not in cls._param_schemas:
            cls.load_schema(method)
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
        final_params = {**optional_params, **params_config}

        # 6. 获取具体实现类
        impl_class = cls._registry.get(schema_key)
        if not impl_class:
            raise ValueError(f"No implementation for {schema_key}")

        # 7. 过滤无效参数
        sig = inspect.signature(impl_class.__init__)
        valid_params = {k: v for k, v in final_params.items() if k in sig.parameters}

        # 8. 实例化并返回
        return impl_class(estimator, **valid_params)

    @classmethod
    def register(cls, name):
        def decorator(attack_class):
            cls._registry[name.lower()] = attack_class
            return attack_class
        return decorator