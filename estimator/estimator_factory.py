import yaml
import inspect


class EstimatorFactory:
    _registry = {}  # 存储注册的估计器类
    _param_schemas = {}  # 存储参数模板

    @classmethod
    def load_schema(cls, framework: str, task: str):
        """加载参数校验模板"""
        schema_path = f"config/estimator/{framework}_{task}.yaml"
        with open(schema_path) as f:
            schema = yaml.safe_load(f)
            key = f"{framework}_{task}"
            cls._param_schemas[key] = schema

    @classmethod
    def create(cls, framework: str, task: str, ** kwargs):
        """
        创建估计器的智能入口
        :param framework: 框架名称 (pytorch/tensorflow)
        :param task: 任务类型 (classification/detection)
        """
        # 1. 加载参数模板
        schema_key = f"{framework}_{task}"
        if schema_key not in cls._param_schemas:
            cls.load_schema(framework, task)
        schema = cls._param_schemas[schema_key]

        # 2. 参数校验
        required_params = schema['parameters']['required']
        optional_params = schema['parameters']['optional']

        # 检查缺失的必要参数
        missing = [p for p in required_params if p not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        # 合并可选参数默认值
        final_params = {**optional_params, ** kwargs}

        # 3. 获取具体实现类
        impl_class = cls._registry.get(schema_key)
        if not impl_class:
            raise ValueError(f"No implementation for {schema_key}")

        # 4. 过滤无效参数
        sig = inspect.signature(impl_class.__init__)
        valid_params = {k: v for k, v in final_params.items() if k in sig.parameters}

        # 5. 实例化并返回
        return impl_class(**valid_params)

    @classmethod
    def register(cls, framework: str, task: str):
        """带参数模板的装饰器"""

        def decorator(impl_class):
            key = f"{framework}_{task}"
            cls._registry[key] = impl_class
            return impl_class

        return decorator