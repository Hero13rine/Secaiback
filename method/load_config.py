import yaml


def load_config(config_path):
    # 1.加载配置
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config
