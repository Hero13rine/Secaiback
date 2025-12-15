import yaml


def load_config(config_path):
    # 1.加载配置
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
