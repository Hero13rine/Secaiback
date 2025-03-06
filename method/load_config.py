import yaml


def load_config(model_config_path, attack_config_path):
    # 1.加载配置
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)
    with open(attack_config_path) as f:
        attack_config = yaml.safe_load(f)

    return model_config, attack_config


def load_system_config(config_path):
    # 1.加载配置
    with open(config_path) as f:
        system_config = yaml.safe_load(f)

    return system_config
