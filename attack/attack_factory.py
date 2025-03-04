class AttackFactory:
    _registry = {}

    @classmethod
    def create(cls, name, estimator, **kwargs):
        print("Registered attacks:", AttackFactory._registry.keys())
        attack_class = cls._registry.get(name.lower())
        if not attack_class:
            raise ValueError(f"Attack {name} not registered!")
        return attack_class(estimator, **kwargs)

    @classmethod
    def register(cls, name):
        def decorator(attack_class):
            cls._registry[name.lower()] = attack_class
            return attack_class
        return decorator