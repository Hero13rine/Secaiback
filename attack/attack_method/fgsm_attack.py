from art.attacks.evasion import FastGradientMethod

from attack.attack_factory import AttackFactory


@AttackFactory.register("fgsm")
class FGSMAttack:
    def __init__(self, estimator, eps=0.3):
        self.attack = FastGradientMethod(
            estimator=estimator,
            eps=eps
        )

    def generate(self, x):
        return self.attack.generate(x=x)