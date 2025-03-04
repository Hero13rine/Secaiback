from art.attacks.evasion import ProjectedGradientDescent

from attack.attack_factory import AttackFactory


@AttackFactory.register("pgd")
class PGDAttack:
    def __init__(self, estimator, eps=0.3, max_iter=10):
        self.attack = ProjectedGradientDescent(
            estimator=estimator,
            eps=eps,
            max_iter=max_iter
        )

    def generate(self, x):
        return self.attack.generate(x=x)