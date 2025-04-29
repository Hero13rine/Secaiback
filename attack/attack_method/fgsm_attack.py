from art.attacks.evasion import FastGradientMethod

from attack.attack_factory import AttackFactory


@AttackFactory.register("fgsm")
class FGSMAttack:
    def __init__(self, estimator,
                 norm,
                 eps,
                 eps_step,
                 targeted,
                 num_random_init,
                 batch_size,
                 minimal,
                 summary_writer):
        self.attack = FastGradientMethod(
            estimator=estimator,
            eps=eps,
            norm=norm,
            eps_step=eps_step,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            minimal=minimal,
            summary_writer=summary_writer
        )

    def generate(self, x):
        return self.attack.generate(x=x)
