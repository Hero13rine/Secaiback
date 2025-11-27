from art.attacks.evasion import ProjectedGradientDescent

from attack.attack_factory import AttackFactory


@AttackFactory.register("pgd")
class PGDAttack:
    def __init__(
        self,
        estimator,
        norm,
        eps,
        eps_step,
        decay,
        max_iter,
        targeted,
        num_random_init,
        batch_size,
        random_eps,
        summary_writer,
        verbose,
        steps=None,
    ):
        if steps is not None:
            max_iter = steps
        self.attack = ProjectedGradientDescent(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            decay=decay,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps,
            summary_writer=summary_writer,
            verbose=verbose
        )

    def generate(self, x):
        return self.attack.generate(x=x)
