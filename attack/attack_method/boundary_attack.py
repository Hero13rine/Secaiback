from art.attacks.evasion import BoundaryAttack
from attack.attack_factory import AttackFactory


@AttackFactory.register("boundary")
class MyBoundaryAttack:
    def __init__(self, estimator,
                 targeted=True,
                 delta=0.01,
                 epsilon=0.01,
                 step_adapt=0.667,
                 max_iter=5000,
                 num_trial=25,
                 sample_size=20,
                 init_size=100,
                 min_epsilon=0.0,
                 batch_size=64,
                 verbose=True):
        self.attack = BoundaryAttack(
            estimator=estimator,
            targeted=targeted,
            delta=delta,
            epsilon=epsilon,
            step_adapt=step_adapt,
            max_iter=max_iter,
            num_trial=num_trial,
            sample_size=sample_size,
            init_size=init_size,
            min_epsilon=min_epsilon,
            batch_size=batch_size,
            verbose=verbose
        )

    def generate(self, x, y=None, **kwargs):
        return self.attack.generate(x=x, y=y, **kwargs)