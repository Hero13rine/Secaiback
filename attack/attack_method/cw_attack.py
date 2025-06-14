from art.attacks.evasion.carlini import CarliniL2Method, CarliniL0Method, CarliniLInfMethod

from attack.attack_factory import AttackFactory


@AttackFactory.register("cw2")
class CW2Attack:
    def __init__(self, estimator,
                 confidence=0.0,
                 targeted=False,
                 learning_rate=0.01,
                 binary_search_steps=10,
                 max_iter=10,
                 initial_const=0.01,
                 max_halving=5,
                 max_doubling=5,
                 batch_size=1,
                 verbose=True):
        self.attack = CarliniL2Method(
            classifier=estimator,
            confidence=confidence,
            targeted=targeted,
            learning_rate=learning_rate,
            binary_search_steps=binary_search_steps,
            max_iter=max_iter,
            initial_const=initial_const,
            max_halving=max_halving,
            max_doubling=max_doubling,
            batch_size=batch_size,
            verbose=verbose
        )

    def generate(self, x, y=None):
        return self.attack.generate(x=x, y=y)


@AttackFactory.register("cw0")
class CW0Attack:
    def __init__(self, estimator,
                 confidence=0.0,
                 targeted=False,
                 learning_rate=0.01,
                 binary_search_steps=10,
                 max_iter=10,
                 initial_const=0.01,
                 mask=None,
                 warm_start=True,
                 max_halving=5,
                 max_doubling=5,
                 batch_size=1,
                 verbose=True):
        self.attack = CarliniL0Method(
            classifier=estimator,
            confidence=confidence,
            targeted=targeted,
            learning_rate=learning_rate,
            binary_search_steps=binary_search_steps,
            max_iter=max_iter,
            initial_const=initial_const,
            mask=mask,
            warm_start=warm_start,
            max_halving=max_halving,
            max_doubling=max_doubling,
            batch_size=batch_size,
            verbose=verbose
        )

    def generate(self, x, y=None):
        return self.attack.generate(x=x, y=y)


@AttackFactory.register("cwInf")
class CWInfAttack:
    def __init__(self, estimator,
                 confidence=0.0,
                 targeted=False,
                 learning_rate=0.01,
                 max_iter=10,
                 decrease_factor=0.9,
                 initial_const=1e-5,
                 largest_const=20.0,
                 const_factor=2.0,
                 batch_size=1,
                 verbose=True):
        self.attack = CarliniLInfMethod(
            classifier=estimator,
            confidence=confidence,
            targeted=targeted,
            learning_rate=learning_rate,
            max_iter=max_iter,
            decrease_factor=decrease_factor,
            initial_const=initial_const,
            largest_const=largest_const,
            const_factor=const_factor,
            batch_size=batch_size,
            verbose=verbose
        )

    def generate(self, x, y=None):
        return self.attack.generate(x=x, y=y)