""""
    Binary bayes filter as a state estimator
    According to Probabilistic Robotics p. 86
    https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf
    used to estimate whether an object is dynamic or static
    requires an inverse model: e.g: P(x | z)
        probability that an object is dynamic -> state simple, measurement hard 
"""
import numpy as np

class BinaryBayes(object):
    """
        pxz is the probability that the state is true, given an observation z
        e.g. probability that the state is dynamic, given the observation
        pxz 0 
    """

    def __init__(self, pxz : np.ndarray = [0.7, 0.3], prior : float = 0.5) -> None:
        assert pxz.shape == (2,), "Probability must be a 2-vector!"
        assert prior < 1.0 and prior > 0.0, "Prior must be between 0. and 1. Fix!"
        self._pxz = pxz
        self._prior = prior
        self._lt = prior

    @property
    def pxz(self) -> float:
        return self._pxz
    
    @property
    def prior(self) -> float:
        return self._prior

    @property
    def lt(self) -> float:
        return self._lt

    def update(self, z : int):
        """
            Update function,
            see table 4.2 in probabilistic robotics
            z is the measurement. Either 0 (static) or 1 (dynamic)
            if we measure 0, pxz is the probability that it is actually static
            use this to incrementally update.
        """
        pxz = self.pxz[z]
        lt = self.lt + np.log(pxz / (1. - pxz)) - self.prior
        self._lt = lt
        return lt

    def get_belief(self):
        """
            Use this factor to normalize between 0 and 1 - turn into a probability again
            works as a scalar for V?s
        """
        return 1. - (1. / np.exp(self.lt))