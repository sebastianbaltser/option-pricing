import numpy as np
import scipy.stats


class Barrier:
    def __init__(self, strike, barrier, timeline, expiration, underlying, **kwargs):
        self.K = strike
        self.T = expiration
        self.timeline = np.array(timeline)
        self.B = barrier
        self.underlying = underlying

        self.k = kwargs.get("k", 10)

    def __repr__(self):
        return f"{self.__class__.__name__}(K = {self.K}, T = {self.T}, B = {self.B})"

    def __str__(self):
        return self.__repr__()


class BarrierDiscreteDownAndOut(Barrier):
    def payoff(self, S, adjoint_mode=False, adjoints=None):
        intrinsic_value = np.maximum(S[:, -1] - self.K, 0)
        barrier_discounting = 1/(1 + np.exp(-self.k * (np.min(S - self.B, axis=1))))
        payoff = intrinsic_value * barrier_discounting

        if not adjoint_mode:
            return payoff

        logistic_PDF = scipy.stats.logistic.pdf(np.min(S, axis = 1), loc = self.B, scale = 1/self.k)

        S_ = np.zeros(S.shape)
        S_[:, -1] = ((S[:, -1] > self.K) * 1/(1 + np.exp(-self.k * (np.min(S, axis = 1) - self.B)))) * adjoints["P_"]
        S_ += (np.maximum(S[:, -1] - self.K, 0) * logistic_PDF * (S < self.B).T).T * adjoints["P_"]
        adjoints["S_"] += S_
        adjoints["K_"] += -1 * (S[:, -1] > self.K) * 1/(1 + np.exp(-self.k * (np.min(S, axis = 1) - self.B))) * adjoints["P_"]


class BarrierDiscreteUpAndIn(Barrier):
    def payoff(self, S, adjoint_mode=False, adjoints=None):
        intrinsic_value = np.maximum(S[:, -1] - self.K, 0)
        barrier_discounting = 1/(1 + np.exp(-self.k * (np.max(S, axis=1) - self.B)))
        payoff = intrinsic_value * barrier_discounting

        if not adjoint_mode:
            return payoff

        logistic_PDF = scipy.stats.logistic.pdf(np.max(S, axis = 1), loc = self.B, scale = 1/self.k)

        S_ = np.zeros(S.shape)
        S_[:, -1] = ((S[:, -1] > self.K) * 1/(1 + np.exp(-self.k * (np.max(S, axis = 1) - self.B)))) * adjoints["P_"]
        S_ += (np.maximum(S[:, -1] - self.K, 0) * logistic_PDF * ~(S.T < np.max(S, axis = 1))).T * adjoints["P"]

        #(max(S[:, -1] - self.K, 0) * logistic_PDF * (S.T == np.max(S, axis = 1))).T * adjoints["P_"]
        adjoints["S_"] += S_
        adjoints["K_"] += -1 * (S[:, -1] > self.K) * 1/(1 + np.exp(-self.k * (np.max(S, axis = 1) - self.B))) * adjoints["P_"]


class BlackCoxDebt:
    def __init__(self, strike, timeline, barrier, underlying):
        self.K = strike
        self.timeline = np.array(timeline)
        # Construct barrier array from barrier-function
        self.B = np.array([barrier(t) for t in np.concatenate(([0], timeline))])
        self.underlying = underlying

    def payoff(self, S):
        # Determine states where the barrier was breached
        barrier_breached = S <= self.B
        # Determine tau, i.e. the first default time.
        # If there is no default tau is set to -1. In a boolean array np.argmax over axis 1
        # evaluates to the index of the first True-value in each row, i.e. the first default time.
        tau_idx = np.where(~barrier_breached.any(axis=1), -1, np.argmax(barrier_breached, axis=1))

        Bm = np.minimum(S[:, -1], self.K)

        # Calculate the payoff if the boundary is hit forward-discounted to time-T
        Bb = np.take_along_axis(S, tau_idx[:, None], axis=1).flatten()
        # Instead of tau being indexes take indices from the timeline to transform tau to timestamps
        tau = self.timeline.take(tau_idx-1)
        Bb *= np.exp(self.underlying.drift * (self.timeline[-1] - tau))

        # If the barrier was not breached the payoff is determined at maturity
        # otherwise the payoff is determined at time-tau
        P = np.where(tau_idx == -1, Bm, Bb)

        return P


class BarrierSimple:
    """
    Base-class for simple barrier options
    """
    def __init__(self, strike, expiration, underlying, barrier, **kwargs):
        self.K = strike
        self.T = expiration
        self.timeline = np.array([self.T])
        self.B = barrier
        self.underlying = underlying
            
        self.k = kwargs.get("k", 10) #Precision of the logistic function

    def __repr__(self):
        return f"{self.__class__.__name__}(K = {self.K}, T = {self.T}, B = {self.B})"

    def __str__(self):
        return self.__repr__()


class BarrierSimpleUpAndIn(BarrierSimple):
    def payoff(self, S, adjoint_mode=False, adjoints=None):
        """
        Calculates payoff given a matrix with underlying prices
        """
        ST = S[:, -1]
        P = np.maximum(ST - self.K, 0)
        P = P * 1/(1 + np.exp(-self.k * (ST - self.B)))

        if not adjoint_mode:
            return P

        logistic_PDF = scipy.stats.logistic.pdf(ST, loc = self.B, scale = 1/self.k)

        adjoints["K_"] += (1/(1 + np.exp(-self.k * (ST - self.B))) * -1 * (ST - self.K > 0)) * adjoints["P_"]
        adjoints["B_"] += -1 * logistic_PDF * np.maximum(ST - self.K, 0) * adjoints["P_"]
        #Only the last column in S affects payoff. All other adjoints are zeroed:
        S_ = np.zeros(S.shape)
        S_[:, -1] = (logistic_PDF * np.maximum(ST - self.K, 0) + (1/(1 + np.exp(-self.k * (ST - self.B))) * ((ST - self.K) > 0))) * adjoints["P_"]
        adjoints["S_"] += S_

        return None


class BarrierSimpleUpAndOut(BarrierSimple):
    def payoff(self, S, adjoint_mode=False, adjoints=None):
        """
        Calculates payoff given a matrix with underlying prices
        """
        ST = S[:, -1]
        P = np.maximum(ST - self.K, 0)
        P = P * 1/(1 + np.exp(self.k * (ST - self.B)))

        if not adjoint_mode:
            return P

        logistic_PDF = -1*scipy.stats.logistic.pdf(ST, loc = self.B, scale = 1/self.k)

        adjoints["K_"] += (1/(1 + np.exp(self.k * (ST - self.B))) * -1 * (ST-self.K > 0)) * adjoints["P_"]
        adjoints["B_"] += -1 * logistic_PDF * np.maximum(ST - self.K, 0) * adjoints["P_"]
        #Only the last column in S affects payoff. All other adjoints are zeroed:
        S_ = np.zeros(S.shape)
        S_[:, -1] = (logistic_PDF * np.maximum(ST - self.K, 0) + 1/(1 + np.exp(self.k * (ST - self.B))) * ((ST - self.K) > 0)) * adjoints["P_"]
        adjoints["S_"] += S_

        return None
