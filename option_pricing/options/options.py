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
        P = np.maximum(S[:, -1] - self.K, 0) #'Intrinsic value'
        P *= 1/(1 + np.exp(-self.k * (np.min(S - self.B, axis = 1)))) #Zero payoff if barrier was hit

        if not adjoint_mode:
            return P

        logistic_PDF = scipy.stats.logistic.pdf(np.min(S, axis = 1), loc = self.B, scale = 1/self.k)

        S_ = np.zeros(S.shape)
        S_[:, -1] = ((S[:, -1] > self.K) * 1/(1 + np.exp(-self.k * (np.min(S, axis = 1) - self.B)))) * adjoints["P_"]
        S_ += (np.maximum(S[:, -1] - self.K, 0) * logistic_PDF * (S < self.B).T).T * adjoints["P_"]
        adjoints["S_"] += S_
        adjoints["K_"] += -1 * (S[:, -1] > self.K) * 1/(1 + np.exp(-self.k * (np.min(S, axis = 1) - self.B))) * adjoints["P_"]


class BarrierDiscreteUpAndIn(Barrier):
    def payoff(self, S, adjoint_mode=False, adjoints=None):
        P = np.maximum(S[:, -1] - self.K, 0) #'Intrinsic value'
        P *= 1/(1 + np.exp(-self.k * (np.max(S, axis = 1) - self.B))) #Zero payoff if barrier was hit

        if not adjoint_mode:
            return P

        logistic_PDF = scipy.stats.logistic.pdf(np.max(S, axis = 1), loc = self.B, scale = 1/self.k)

        S_ = np.zeros(S.shape)
        S_[:, -1] = ((S[:, -1] > self.K) * 1/(1 + np.exp(-self.k * (np.max(S, axis = 1) - self.B)))) * adjoints["P_"]
        S_ += (np.maximum(S[:, -1] - self.K, 0) * logistic_PDF * ~(S.T < np.max(S, axis = 1))).T * adjoints["P"]

        #(max(S[:, -1] - self.K, 0) * logistic_PDF * (S.T == np.max(S, axis = 1))).T * adjoints["P_"]
        adjoints["S_"] += S_
        adjoints["K_"] += -1 * (S[:, -1] > self.K) * 1/(1 + np.exp(-self.k * (np.max(S, axis = 1) - self.B))) * adjoints["P_"]


class BarrierDownAndOut(Barrier):
    def payoff(self, S, adjoint_mode=False, adjoints=None):
        """
        Calculates payoff given a matrix with underlying prices.
        Uses the Brownian Bridge approach to approximate the price of a continuous barrier option.
        """
        sigma = self.underlying.volatility
        dt = np.diff(self.timeline, prepend = 0)
        
        #To avoid recalculation st and stp is defined:
        st = np.maximum(np.log(S[:, :-1]/self.B), 0) 
        stp = np.maximum(np.log(S[:, 1:]/self.B), 0)

        p = np.exp(-2 * st * stp / (sigma**2 * dt))
        q = np.prod(1 - p, axis = 1)
        P = np.maximum(S[:, -1] - self.K, 0) * q

        if not adjoint_mode:
            return P

        S_ = np.zeros(S.shape)
        S_[:, -1] = (S[:, -1] > self.K) * q * adjoints["P_"]
        adjoints["S_"] += S_
        adjoints["K_"] += -1 * (S[:, -1] > self.K) * q * adjoints["P_"]
        adjoints["q_"] += np.maximum(S[:, -1] - self.K, 0) * adjoints["P_"]

        p_ = np.zeros(p.shape)
        for i in range(p.shape[1]):
            #Product of every column except column i:
            p_[:, i] = -1 * (1 - np.hstack([p[:, :i], p[:, i+1:]])).prod(axis = 1) * adjoints["q_"]
        adjoints["p_"] += p_

        lnSBI = (np.log(S/self.B) > 0) / S #To avoid recalculation
        adjoints["S_"][:, :-1] += -2 * stp * lnSBI[:, :-1] / (sigma**2 * dt) * p * adjoints["p_"]
        adjoints["S_"][:, 1:] += -2 * st * lnSBI[:, 1:] / (sigma**2 * dt) * p * adjoints["p_"]
        adjoints["sigma_"] += np.sum((4 * st * stp / (sigma**3 * dt)) * p * adjoints["p_"], axis = 1)
        adjoints["dt_"] += 2 * st * stp / (sigma**2 * dt**2) * p * adjoints["p_"]


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


class Euro():
    """
    Base-class for european options.
    """
    def __init__(self, K, T, underlying):
        self.K = K
        self.T = T
        self.timeline = np.array([T])
        self.underlying = underlying

    def __repr__(self):
        return f"{self.__class__.__name__}(K = {self.K}, T = {self.T})"

    def __str__(self):
        return self.__repr__()


class EuroCall(Euro):
    def payoff(self, S, adjoint_mode=False, adjoints=None):
        """
        Calculates payoff given a matrix with underlying prices
        """
        
        P = np.maximum(S[:, -1] - self.K, 0)

        if not adjoint_mode:
            return P

        adjoints["K_"] += -1*(P > 0) * adjoints["P_"]
        #Only the last column in S affects payoff. All other adjoints are zeroed:
        S_ = np.zeros(S.shape)
        S_[:, -1] = 1*(P > 0) * adjoints["P_"]
        adjoints["S_"] += S_

        return None


class EuroPut(Euro):
    def payoff(self, S, adjoint_mode=False, adjoints=None):
        """
        Calculates payoff given a matrix with underlying prices
        """
        
        P = np.maximum(self.K - S[:, -1], 0)

        if not adjoint_mode:
            return P

        adjoints["K_"] += -1*(P < 0) * adjoints["P_"]
        #Only the last column in S affects payoff. All other adjoints are zeroed:
        S_ = np.zeros(S.shape)
        S_[:, -1] = 1*(P < 0) * adjoints["P_"]
        adjoints["S_"] += S_

        return None
