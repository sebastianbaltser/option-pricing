"""Classes representing underlying assets
"""
import numpy as np


class JumpDiffusion:
    def __init__(self, initial_price, riskfree_rate, volatility, jump_intensity, jump_size_mean, jump_size_variance):
        self.S0 = initial_price
        self.mu = riskfree_rate
        self.sigma = volatility
        self.lam = jump_intensity
        self.k = jump_size_mean
        self.delta = jump_size_variance

    def simulate_states(self, timeline, n):
        timeline = np.concatenate(([0], timeline))
        dt = np.diff(timeline)

        Z = np.random.normal(0, 1, size = (int(n/2), len(timeline)-1))
        Z = np.concatenate([Z, -Z], axis=0)

        # Simulate the number of jumps arriving in each timeline-interval for each path
        frequencies = np.repeat(np.array([self.lam * dt]), n, axis=0)
        num_arrivals = np.random.poisson(frequencies)
        # Given the number of jumps in each interval simulate the normal jump sizes
        # The mean and variance of the jump size depends on the number of jumps in the interval in question
        # If there is multiple jumps in one interval the total jump size is a sum of normal random variables
        means = (np.log(self.k+1) - (self.delta**2)/2) * num_arrivals
        std = self.delta * np.sqrt(num_arrivals)
        jumps = np.random.normal(0, 1, size=(int(n/2), len(timeline)-1))
        # Apply the antithetic variates method
        jumps = np.concatenate([jumps, -jumps], axis=0) * std + means
        
        # Calculate state matrix
        d1 = self.mu - (self.sigma**2) / 2 - self.lam * self.k
        d2 = d1*dt + self.sigma*np.sqrt(dt)*Z + jumps
        Sn = np.exp(d2)
        S0 = np.full((n, 1), self.S0)
        S = np.hstack([S0, Sn])
        S = S.cumprod(axis = 1)

        return S

    def __repr__(self):
        varstring = ', '.join(f"{k} = {v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({varstring})"


class BrownianMotion:
    def __init__(self, initial_price, riskfree_rate, volatility, dividend_yield=0):
        self.S0 = initial_price
        self.mu = riskfree_rate
        self.sigma = volatility
        self.a = dividend_yield

    def simulate_states(self, timeline, n):
        timeline = np.concatenate(([0], timeline))
        dt = np.diff(timeline)

        Z = np.random.normal(0, 1, size = (int(n/2), len(timeline)-1))
        # Apply the antithetic variates method
        Z = np.concatenate([Z, -Z], axis=0)

        # Calculate state matrix
        d1 = self.mu - self.a - (self.sigma**2) / 2
        d2 = d1*dt + self.sigma*np.sqrt(dt)*Z
        S = np.hstack([np.full((n, 1), self.S0), np.exp(d2)]).cumprod(axis = 1)

        return S

    def __repr__(self):
        varstring = ', '.join(f"{k} = {v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({varstring})"
