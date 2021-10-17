"""Classes representing underlying assets
"""
import abc
import numpy as np


class Asset(abc.ABC):
    def __repr__(self):
        varstring = ', '.join(f"{k} = {v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({varstring})"

    @abc.abstractmethod
    def simulate_states(self, timeline, n):
        """Simulate n states for the specified timeline"""


class JumpDiffusion(Asset):
    def __init__(self, initial_price, drift, volatility, jump_intensity, jump_size_mean, jump_size_variance):
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.jump_intensity = jump_intensity
        self.jump_size_mean = jump_size_mean
        self.jump_size_variance = jump_size_variance

    def simulate_states(self, timeline, n):
        timeline = np.concatenate(([0], timeline))
        dt = np.diff(timeline)

        Z = np.random.normal(0, 1, size = (int(n/2), len(timeline)-1))
        Z = np.concatenate([Z, -Z], axis=0)

        # Simulate the number of jumps arriving in each timeline-interval for each path
        frequencies = np.repeat(np.array([self.jump_intensity * dt]), n, axis=0)
        num_arrivals = np.random.poisson(frequencies)
        # Given the number of jumps in each interval simulate the normal jump sizes
        # The mean and variance of the jump size depends on the number of jumps in the interval in question
        # If there is multiple jumps in one interval the total jump size is a sum of normal random variables
        means = (np.log(self.jump_size_mean + 1) - (self.jump_size_variance ** 2) / 2) * num_arrivals
        std = self.jump_size_variance * np.sqrt(num_arrivals)
        jumps = np.random.normal(0, 1, size=(int(n/2), len(timeline)-1))
        # Apply the antithetic variates method
        jumps = np.concatenate([jumps, -jumps], axis=0) * std + means
        
        # Calculate state matrix
        d1 = self.drift - (self.volatility ** 2) / 2 - self.jump_intensity * self.jump_size_mean
        d2 = d1 * dt + self.volatility * np.sqrt(dt) * Z + jumps
        Sn = np.exp(d2)
        S0 = np.full((n, 1), self.initial_price)
        S = np.hstack([S0, Sn])
        S = S.cumprod(axis = 1)

        return S


class BrownianMotion(Asset):
    def __init__(self, initial_price, drift, volatility, dividend_yield=0):
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.dividend_yield = dividend_yield

    def simulate_states(self, timeline, n):
        timeline = np.concatenate(([0], timeline))
        dt = np.diff(timeline)

        Z = np.random.normal(0, 1, size = (int(n/2), len(timeline)-1))
        # Apply the antithetic variates method
        Z = np.concatenate([Z, -Z], axis=0)

        # Calculate state matrix
        d1 = self.drift - self.dividend_yield - (self.volatility ** 2) / 2
        d2 = d1 * dt + self.volatility * np.sqrt(dt) * Z
        S = np.hstack([np.full((n, 1), self.initial_price), np.exp(d2)]).cumprod(axis = 1)

        return S
