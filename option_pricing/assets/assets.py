"""Classes representing underlying assets
"""
import abc
import numpy as np
from typing import NamedTuple, Literal


class SimulatedStates(NamedTuple):
    states: np.ndarray
    timeline: tuple[float]


class Asset(abc.ABC):
    def __repr__(self):
        varstring = ', '.join(f"{k} = {v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({varstring})"

    @abc.abstractmethod
    def simulate_states(self, timeline, n):
        """Simulate n states for the specified timeline"""


def simulate_wiener_process(n_periods, n_paths):
    return np.random.normal(0, 1, size=(n_paths, n_periods))


def simulate_correlated_wiener_process(wiener_process, correlation):
    n_paths, n_periods = wiener_process.shape
    temp_wiener_process = simulate_wiener_process(n_periods, n_paths)
    return correlation*wiener_process + np.sqrt(1 - correlation ** 2)*temp_wiener_process


def handle_timeline(timeline):
    if 0 not in timeline:
        timeline = np.concatenate(([0], timeline))
    return timeline


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

        return SimulatedStates(S, tuple(timeline))


class BrownianMotion(Asset):
    def __init__(self, initial_price, drift, volatility, dividend_yield=0):
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.dividend_yield = dividend_yield

    def calculate_states(self, timeline, wiener_process):
        dt = np.diff(timeline)

        wiener_process = np.concatenate([wiener_process, -wiener_process], axis=0)

        d1 = self.drift - self.dividend_yield - (self.volatility ** 2) / 2
        d2 = d1 * dt + self.volatility * np.sqrt(dt) * wiener_process
        states = np.hstack([np.full((len(wiener_process), 1), self.initial_price), np.exp(d2)]).cumprod(axis=1)

        return SimulatedStates(states, tuple(timeline))

    def simulate_states(self, timeline, n):
        timeline = handle_timeline(timeline)
        wiener_process = simulate_wiener_process(len(timeline)-1, int(n/2))
        return self.calculate_states(timeline, wiener_process)


class CIRProcess(Asset):
    r"""
    The Cox-Ingersoll-Ross one factor model.

    Args:
        initial_level (float):
            The starting point of the process.
        mean (float):
            The mean which the process will revert to, often denoted $b$ or $\theta$.
        speed_of_mean_reversion (float):
            The speed of adjustment to the mean, often denoted $a$ or $\kappa$.
        volatility (float):
            The volatility of the process, often denoted $\sigma$.
        negative_level_method:
            The method by which negative levels of the process are handled. Negative levels can occur due to the
            discretization of the process. If "reflecting" then if r<0 then r is set to -r. If absorbing then
            if r<0 then r is set to 0.
    """
    def __init__(self, initial_level, mean, speed_of_mean_reversion, volatility,
                 negative_level_method: Literal['reflecting', 'absorbing'] = "reflecting"):
        self.initial_level = initial_level
        self.mean = mean
        self.speed_of_mean_reversion = speed_of_mean_reversion
        self.volatility = volatility
        self.negative_level_method = negative_level_method

    def calculate_states(self, timeline, wiener_process):
        dt = np.diff(timeline)

        wiener_process = np.concatenate([wiener_process, -wiener_process], axis=0)

        d1 = self.speed_of_mean_reversion * dt
        d2 = np.sqrt(dt) * self.volatility * wiener_process

        levels = [np.repeat(self.initial_level, len(wiener_process))]
        for t in range(1, len(timeline)):
            current_level = levels[t-1] + d1[t-1]*(self.mean-levels[t-1]) + d2[:, t-1]*np.sqrt(levels[t-1])
            levels.append(self.handle_negative_states(current_level))

        levels = np.stack(levels, axis=1)

        return SimulatedStates(levels, tuple(timeline))

    def simulate_states(self, timeline, n):
        timeline = handle_timeline(timeline)
        wiener_process = simulate_wiener_process(len(timeline)-1, int(n/2))
        return self.calculate_states(timeline, wiener_process)

    def handle_negative_states(self, states):
        if self.negative_level_method == "reflecting":
            return np.abs(states)
        elif self.negative_level_method == "absorbing":
            return np.maximum(states, 0)


class HestonProcess(Asset):
    def __init__(self, initial_price, drift, correlation, initial_variance_level, variance_mean,
                 speed_of_variance_mean_reversion, vol_of_vol):
        self.variance = CIRProcess(initial_variance_level, variance_mean, speed_of_variance_mean_reversion, vol_of_vol)
        self.initial_price = initial_price
        self.drift = drift
        self.correlation = correlation

    def calculate_variance_states(self, timeline, wiener_process):
        variance = self.variance.calculate_states(timeline, wiener_process).states
        # The first column is the initial_variance_level which is just a parameter so it can be removed.
        return variance[:, 1:]

    def calculate_price_states(self, timeline, variance, wiener_process):
        prices = BrownianMotion(self.initial_price, self.drift, np.sqrt(variance))
        return prices.calculate_states(timeline, wiener_process)

    def simulate_states(self, timeline, n):
        timeline = handle_timeline(timeline)
        variance_wiener_process = simulate_wiener_process(len(timeline)-1, n)
        variance = self.calculate_variance_states(timeline, variance_wiener_process)
        price_wiener_process = simulate_correlated_wiener_process(variance_wiener_process, self.correlation)

        return self.calculate_price_states(timeline, variance, price_wiener_process)
