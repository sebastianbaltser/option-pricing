from abc import ABC

import numpy as np

from .abstract_bases import Option, Payoffs
from option_pricing.assets.assets import SimulatedStates


class BarrierOption(Option, ABC):
    """
    Args:
        barrier (float):
            The barrier of the option.
    """

    def __init__(self, strike, barrier, expiration, underlying):
        super().__init__(strike, expiration, underlying)
        self.barrier = barrier


class DiscreteBarrierOption(BarrierOption, ABC):
    """
    Args:
        observation_points:
            An array of points in time where the barrier is active, i.e. where the option can be knocked in or out.
    """
    def __init__(self, strike, barrier, expiration, observation_points, underlying):
        super().__init__(strike, barrier, expiration, underlying)
        self.observation_points = observation_points
        self.simulation_points = np.union1d(observation_points, [expiration])

    def simulate_underlying_states(self, n):
        return self.underlying.simulate_states(self.simulation_points, n)


def calculate_brownian_bridge_discount(states, barrier, underlying_volatility):
    """
    Uses the Brownian Bridge approach to calculate payoff-discount-factors by considering the probability
    of hitting the barrier between observation-points.
    """
    dt = np.diff(states.timeline)

    st = np.maximum(np.log(states.states[:, :-1] / barrier), 0)
    stp = np.maximum(np.log(states.states[:, 1:] / barrier), 0)

    p = np.exp(-2 * st * stp / (underlying_volatility ** 2 * dt))
    q = np.prod(1 - p, axis=1)

    return q


class BarrierCallDownAndOut(DiscreteBarrierOption):
    def payoff(self, states):
        """
        Calculates payoff given a matrix with underlying prices.
        Uses the Brownian Bridge approach to approximate the price of a continuous barrier option.

        Args:
            states (SimulatedStates):
        """

        payoffs = np.maximum(states.states[:, states.timeline.index(self.expiration)] - self.strike, 0)

        payoff_discounts = calculate_brownian_bridge_discount(states, self.barrier, self.underlying.volatility)
        payoffs *= payoff_discounts

        return Payoffs(payoffs.reshape(-1, 1), (self.expiration, ))
