from abc import ABC

import numpy as np

from .abstract_bases import Option


class EuropeanOption(Option, ABC):
    def value(self, states):
        payoffs = self.payoff(states)

        time_zero_discount = np.exp(-self.underlying.drift * self.expiration)
        present_value = time_zero_discount * payoffs

        return present_value

    def simulate_underlying_states(self, n):
        return self.underlying.simulate_states(np.array([self.expiration]), n)


class EuroCall(EuropeanOption):
    def payoff(self, states):
        payoffs = np.maximum(states[:, -1] - self.strike, 0)

        return payoffs


class EuroPut(EuropeanOption):
    def payoff(self, states):
        payoffs = np.maximum(self.strike - states[:, -1], 0)

        return payoffs
