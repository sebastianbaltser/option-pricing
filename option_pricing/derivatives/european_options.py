from abc import ABC

import numpy as np

from .abstract_bases import Option, Payoffs


class EuropeanOption(Option, ABC):
    def simulate_underlying_states(self, n):
        return self.underlying.simulate_states(np.array([self.expiration]), n)


class EuroCall(EuropeanOption):
    def payoff(self, states):
        payoffs = np.maximum(states.states[:, [states.timeline.index(self.expiration)]] - self.strike, 0)

        return Payoffs(payoffs, (self.expiration, ))


class EuroPut(EuropeanOption):
    def payoff(self, states):
        payoffs = np.maximum(self.strike - states.states[:, [states.timeline.index(self.expiration)]], 0)

        return Payoffs(payoffs, (self.expiration, ))
