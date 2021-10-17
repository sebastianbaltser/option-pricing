import abc

from option_pricing.assets.assets import Asset


class Derivative(abc.ABC):
    """
    A contract that bases it's value on an underlying asset.

    Args:
        underlying (Asset):
            The underlying asset.
    """
    def __init__(self, underlying):
        self.underlying = underlying

    @abc.abstractmethod
    def value(self, states):
        """Calculates the time zero value of the derivative given the states"""


class Option(Derivative):
    """
    A contract that allows the owner to buy or sell the underlying asset at the strike on or before the expiration.

    Args:
        strike (float):
            The agreed on price that the option owner can buy or sell at.
        expiration (float):
            The expiration date of the option.
        underlying (Asset):
            The underlying asset that the owner can buy or sell.
    """
    def __init__(self, strike, expiration, underlying):
        super().__init__(underlying)
        self.strike = strike
        self.expiration = expiration

    @abc.abstractmethod
    def payoff(self, states):
        """Calculate the option payoff from underlying states."""
