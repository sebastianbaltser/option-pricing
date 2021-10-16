import numpy as np


def calculate_option_price(n, option):
    """
    Approximate the price for the option specified in the option argument using Monte Carlo simulations.

    Args:
        n (int): The number of simulations used.
        option: Instance of an option-class

    Returns:
        (tuple[float, numpy.ndarray]):
            A tuple with the approximated option price and an array of simulated states of the underlying
    """
    expiration = option.timeline[-1]
    mu = option.underlying.drift

    underlying_states = option.underlying.simulate_states(option.timeline, n)

    payoffs = option.payoff(underlying_states)

    time_zero_discount = np.exp(-mu*expiration)
    present_value = time_zero_discount * payoffs

    present_value = np.sum(present_value)/n

    return present_value, underlying_states
