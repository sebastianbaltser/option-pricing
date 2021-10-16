import numpy as np


def calc_optionprice(n, option):
    """
    Approximates optionprices for the option specified in the option argument using MC-simulations.

    Args:
        n (int): number of simulations in MC-simulations
        option: instance of an option-class

    Returns:
        (tuple[float, numpy.array]):
            A tuple with calculated option price and statematrix
    """
    expiration = option.timeline[-1]
    mu = option.underlying.mu

    underlying_states = option.underlying.simulate_states(option.timeline, n)

    # Calculate payoff with the payoff-method of the corresponding option
    payoffs = option.payoff(underlying_states)

    # Apply time-zero discounting
    present_value = np.exp(-mu*expiration) * payoffs

    # Calculate the MC-estimate
    present_value = np.sum(present_value)/n

    return present_value, underlying_states
