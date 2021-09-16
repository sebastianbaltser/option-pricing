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
    T = option.timeline[-1]
    mu = option.underlying.mu

    S = option.underlying.simulate_states(option.timeline, n)

    # Calculate payoff with the payoff-method of the corresponding option
    P = option.payoff(S)

    # Apply time-zero discounting
    V = np.exp(-mu*T) * P

    # Calculate the MC-estimate
    V = np.sum(V)/n

    return V, S
