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
    underlying_states = option.simulate_underlying_states(n)

    present_values = option.value(underlying_states)

    present_value = np.sum(present_values)/n

    return present_value, underlying_states
