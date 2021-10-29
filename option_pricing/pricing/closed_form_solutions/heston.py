import logging

import numpy as np
from scipy import integrate


class HestonModel:
    def __init__(self, reversion_speed, correlation, vol_of_vol, long_term_price_var, risk_free_rate):
        """

        Args:
            reversion_speed: lambda
            correlation: rho
            vol_of_vol: eta
            long_term_price_var: vbar
            risk_free_rate: r

        Returns:

        """
        self.reversion_speed = reversion_speed
        self.correlation = correlation
        self.vol_of_vol = vol_of_vol
        self.long_term_price_var = long_term_price_var
        self.risk_free_rate = risk_free_rate

    def european_call_price(self, price, strike, variance, expiration):
        """

        Args:
            price (float): Price of underlying, S.
            strike (float): Strike price of option, K.
            variance (float): Volatility of underlying squared, v.
            expiration (float): Time to expiration, T-t.

        Returns:
            (float):

        """
        log_price = np.log(price*np.exp(self.risk_free_rate*expiration) / strike)
        return strike * (
                np.exp(log_price) * self.p_functions(1, log_price, variance, expiration)
                - self.p_functions(0, log_price, variance, expiration)
        )

    def p_functions(self, j, log_price, variance, expiration):
        def integrand(u):
            alpha = - (u ** 2 / 2) - 1j * u / 2 + 1j * j * u
            beta = (self.reversion_speed - self.correlation * self.vol_of_vol * j
                    - self.correlation * self.vol_of_vol * 1j * u)
            gamma = self.vol_of_vol ** 2 / 2

            discriminant = np.sqrt(beta ** 2 - 4 * alpha * gamma)
            r_plus = (beta + discriminant) / (self.vol_of_vol**2)
            r_minus = (beta - discriminant) / (self.vol_of_vol**2)

            c1 = r_minus * (1 - np.exp(-discriminant * expiration)) \
                / (1 - (r_minus / r_plus) * np.exp(-discriminant * expiration))

            c2 = self.reversion_speed * (
                r_minus * expiration
                - (2 / (self.vol_of_vol**2))
                * np.log((1 - (r_minus/r_plus) * np.exp(-discriminant * expiration)) / (1 - (r_minus/r_plus)))
            )

            characteristic_function = np.exp(c2*self.long_term_price_var + c1*variance + 1j*u*log_price)
            return np.real(characteristic_function / (1j*u))

        integration = integrate.quad(integrand, 0.0, np.inf)
        logging.debug(f"Integration message: {integration}")

        return 1/2 + integration[0] / np.pi

