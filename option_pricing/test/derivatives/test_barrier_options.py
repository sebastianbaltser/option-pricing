import math

import numpy as np

from option_pricing.assets import BrownianMotion
from option_pricing.derivatives import BarrierCallDownAndOut
from option_pricing.pricing import calculate_derivative_price


class TestBarrierCallDownAndOut:
    def test_option_price(self):
        underlying = BrownianMotion(12.418, 0.01606, 0.4075, 0)
        option = BarrierCallDownAndOut(13.55, 12, 1 / 12, np.linspace(0, 1 / 12, num=10)[1:], underlying)

        result = calculate_derivative_price(1_000_000, option, 0.01606)[0]

        assert math.isclose(result, 0.1395, abs_tol=0.001)
