from option_pricing.pricing.closed_form_solutions import heston


class TestHestonModel:
    def test_call_option_price(self):
        model = heston.HestonModel(reversion_speed=0.00657, correlation=-0.00198, vol_of_vol=0.000509,
                                   long_term_price_var=0.0000647, risk_free_rate=0.000640)

        model.european_call_price(425.73, 395, 0.00723**2, 24/(365/12*3))
        model.european_call_price(425.73, 380, 0.00723**2, 115/(365/12*3))
