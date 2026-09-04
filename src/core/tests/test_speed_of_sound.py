import math
import pytest
import mitsuba as mi


def _as_float(x):
    """speed_of_sound()/apply_pure_tone_attenuation() return the variant's
    native Float (needed so gradients survive under *_ad_* variants, see
    acoustic.h) -- a bare Python float for scalar variants, a width-1 drjit
    array otherwise. Extract a plain Python float uniformly for comparison
    against pytest.approx()/==, which can't handle drjit arrays directly."""
    return float(x) if isinstance(x, float) else x[0]


#copied python implementation from pyfar constants:

def _simple(temperature):

    t_ref = 20.0
    t_0 = -273.15
    c_ref = 343.2
    return c_ref*math.sqrt((temperature-t_0)/(t_ref-t_0))


def _ideal_gas(temperature, relative_humidity, atmospheric_pressure, saturation_vapor_pressure=None):
    R = 8.314
    gamma_a = 1.400
    gamma_w = 1.330
    mu_a = 28.97*1e-3
    mu_w = 18.02*1e-3
    R_a = R / mu_a

    temperature_kelvin = temperature + 273.15
    p = atmospheric_pressure

    # matches the Magnus-formula estimate used when saturation_vapor_pressure
    # is not specified
    if saturation_vapor_pressure is None:
        e_s = 6.1094 * math.exp((17.625 * temperature) / (temperature + 243.04))
        e_s = 100 * e_s  # hPa -> Pa
    else:
        e_s = saturation_vapor_pressure

    e = e_s * relative_humidity  # water vapor partial pressure
    alpha = mu_a / mu_w
    delta = (1.0 - (1.0 / gamma_a)) / (1.0 - (1.0 / gamma_w))
    nu = (gamma_a - 1.0) / (gamma_w - 1.0)
    C = (e / p) / (alpha * (1.0 - e / p))

    return math.sqrt(gamma_a * R_a * temperature_kelvin *
                      (1.0 + (alpha * (1.0 + delta - nu) - 1.0) * C))


def _cramer(temperature, relative_humidity, atmospheric_pressure, co2_ppm):
    # mole fraction of CO2 in air
    x_c = co2_ppm * 1e-6
    # pressure(atmospheric pressure) in Pa
    p = atmospheric_pressure
    # relative humidity as a fraction
    h = relative_humidity
    # temperature in Celsius
    t = temperature
    # thermodynamic temperature
    T = t + 273.15
    # enhancement factor f (Equation A2)
    f = 1.00062+3.14e-8 * p + 5.6e-7*temperature**2
    # saturation vapor pressure in air (Equation A3)
    p_sv = math.exp(1.2811805e-5*T**2-1.9509874e-2*T+34.04926034-6.3536311e3/T)
    # water vapor mole fraction (Equation A1)
    x_w = h * f * p_sv / p

    a0 = 331.5024
    a1 = 0.603055
    a2 = -0.000528
    a3 = 51.471935
    a4 = 0.1495874
    a5 = -0.000782
    a6 = -1.82e-7
    a7 = 3.73e-8
    a8 = -2.93e-10
    a9 = -85.20931
    a10 = -0.228525
    a11 = 5.91e-5
    a12 = -2.835149
    a13 = -2.15e-13
    a14 = 29.179762
    a15 = 0.000486

    # approximation for c according to Cramer (Eq. 15)
    c1 = a0+a1*t+a2*t**2
    c2 = (a3+a4*t+a5*t**2)*x_w
    c3 = (a6+a7*t+a8*t**2)*p
    c4 = (a9+a10*t+a11*t**2)*x_c
    c5 = a12*x_w**2 + a13*p**2 + a14*x_c**2 + a15*x_w*p*x_c

    return c1 + c2 + c3 + c4 + c5

def test01_hasattr_acoustic(variants_all_acoustic):
    assert hasattr(mi, "acoustic")


def test02_method_selection(variants_all_acoustic):
    assert _as_float(mi.acoustic.speed_of_sound(temperature=20.0)) == pytest.approx(_simple(20.0), rel=1e-6) # function selects automatically
    assert _as_float(mi.acoustic.speed_of_sound(temperature=20.0, method="auto")) == pytest.approx(_simple(20.0), rel=1e-6)  # function selects automatically
    assert _as_float(mi.acoustic.speed_of_sound(temperature=25.0, method="simple")) == pytest.approx(_simple(25.0), rel=1e-6)  # function should use simple method


def test03_auto_selects_simple_without_humidity(variants_all_acoustic):
    # relative_humidity not provided -> "auto" should fall back to "simple"
    result = _as_float(mi.acoustic.speed_of_sound(temperature=15.0))
    assert result == pytest.approx(_simple(15.0), rel=1e-6)


def test04_auto_selects_ideal_gas_with_humidity_only(variants_all_acoustic):
    # relative_humidity provided, co2_ppm not provided -> "auto" should select "ideal_gas"
    result = _as_float(mi.acoustic.speed_of_sound(
        temperature=20.0,
        relative_humidity=0.5,
        atmospheric_pressure=101325.0,
    ))
    expected = _ideal_gas(20.0, 0.5, 101325.0)
    assert result == pytest.approx(expected, rel=1e-5)


def test05_auto_selects_cramer_with_humidity_and_co2(variants_all_acoustic):
    # relative_humidity and co2_ppm both provided -> "auto" should select "cramer"
    result = _as_float(mi.acoustic.speed_of_sound(
        temperature=20.0,
        relative_humidity=0.5,
        atmospheric_pressure=101325.0,
        co2_ppm=400.0,
    ))
    expected = _cramer(20.0, 0.5, 101325.0, 400.0)
    assert result == pytest.approx(expected, rel=1e-5)


def test06_simple_method_matches_formula(variants_all_acoustic):
    for temperature in (-10.0, 0.0, 25.0, 40.0):
        result = _as_float(mi.acoustic.speed_of_sound(temperature=temperature, method="simple"))
        assert result == pytest.approx(_simple(temperature), rel=1e-6)


# Range checks (this and the other *_raises tests below) are scalar-only by
# design, matching energy_attenuation_coefficient()'s established pattern:
# vectorized/JIT Value skips them and relies on the branch-free formula
# instead (a host `if` can't collapse a per-lane JIT comparison to a single
# bool). variant_scalar_acoustic exercises exactly the code path that does
# validate.
def test07_simple_method_temperature_too_low_raises(variant_scalar_acoustic):
    with pytest.raises(Exception):
        mi.acoustic.speed_of_sound(temperature=-25.0, method="simple")


def test08_simple_method_temperature_too_high_raises(variant_scalar_acoustic):
    with pytest.raises(Exception):
        mi.acoustic.speed_of_sound(temperature=55.0, method="simple")


def test09_ideal_gas_method_matches_formula(variants_all_acoustic):
    temperature, rh, pressure = 22.5, 0.4, 100000.0
    result = _as_float(mi.acoustic.speed_of_sound(
        temperature=temperature,
        relative_humidity=rh,
        atmospheric_pressure=pressure,
        method="ideal_gas",
    ))
    expected = _ideal_gas(temperature, rh, pressure)
    assert result == pytest.approx(expected, rel=1e-5)


def test10_ideal_gas_relative_humidity_out_of_range_raises(variant_scalar_acoustic):
    with pytest.raises(Exception):
        mi.acoustic.speed_of_sound(
            temperature=20.0,
            relative_humidity=1.5,
            atmospheric_pressure=101325.0,
            method="ideal_gas",
        )
    with pytest.raises(Exception):
        mi.acoustic.speed_of_sound(
            temperature=20.0,
            relative_humidity=-0.1,
            atmospheric_pressure=101325.0,
            method="ideal_gas",
        )


def test11_ideal_gas_negative_pressure_raises(variant_scalar_acoustic):
    with pytest.raises(Exception):
        mi.acoustic.speed_of_sound(
            temperature=20.0,
            relative_humidity=0.5,
            atmospheric_pressure=-1.0,
            method="ideal_gas",
        )


def test12_cramer_method_matches_formula(variants_all_acoustic):
    temperature, rh, pressure, co2 = 20.0, 0.5, 101325.0, 400.0
    result = _as_float(mi.acoustic.speed_of_sound(
        temperature=temperature,
        relative_humidity=rh,
        atmospheric_pressure=pressure,
        co2_ppm=co2,
        method="cramer",
    ))
    expected = _cramer(temperature, rh, pressure, co2)
    assert result == pytest.approx(expected, rel=1e-5)


def test13_cramer_temperature_out_of_range_raises(variant_scalar_acoustic):
    with pytest.raises(Exception):
        mi.acoustic.speed_of_sound(
            temperature=-5.0,
            relative_humidity=0.5,
            atmospheric_pressure=101325.0,
            co2_ppm=400.0,
            method="cramer",
        )
    with pytest.raises(Exception):
        mi.acoustic.speed_of_sound(
            temperature=35.0,
            relative_humidity=0.5,
            atmospheric_pressure=101325.0,
            co2_ppm=400.0,
            method="cramer",
        )


def test14_cramer_pressure_out_of_range_raises(variant_scalar_acoustic):
    with pytest.raises(Exception):
        mi.acoustic.speed_of_sound(
            temperature=20.0,
            relative_humidity=0.5,
            atmospheric_pressure=50000.0,
            co2_ppm=400.0,
            method="cramer",
        )
    with pytest.raises(Exception):
        mi.acoustic.speed_of_sound(
            temperature=20.0,
            relative_humidity=0.5,
            atmospheric_pressure=110000.0,
            co2_ppm=400.0,
            method="cramer",
        )


def test15_cramer_co2_out_of_range_raises(variant_scalar_acoustic):
    with pytest.raises(Exception):
        mi.acoustic.speed_of_sound(
            temperature=20.0,
            relative_humidity=0.5,
            atmospheric_pressure=101325.0,
            co2_ppm=-1.0,
            method="cramer",
        )
    with pytest.raises(Exception):
        mi.acoustic.speed_of_sound(
            temperature=20.0,
            relative_humidity=0.5,
            atmospheric_pressure=101325.0,
            co2_ppm=15000.0,
            method="cramer",
        )

# 0.0 > x_w > 0.06 cannot be reached with current input validation
# def test16_cramer_mole_fraction_out_of_range_raises(variants_all_acoustic):
#     # very highest humidity at lowest temperature pushes the computed water vapor
#     # mole fraction above the valid 0-0.06 range
#     with pytest.raises(Exception):
#         mi.acoustic.speed_of_sound(
#             temperature=0.0,
#             relative_humidity=1.0,
#             atmospheric_pressure=75000.0,
#             co2_ppm=400.0,
#             method="cramer",
#         )


def test17_invalid_method_raises(variants_all_acoustic):
    with pytest.raises(Exception):
        mi.acoustic.speed_of_sound(temperature=20.0, method="not_a_method")


def test18_missing_temperature_raises(variants_all_acoustic):
    with pytest.raises(Exception):
        mi.acoustic.speed_of_sound()


def test19_ideal_gas_explicit_saturation_vapor_pressure_matches_formula(variants_all_acoustic):
    # regression test for a bug where the water vapor partial pressure was
    # computed as atmospheric_pressure * relative_humidity instead of
    # saturation_vapor_pressure * relative_humidity, making an explicitly
    # provided saturation_vapor_pressure have no effect on the result
    temperature, rh, pressure, e_s = 20.0, 1.0, 90000.0, 500.0
    result = _as_float(mi.acoustic.speed_of_sound(
        temperature=temperature,
        relative_humidity=rh,
        atmospheric_pressure=pressure,
        saturation_vapor_pressure=e_s,
        method="ideal_gas",
    ))
    expected = _ideal_gas(temperature, rh, pressure, e_s)
    assert result == pytest.approx(expected, rel=1e-5)
    # e_s=500 is far from the ~2340 Pa Magnus-formula default at 20°C, so
    # using it explicitly must give a measurably different result than
    # leaving saturation_vapor_pressure unspecified.
    default_result = _ideal_gas(temperature, rh, pressure)
    assert abs(result - default_result) > 0.5
