import math
import pytest
import mitsuba as mi

r"""
Test Idea:
 - the most potencial for errors is in the energy_attenuation_coefficient as it s more compoley than (just) applying
 - energy_attenuation_coefficient is not directly exposed to python
 - however m can be calculated from apply_pure_tone_attenuation with

.. math:: m = -\frac{\ln{ETC_out / ETC_in}}{d}

 - this way the coefficent can be tested against the example ISO values
"""

# data from the ISO 9613-1:1993 (transcripted from image of table by AI; no guarantee of correctness)

# air_temperature_category -> frequency_hz -> humidity_percent -> value
ABSORPTION_DATA_MINUS_20C = {
    50:    {10: 0.589, 15: 0.509, 20: 0.418, 30: 0.285, 40: 0.211, 50: 0.168, 60: 0.142, 70: 0.125, 80: 0.114, 90: 0.105, 100: 0.0992},
    63:    {10: 0.756, 15: 0.704, 20: 0.602, 30: 0.421, 40: 0.308, 50: 0.241, 60: 0.2,   70: 0.173, 80: 0.155, 90: 0.142, 100: 0.133},
    80:    {10: 0.924, 15: 0.935, 20: 0.846, 30: 0.619, 40: 0.455, 50: 0.352, 60: 0.286, 70: 0.243, 80: 0.214, 90: 0.194, 100: 0.179},
    100:   {10: 1.08,  15: 1.18,  20: 1.15,  30: 0.902, 40: 0.675, 50: 0.521, 60: 0.419, 70: 0.35,  80: 0.303, 90: 0.269, 100: 0.245},
    125:   {10: 1.2,   15: 1.43,  20: 1.49,  30: 1.28,  40: 0.998, 50: 0.776, 60: 0.622, 70: 0.514, 80: 0.439, 90: 0.384, 100: 0.344},
    160:   {10: 1.3,   15: 1.64,  20: 1.83,  30: 1.77,  40: 1.45,  50: 1.16,  60: 0.93,  70: 0.766, 80: 0.648, 90: 0.561, 100: 0.496},
    200:   {10: 1.37,  15: 1.82,  20: 2.15,  30: 2.33,  40: 2.06,  50: 1.7,   60: 1.39,  70: 1.15,  80: 0.97,  90: 0.834, 100: 0.731},
    250:   {10: 1.43,  15: 1.95,  20: 2.42,  30: 2.93,  40: 2.83,  50: 2.46,  60: 2.06,  70: 1.73,  80: 1.46,  90: 1.26,  100: 1.09},
    315:   {10: 1.46,  15: 2.05,  20: 2.63,  30: 3.49,  40: 3.7,   50: 3.43,  60: 3.0,   70: 2.57,  80: 2.2,   90: 1.9,   100: 1.65},
    400:   {10: 1.49,  15: 2.12,  20: 2.79,  30: 3.99,  40: 4.6,   50: 4.59,  60: 4.23,  70: 3.74,  80: 3.27,  90: 2.85,  100: 2.5},
    500:   {10: 1.52,  15: 2.17,  20: 2.91,  30: 4.38,  40: 5.45,  50: 5.86,  60: 5.72,  70: 5.29,  80: 4.76,  90: 4.23,  100: 3.76},
    630:   {10: 1.55,  15: 2.22,  20: 3.0,   30: 4.68,  40: 6.17,  50: 7.1,   60: 7.39,  70: 7.19,  80: 6.71,  90: 6.13,  100: 5.55},
    800:   {10: 1.59,  15: 2.27,  20: 3.08,  30: 4.92,  40: 6.75,  50: 8.22,  60: 9.07,  70: 9.31,  80: 9.09,  90: 8.6,   100: 7.98},
    1000:  {10: 1.65,  15: 2.34,  20: 3.16,  30: 5.11,  40: 7.21,  50: 9.14,  60: 10.6,  70: 11.5,  80: 11.7,  90: 11.6,  100: 11.1},
    1250:  {10: 1.74,  15: 2.43,  20: 3.27,  30: 5.28,  40: 7.57,  50: 9.88,  60: 11.9,  70: 13.5,  80: 14.4,  90: 14.8,  100: 14.7},
    1600:  {10: 1.88,  15: 2.58,  20: 3.42,  30: 5.48,  40: 7.9,   50: 10.5,  60: 13.0,  70: 15.2,  80: 16.9,  90: 18.0,  100: 18.6},
    2000:  {10: 2.1,   15: 2.8,   20: 3.65,  30: 5.73,  40: 8.24,  50: 11.0,  60: 13.9,  70: 16.6,  80: 19.0,  90: 21.0,  100: 22.4},
    2500:  {10: 2.44,  15: 3.15,  20: 4.0,   30: 6.1,   40: 8.66,  50: 11.6,  60: 14.7,  70: 17.8,  80: 20.8,  90: 23.5,  100: 25.8},
    3150:  {10: 2.99,  15: 3.69,  20: 4.55,  30: 6.66,  40: 9.26,  50: 12.3,  60: 15.5,  70: 19.0,  80: 22.4,  90: 25.7,  100: 28.8},
    4000:  {10: 3.86,  15: 4.56,  20: 5.42,  30: 7.54,  40: 10.2,  50: 13.2,  60: 16.6,  70: 20.2,  80: 24.0,  90: 27.8,  100: 31.4},
    5000:  {10: 5.24,  15: 5.94,  20: 6.8,   30: 8.92,  40: 11.6,  50: 14.6,  60: 18.1,  70: 21.9,  80: 25.9,  90: 30.0,  100: 34.1},
    6300:  {10: 7.42,  15: 8.12,  20: 8.98,  30: 11.1,  40: 13.8,  50: 16.9,  60: 20.4,  70: 24.2,  80: 28.3,  90: 32.7,  100: 37.1},
    8000:  {10: 10.9,  15: 11.6,  20: 12.4,  30: 14.6,  40: 17.2,  50: 20.3,  60: 23.9,  70: 27.8,  80: 32.0,  90: 36.5,  100: 41.1},
    10000: {10: 16.4,  15: 17.1,  20: 17.9,  30: 20.1,  40: 22.7,  50: 25.8,  60: 29.4,  70: 33.3,  80: 37.6,  90: 42.2,  100: 47.0},
}

ABSORPTION_DATA_25C = {
    50:    {10: 0.262, 15: 0.197, 20: 0.156, 30: 0.109, 40: 0.083, 50: 0.0671, 60: 0.0563, 70: 0.0485, 80: 0.0426, 90: 0.0379, 100: 0.0342},
    63:    {10: 0.374, 15: 0.295, 20: 0.238, 30: 0.169, 40: 0.13, 50: 0.106, 60: 0.0888, 70: 0.0765, 80: 0.0673, 90: 0.06, 100: 0.0541},
    80:    {10: 0.515, 15: 0.429, 20: 0.357, 30: 0.261, 40: 0.203, 50: 0.166, 60: 0.14, 70: 0.121, 80: 0.106, 90: 0.0948, 100: 0.0856},
    100:   {10: 0.681, 15: 0.604, 20: 0.523, 30: 0.397, 40: 0.314, 50: 0.258, 60: 0.219, 70: 0.19, 80: 0.167, 90: 0.149, 100: 0.135},
    125:   {10: 0.867, 15: 0.816, 20: 0.74, 30: 0.591, 40: 0.479, 50: 0.399, 60: 0.34, 70: 0.296, 80: 0.262, 90: 0.235, 100: 0.213},
    160:   {10: 1.07, 15: 1.06, 20: 1.01, 30: 0.856, 40: 0.717, 50: 0.608, 60: 0.525, 70: 0.46, 80: 0.409, 90: 0.367, 100: 0.333},
    200:   {10: 1.31, 15: 1.32, 20: 1.31, 30: 1.2, 40: 1.05, 50: 0.909, 60: 0.797, 70: 0.706, 80: 0.631, 90: 0.57, 100: 0.52},
    250:   {10: 1.61, 15: 1.6, 20: 1.64, 30: 1.6, 40: 1.47, 50: 1.32, 60: 1.18, 70: 1.06, 80: 0.963, 90: 0.876, 100: 0.803},
    315:   {10: 2.02, 15: 1.93, 20: 1.99, 30: 2.05, 40: 1.99, 50: 1.86, 60: 1.71, 70: 1.57, 80: 1.44, 90: 1.32, 100: 1.22},
    400:   {10: 2.63, 15: 2.35, 20: 2.38, 30: 2.53, 40: 2.57, 50: 2.51, 60: 2.38, 70: 2.24, 80: 2.1, 90: 1.96, 100: 1.83},
    500:   {10: 3.56, 15: 2.92, 20: 2.86, 30: 3.04, 40: 3.19, 50: 3.23, 60: 3.18, 70: 3.08, 80: 2.95, 90: 2.8, 100: 2.66},
    630:   {10: 5.0, 15: 3.78, 20: 3.5, 30: 3.61, 40: 3.84, 50: 4.0, 60: 4.06, 70: 4.05, 80: 3.97, 90: 3.86, 100: 3.73},
    800:   {10: 7.24, 15: 5.09, 20: 4.44, 30: 4.31, 40: 4.55, 50: 4.8, 60: 4.99, 70: 5.09, 80: 5.12, 90: 5.09, 100: 5.02},
    1000:  {10: 10.7, 15: 7.13, 20: 5.87, 30: 5.27, 40: 5.39, 50: 5.68, 60: 5.96, 70: 6.19, 80: 6.35, 90: 6.44, 100: 6.47},
    1250:  {10: 16.1, 15: 10.3, 20: 8.09, 30: 6.68, 40: 6.52, 50: 6.73, 60: 7.04, 70: 7.35, 80: 7.62, 90: 7.84, 100: 8.0},
    1600:  {10: 24.3, 15: 15.3, 20: 11.6, 30: 8.85, 40: 8.16, 50: 8.14, 60: 8.36, 70: 8.68, 80: 9.01, 90: 9.33, 100: 9.61},
    2000:  {10: 36.6, 15: 23.0, 20: 17.0, 30: 12.2, 40: 10.7, 50: 10.2, 60: 10.2, 70: 10.4, 80: 10.7, 90: 11.0, 100: 11.4},
    2500:  {10: 54.2, 15: 34.9, 20: 25.5, 30: 17.5, 40: 14.5, 50: 13.3, 60: 12.8, 70: 12.8, 80: 12.9, 90: 13.2, 100: 13.5},
    3150:  {10: 78.6, 15: 52.9, 20: 38.6, 30: 25.8, 40: 20.6, 50: 18.1, 60: 16.9, 70: 16.3, 80: 16.2, 90: 16.2, 100: 16.4},
    4000:  {10: 110.0, 15: 79.4, 20: 58.8, 30: 38.8, 40: 30.1, 50: 25.7, 60: 23.2, 70: 21.9, 80: 21.1, 90: 20.8, 100: 20.6},
    5000:  {10: 149.0, 15: 117.0, 20: 89.1, 30: 59.0, 40: 45.0, 50: 37.6, 60: 33.2, 70: 30.5, 80: 28.8, 90: 27.7, 100: 27.1},
    6300:  {10: 191.0, 15: 168.0, 20: 133.0, 30: 90.0, 40: 68.3, 50: 56.2, 60: 48.8, 70: 44.1, 80: 40.8, 90: 38.6, 100: 37.1},
    8000:  {10: 233.0, 15: 232.0, 20: 196.0, 30: 137.0, 40: 104.0, 50: 85.4, 60: 73.4, 70: 65.4, 80: 59.8, 90: 55.8, 100: 52.8},
    10000: {10: 274.0, 15: 308.0, 20: 279.0, 30: 207.0, 40: 160.0, 50: 131.0, 60: 112.0, 70: 98.9, 80: 89.6, 90: 82.8, 100: 77.6},
}

ABSORPTION_DATA_5C = {
    50:    {10: 0.268, 15: 0.22, 20: 0.197, 30: 0.164, 40: 0.138, 50: 0.118, 60: 0.103, 70: 0.0909, 80: 0.0812, 90: 0.0733, 100: 0.0667},
    63:    {10: 0.359, 15: 0.288, 20: 0.261, 30: 0.227, 40: 0.199, 50: 0.175, 60: 0.155, 70: 0.138, 80: 0.124, 90: 0.113, 100: 0.103},
    80:    {10: 0.488, 15: 0.375, 20: 0.337, 30: 0.303, 40: 0.276, 50: 0.25, 60: 0.227, 70: 0.206, 80: 0.188, 90: 0.172, 100: 0.158},
    100:   {10: 0.68, 15: 0.492, 20: 0.431, 30: 0.391, 40: 0.369, 50: 0.345, 60: 0.321, 70: 0.298, 80: 0.276, 90: 0.256, 100: 0.238},
    125:   {10: 0.971, 15: 0.661, 20: 0.554, 30: 0.493, 40: 0.474, 50: 0.458, 60: 0.438, 70: 0.416, 80: 0.393, 90: 0.371, 100: 0.349},
    160:   {10: 1.42, 15: 0.914, 20: 0.729, 30: 0.617, 40: 0.594, 50: 0.585, 60: 0.574, 70: 0.558, 80: 0.539, 90: 0.518, 100: 0.496},
    200:   {10: 2.09, 15: 1.3, 20: 0.988, 30: 0.781, 40: 0.735, 50: 0.727, 60: 0.725, 70: 0.72, 80: 0.71, 90: 0.696, 100: 0.678},
    250:   {10: 3.11, 15: 1.9, 20: 1.38, 30: 1.01, 40: 0.915, 50: 0.892, 60: 0.892, 70: 0.897, 80: 0.898, 90: 0.896, 100: 0.888},
    315:   {10: 4.58, 15: 2.82, 20: 2.0, 30: 1.36, 40: 1.16, 50: 1.1, 60: 1.09, 70: 1.09, 80: 1.1, 90: 1.11, 100: 1.12},
    400:   {10: 6.64, 15: 4.23, 20: 2.95, 30: 1.9, 40: 1.53, 50: 1.39, 60: 1.34, 70: 1.33, 80: 1.34, 90: 1.35, 100: 1.37},
    500:   {10: 9.34, 15: 6.32, 20: 4.42, 30: 2.74, 40: 2.1, 50: 1.82, 60: 1.69, 70: 1.64, 80: 1.63, 90: 1.64, 100: 1.66},
    630:   {10: 12.6, 15: 9.34, 20: 6.66, 30: 4.04, 40: 2.97, 50: 2.47, 60: 2.22, 70: 2.09, 80: 2.03, 90: 2.01, 100: 2.01},
    800:   {10: 16.3, 15: 13.5, 20: 9.99, 30: 6.06, 40: 4.34, 50: 3.49, 60: 3.03, 70: 2.76, 80: 2.61, 90: 2.53, 100: 2.49},
    1000:  {10: 20.0, 15: 18.9, 20: 14.8, 30: 9.18, 40: 6.48, 50: 5.08, 60: 4.29, 70: 3.8, 80: 3.5, 90: 3.31, 100: 3.2},
    1250:  {10: 23.4, 15: 25.4, 20: 21.5, 30: 13.9, 40: 9.81, 50: 7.58, 60: 6.26, 70: 5.43, 80: 4.89, 90: 4.52, 100: 4.27},
    1600:  {10: 26.2, 15: 32.6, 20: 30.1, 30: 20.9, 40: 14.9, 50: 11.5, 60: 9.35, 70: 7.99, 80: 7.06, 90: 6.42, 100: 5.95},
    2000:  {10: 28.5, 15: 39.6, 20: 40.5, 30: 30.9, 40: 22.7, 50: 17.5, 60: 14.2, 70: 12.0, 80: 10.5, 90: 9.39, 100: 8.58},
    2500:  {10: 30.4, 15: 46.1, 20: 51.9, 30: 44.6, 40: 34.1, 50: 26.6, 60: 21.7, 70: 18.2, 80: 15.8, 90: 14.1, 100: 12.7},
    3150:  {10: 31.9, 15: 51.6, 20: 63.2, 30: 62.0, 40: 50.4, 50: 40.3, 60: 33.1, 70: 27.9, 80: 24.2, 90: 21.4, 100: 19.2},
    4000:  {10: 33.5, 15: 56.2, 20: 73.7, 30: 82.6, 40: 72.5, 50: 60.2, 60: 50.2, 70: 42.7, 80: 37.0, 90: 32.7, 100: 29.4},
    5000:  {10: 35.4, 15: 60.2, 20: 82.8, 30: 105.0, 40: 101.0, 50: 87.8, 60: 75.2, 70: 64.8, 80: 56.6, 90: 50.2, 100: 45.1},
    6300:  {10: 38.0, 15: 64.3, 20: 90.9, 30: 127.0, 40: 133.0, 50: 124.0, 60: 110.0, 70: 97.0, 80: 85.8, 90: 76.6, 100: 69.1},
    8000:  {10: 41.8, 15: 69.1, 20: 98.7, 30: 147.0, 40: 169.0, 50: 168.0, 60: 156.0, 70: 142.0, 80: 128.0, 90: 116.0, 100: 105.0},
    10000: {10: 47.7, 15: 75.7, 20: 107.0, 30: 167.0, 40: 205.0, 50: 218.0, 60: 214.0, 70: 201.0, 80: 186.0, 90: 172.0, 100: 158.0},
}






def _get_m(etc_in, etc_out, distance):
    return -math.log(etc_out / etc_in) / distance


def _m_per_m_from_apply(temperature, frequency, relative_humidity_percent,
                         atmospheric_pressure=101325.0, distance=1.0):
    # apply_pure_tone_attenuation with a single time bin at distance=0 (etc_in)
    # and one at `distance` (etc_out), sampling_rate and speed chosen so that
    # time * speed_of_sound_ms == distance.
    etc_in = 1.0
    etc = [etc_in, etc_in]
    out = mi.acoustic.apply_pure_tone_attenuation(
        etc, 1.0, distance, temperature, [frequency],
        relative_humidity_percent / 100.0, atmospheric_pressure)
    etc_out = out[1]
    return _get_m(etc_in, etc_out, distance)


# ISO 9613-1 tables give the attenuation coefficient in dB/km
def _db_per_km_to_m_per_m(db_per_km):
    return db_per_km / 1000.0 / (10.0 * math.log10(math.e))


ISO_CASES = [
    (-20.0, ABSORPTION_DATA_MINUS_20C),
    (5.0, ABSORPTION_DATA_5C),
    (25.0, ABSORPTION_DATA_25C),
]


def test01_hasattr_acoustic(variants_all_acoustic):
    assert hasattr(mi, "acoustic")


@pytest.mark.parametrize("temperature,table", ISO_CASES)
def test02_energy_attenuation_coefficient_matches_ISO(variants_all_acoustic, temperature, table):
    # spot-check a handful of frequency/humidity combinations per temperature
    # category against the ISO 9613-1 reference table (rel. tolerance accounts
    # for the table's own rounding to 2-3 significant digits)
    frequencies = (50, 63, 250, 1000, 4000, 6300, 10000)
    humidities = (10, 20, 30, 50, 70, 100)

    for frequency in frequencies:
        for humidity in humidities:
            expected_db_per_km = table[frequency][humidity]
            expected_m = _db_per_km_to_m_per_m(expected_db_per_km)
            got_m = _m_per_m_from_apply(temperature, frequency, humidity)
            assert got_m == pytest.approx(expected_m, rel=0.015) #1,5% relative tolerance


def test03_apply_pure_tone_attenuation_direct_call(variants_all_acoustic):
    # simple end-to-end sanity check: energy must decay with distance and
    # higher frequencies must attenuate more than lower ones
    n_time_bins = 4
    frequencies = [500.0, 4000.0]
    etc = [1.0] * (n_time_bins * len(frequencies))

    out = mi.acoustic.apply_pure_tone_attenuation(
        etc, sampling_rate=10.0, speed_of_sound_ms=343.0,
        temperature=20.0, frequencies=frequencies,
        relative_humidity=0.5, atmospheric_pressure=101325.0)

    out_low = out[0::2]
    out_high = out[1::2]

    assert out_low[0] == pytest.approx(1.0)
    assert out_high[0] == pytest.approx(1.0)
    for t in range(1, n_time_bins):
        assert out_low[t] < out_low[t - 1]
        assert out_high[t] < out_high[t - 1]
        assert out_high[t] < out_low[t]


def test04_etc_size_not_multiple_of_frequencies_raises(variants_all_acoustic):
    with pytest.raises(Exception):
        mi.acoustic.apply_pure_tone_attenuation(
            etc=[1.0, 1.0, 1.0], sampling_rate=10.0, speed_of_sound_ms=343.0,
            temperature=20.0, frequencies=[1000.0, 2000.0],
            relative_humidity=0.5, atmospheric_pressure=101325.0)


def test05_empty_frequencies_raises(variants_all_acoustic):
    with pytest.raises(Exception):
        mi.acoustic.apply_pure_tone_attenuation(
            etc=[], sampling_rate=10.0, speed_of_sound_ms=343.0,
            temperature=20.0, frequencies=[],
            relative_humidity=0.5, atmospheric_pressure=101325.0)


def test06_temperature_too_low_raises(variants_all_acoustic):
    with pytest.raises(Exception):
        mi.acoustic.apply_pure_tone_attenuation(
            etc=[1.0], sampling_rate=10.0, speed_of_sound_ms=343.0,
            temperature=-100.0, frequencies=[1000.0],
            relative_humidity=0.5, atmospheric_pressure=101325.0)


def test07_frequency_too_low_raises(variants_all_acoustic):
    with pytest.raises(Exception):
        mi.acoustic.apply_pure_tone_attenuation(
            etc=[1.0], sampling_rate=10.0, speed_of_sound_ms=343.0,
            temperature=20.0, frequencies=[10.0],
            relative_humidity=0.5, atmospheric_pressure=101325.0)


def test08_atmospheric_pressure_too_high_raises(variants_all_acoustic):
    with pytest.raises(Exception):
        mi.acoustic.apply_pure_tone_attenuation(
            etc=[1.0], sampling_rate=10.0, speed_of_sound_ms=343.0,
            temperature=20.0, frequencies=[1000.0],
            relative_humidity=0.5, atmospheric_pressure=250000.0)


def test09_frequency_to_pressure_ratio_out_of_range_raises(variants_all_acoustic):
    # ratio below 4e-4 Hz/Pa
    with pytest.raises(Exception):
        mi.acoustic.apply_pure_tone_attenuation(
            etc=[1.0], sampling_rate=10.0, speed_of_sound_ms=343.0,
            temperature=20.0, frequencies=[50.0],
            relative_humidity=0.5, atmospheric_pressure=150000.0)
    # ratio above 10 Hz/Pa
    with pytest.raises(Exception):
        mi.acoustic.apply_pure_tone_attenuation(
            etc=[1.0], sampling_rate=10.0, speed_of_sound_ms=343.0,
            temperature=20.0, frequencies=[10000.0],
            relative_humidity=0.5, atmospheric_pressure=500.0)


def test10_distance_too_large_raises(variants_all_acoustic):
    # time bin 1 is at 1s * 20000 m/s = 20 km > the 10 km limit
    with pytest.raises(Exception):
        mi.acoustic.apply_pure_tone_attenuation(
            etc=[1.0, 1.0], sampling_rate=1.0, speed_of_sound_ms=20000.0,
            temperature=20.0, frequencies=[1000.0],
            relative_humidity=0.5, atmospheric_pressure=101325.0,
            n_time_bins=2, n_frequencies=1)

