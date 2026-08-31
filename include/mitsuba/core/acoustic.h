#pragma once

#include <mitsuba/mitsuba.h>
#include <mitsuba/core/logger.h>
#include <drjit/math.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

NAMESPACE_BEGIN(mitsuba)
NAMESPACE_BEGIN(acoustic)

template <typename Value>
inline bool is_missing_value(const Value &value) {
    if constexpr (std::is_floating_point_v<Value>) {
        return std::isnan(static_cast<double>(value));
    } else {
        return false;
    }
}

// -----------------------------------------------------------------------
//! @{ \name Speed of sound calculation (adapts to the type of input parameters)
// -----------------------------------------------------------------------

/**
 * \brief Speed of sound in air following ISO 9613-1 (Formula A.5).
 *
 * \param temperature
 *      The temperature in degree Celsius. Must be in the range of -20°C to
 *      50°C.
 *
 * \return
 *      The speed of sound in meters per second
 */
template <typename Value>
float speed_of_sound_simple(const Value temperature) {
    if (temperature < -20.0f || temperature > 50.0f) {
        throw std::invalid_argument("Temperature out of range for simple method (-20°C to 50°C).");
    }
    return 343.2f * dr::sqrt((temperature + 273.15f) / 293.15f);
}

/**
 * \brief Speed of sound in air based on chapter 6.3 in V. E. Ostashev and
 * D. K. Wilson, Acoustics in Moving Inhomogeneous Media, 2nd ed. London:
 * CRC Press, 2015. doi: 10.1201/b18922.
 *
 * \param temperature
 *      The temperature in degree Celsius.
 * \param relative_humidity
 *     Relative humidity in the range of 0 to 1.
 * \param atmospheric_pressure
 *    Atmospheric pressure in Pascal, must be non-negative.
 * \param saturation_vapor_pressure
 *    Saturation vapor pressure in Pascal. A negative value (see default)
 *    means "not specified": it is then estimated from \c temperature via
 *    the Magnus formula.
 *
 * \return
 *      The speed of sound in meters per second
 */
template <typename Value>
float speed_of_sound_ideal_gas(const Value temperature,
                               const Value relative_humidity,
                               const Value atmospheric_pressure,
                               const Value saturation_vapor_pressure = Value(-1)) {
    if (relative_humidity < 0.0f || relative_humidity > 1.0f) {
        throw std::invalid_argument("Relative humidity must be in the range of 0 to 1.");
    }
    if (atmospheric_pressure < 0.0f) {
        throw std::invalid_argument("Atmospheric pressure must be non-negative.");
    }

    float temperature_kelvin = static_cast<float>(temperature) + 273.15f;

    // Ideal gas calculation based on Ostashev and Wilson
    float R = 8.314f; // J/(mol*K)
    float gamma_a = 1.400f;
    float gamma_w = 1.330f;
    float mu_a = 28.97f * 1e-3f; //kg/mol
    float mu_w = 18.02f * 1e-3f; //kg/mol
    float R_a = R / mu_a;

    // saturation_vapor_pressure < 0 means "not specified" (matches the
    // Python binding's default of -1). Estimate it from temperature using
    // the Magnus formula (see e.g. O. A. Alduchov and R. E. Eskridge,
    // "Improved Magnus Form Approximation of Saturation Vapor Pressure,"
    // J. Appl. Meteor., 1996).
    Value e_s;
    if (saturation_vapor_pressure < Value(0)) {
        e_s = Value(6.1094f) * dr::exp((Value(17.625f) * temperature) /
                                       (temperature + Value(243.04f)));
        e_s = Value(100) * e_s; // hPa → Pa
    } else {
        e_s = saturation_vapor_pressure;
    }

    float e = static_cast<float>(e_s) * static_cast<float>(relative_humidity); // water vapor partial pressure
    float alpha = mu_a / mu_w;
    float delta = (1.0f - (1.0f / gamma_a)) / (1.0f - (1.0f / gamma_w));
    float nu = (gamma_a - 1.0f) / (gamma_w - 1.0f);
    float C = (e / static_cast<float>(atmospheric_pressure)) /
              (alpha * (1.0f - e / static_cast<float>(atmospheric_pressure)));

    return dr::sqrt(gamma_a * R_a * temperature_kelvin *
                    (1.0f + (alpha * (1.0f + delta - nu) - 1.0f) * C));
}

/**
 * \brief Speed of sound in air using Cramer's method described in
 * O. Cramer, "The variation of the specific heat ratio and the speed of
 * sound in air with temperature, pressure, humidity, and CO2
 * concentration," The Journal of the Acoustical Society of America,
 * vol. 93, no. 5, pp. 2510-2516, May 1993, doi: 10.1121/1.405827.
 *
 * \param temperature
 *      The temperature in degree Celsius. Must be in the range of 0°C to
 *      30°C.
 * \param relative_humidity
 *     Relative humidity in the range of 0 to 1.
 * \param atmospheric_pressure
 *    Atmospheric pressure in Pascal, must be non-negative and in the range
 *    of 75,000 Pa to 102,000 Pa. Missing values (see is_missing_value())
 *    default to 101,325 Pa (standard atmosphere).
 * \param co2_ppm
 *   CO2 concentration in parts per million (ppm), must be in the range of
 *   0 ppm to 10,000 ppm. Missing values (see is_missing_value()) default to
 *   428.73 ppm, the global monthly mean for 2026-07 reported by NOAA GML
 *   (https://doi.org/10.15138/9N0H-ZH07, retrieved 2026-08-28).
 *
 * \return
 *      The speed of sound in meters per second
 */
template <typename Value>
float speed_of_sound_cramer(const Value temperature,
                            const Value relative_humidity,
                            Value atmospheric_pressure,
                            Value co2_ppm = Value(std::numeric_limits<float>::quiet_NaN())) {
    if (relative_humidity < 0.0f || relative_humidity > 1.0f) {
        throw std::invalid_argument("Relative humidity must be in the range of 0 to 1.");
    }
    if (atmospheric_pressure < 0.0f) {
        throw std::invalid_argument("Atmospheric pressure must be non-negative.");
    }

    // Resolve missing values to their defaults before validating ranges, so
    // that the value actually used in the calculation is the one checked.
    if (is_missing_value(atmospheric_pressure)) {
        atmospheric_pressure = Value(101325.0f); // standard atmosphere
    }
    if (is_missing_value(co2_ppm)) {
        co2_ppm = Value(428.73f); // NOAA GML global mean, doi.org/10.15138/9N0H-ZH07
    }

    // Cramer's specific bounds for temperature and pressure
    if (temperature < 0.0f || temperature > 30.0f) {
        throw std::invalid_argument("Temperature out of range for Cramer's method (0°C to 30°C).");
    }
    else if (atmospheric_pressure < 75000.0f || atmospheric_pressure > 102000.0f) {
        throw std::invalid_argument("Atmospheric pressure out of range for Cramer's method (75,000 Pa to 102,000 Pa).");
    }
    else if (co2_ppm < 0.0f || co2_ppm > 10000.0f) {
        throw std::invalid_argument("CO2 concentration out of range for Cramer's method (0 ppm to 10,000 ppm).");
    }

    float x_c = co2_ppm * 1e-6f; // Convert ppm to mole fraction
    float T = temperature + 273.15f; // Convert to Kelvin
    float p = atmospheric_pressure; // in Pa

    float p_sv = dr::exp(1.2811805e-5f * T * T - 1.9509874e-2f * T +
                         34.04926034f - 6.3536311e3f / T);

    float x_w = relative_humidity * p_sv / p; // Mole fraction of water vapor

    // cannot happen with inut parameter limitation
    // if (x_w < 0.0f || x_w > 0.06f) {
    //     throw std::invalid_argument("Calculated mole fraction of water vapor is out of range (0 to 0.06). Check input parameters. This input combination of values is not allowed for cramer.");
    // }

    float a0 = 331.5024f;
    float a1 = 0.603055f;
    float a2 = -0.000528f;
    float a3 = 51.471935f;
    float a4 = 0.1495874f;
    float a5 = -0.000782f;
    float a6 = -1.82e-7f;
    float a7 = 3.73e-8f;
    float a8 = -2.93e-10f;
    float a9 = -85.20931f;
    float a10 = -0.228525f;
    float a11 = 5.91e-5f;
    float a12 = -2.835149f;
    float a13 = -2.15e-13f;
    float a14 = 29.179762f;
    float a15 = 0.000486f;

    return  a0 + a1*temperature + a2*temperature*temperature
            + (a3 + a4*temperature + a5*temperature*temperature) * x_w
            + (a6 + a7*temperature + a8*temperature*temperature) * p
            + (a9 + a10*temperature + a11*temperature*temperature) * x_c
            + (a12*x_w*x_w + a13*p*p + a14*x_c*x_c + a15*x_c*p*x_w);
}

/**
 * \brief Calculation methods and automatic method selector for the speed of sound
 *
 * This function calculates the speed of sound in air, using one of the
 * following methods:
 *
 * "simple": following ISO 9613-1 (Formula A.5),
 * ``c = 343.2 * sqrt((T + 273.15) / 293.15)``. Only uses \c
 * temperature (``T``), which must be in the range of -20°C to 50°C.
 *
 * "ideal_gas": speed of sound of a humid-air mixture treated as an
 * ideal gas, based on chapter 6.3 in V. E. Ostashev and D. K. Wilson,
 * Acoustics in Moving Inhomogeneous Media, 2nd ed. London: CRC Press,
 * 2015. doi: 10.1201/b18922,
 * ``c = sqrt(gamma_a * R_a * T_K * (1 + (alpha * (1 + delta - nu) - 1)
 * * C))``, where ``T_K`` is \c temperature in Kelvin, ``R_a`` the
 * specific gas constant of dry air, ``gamma_a``/``gamma_w`` the heat
 * capacity ratios of dry air and water vapor, ``alpha`` the ratio of
 * their molar masses, and ``C`` the water vapor mole fraction term
 * derived from \c relative_humidity, \c atmospheric_pressure and \c
 * saturation_vapor_pressure; see speed_of_sound_ideal_gas() for the
 * exact constants.
 *
 * "cramer": O. Cramer, "The variation of the specific heat ratio and
 * the speed of sound in air with temperature, pressure, humidity, and
 * CO2 concentration," The Journal of the Acoustical Society of
 * America, vol. 93, no. 5, pp. 2510-2516, May 1993,
 * doi: 10.1121/1.405827, an empirical quadratic fit
 * ``c = a0 + a1*T + a2*T^2 + (a3 + a4*T + a5*T^2)*x_w + (a6 + a7*T +
 * a8*T^2)*p + (a9 + a10*T + a11*T^2)*x_c + a12*x_w^2 + a13*p^2 +
 * a14*x_c^2 + a15*x_c*p*x_w``, where ``x_w`` is the water vapor mole
 * fraction (derived from \c relative_humidity and ``p``), ``p`` is \c
 * atmospheric_pressure and ``x_c`` is the CO2 mole fraction (derived
 * from \c co2_ppm); the 16 empirical coefficients ``a0`` ... ``a15``
 * are given in speed_of_sound_cramer(). Requires \c temperature in
 * the range of 0°C to 30°C and \c atmospheric_pressure in the range
 * of 75,000 Pa to 102,000 Pa.
 *
 * \param temperature
 *      The temperature in degree Celsius.
 * \param relative_humidity
 *      Relative humidity in the range of 0 to 1.
 * \param atmospheric_pressure
 *      Atmospheric pressure in Pascal, must be non-negative. For "cramer",
 *      a missing value (see is_missing_value()) defaults to 101,325 Pa
 *      (standard atmosphere).
 * \param saturation_vapor_pressure
 *      Saturation vapor pressure in Pascal. Only used by the "ideal_gas"
 *      method. A negative value (see default) means "not specified": it is
 *      then estimated from \c temperature via the Magnus formula (see e.g.
 *      O. A. Alduchov and R. E. Eskridge, "Improved Magnus Form
 *      Approximation of Saturation Vapor Pressure," J. Appl. Meteor.,
 *      1996).
 * \param co2_ppm
 *      CO2 concentration in parts per million (ppm). Only used by the
 *      "cramer" method (and to auto-select it, see below), must be in the
 *      range of 0 ppm to 10,000 ppm. A missing value (see
 *      is_missing_value()) defaults to 428.73 ppm, the global monthly mean
 *      for 2026-07 reported by NOAA GML
 *      (https://doi.org/10.15138/9N0H-ZH07, retrieved 2026-08-28).
 * \param method
 *      The method to use for the calculation: "simple", "ideal_gas",
 *      "cramer", or "auto" (default), which automatically selects one of
 *      the other three based on which of the parameters above were
 *      provided (see the warning logged at runtime for which one was
 *      picked).
 *
 * \return
 *      The speed of sound in meters per second
 */
template <typename Value>
float speed_of_sound(const Value temperature,
                     const Value relative_humidity,
                     Value atmospheric_pressure,
                     const Value saturation_vapor_pressure,
                     const Value co2_ppm,
                     const std::string& method = "auto") {

    // input validation - at least temperature must be provided
    if (is_missing_value(temperature)) {
        throw std::invalid_argument("Temperature must be provided.");
    }

    std::string selected_method = method;

    // selection logic
    if (selected_method == "auto") {
        if (is_missing_value(relative_humidity)) {
            selected_method = "simple";
        } else if (!is_missing_value(co2_ppm)) {
            selected_method = "cramer";
        } else {
            selected_method = "ideal_gas";
        }
        // No method was explicitly requested: let the user know which one
        // was picked, since it depends on which parameters were provided
        // and silently changes if that set of parameters changes later.
        // Logged at Warn (not Info) since mitsuba's default log level is
        // Warn; an Info-level message here would be silently suppressed
        // unless the user explicitly lowers the log level.
        // Note: the extra parentheses around the function name prevent this
        // call from being expanded by the member-function-only `Log(...)`
        // macro defined in logger.h (this is a free function, no m_class).
        (mitsuba::detail::Log)(Warn, nullptr, __FILE__, __LINE__,
            "speed_of_sound(): no method specified, automatically selected "
            "\"%s\" based on the provided parameters.", selected_method);
    }

    if (selected_method == "simple") {
        return speed_of_sound_simple<Value>(temperature);
    } else if (selected_method == "ideal_gas") {
        return speed_of_sound_ideal_gas<Value>(temperature, relative_humidity,
                                               atmospheric_pressure, saturation_vapor_pressure);
    } else if (selected_method == "cramer") {
        return speed_of_sound_cramer<Value>(temperature, relative_humidity,
                                            atmospheric_pressure, co2_ppm);
    } else {
        throw std::invalid_argument("Invalid method specified for speed of sound calculation. "
                                    "Valid options are 'auto', 'simple', 'cramer', 'ideal_gas' or no argument.");
    }
}

/**
 * \brief Pure tone energy attenuation coefficient, following ISO 9613-1:1993.
 *
 * \param temperature
 *      Temperature in degree Celsius. Must be greater than -73 °C for accuracy of +/-50% (and success).
 *      Must be in the range of -20 °C to 50 °C for accuracy of +/-10%.
 * \param frequency
 *      Frequency in Hz. Must be greater than 50 Hz.
 *      Frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa for accuracy of +/-50%.
 * \param relative_humidity
 *      Relative humidity in the range of 0 to 1.
 * \param atmospheric_pressure
 *      Atmospheric pressure in Pascal. Must be less than 200 kPa.
 *      Frequency-to-pressure ratio: 4 x 10-4 HzjPa to 10 Hz/Pa for accuracy of +/-50%.
 *
 * \return
 *      Energy decay coefficient m in 1/m.
 */
template <typename Value>
Value energy_attenuation_coefficient(Value temperature,
                            Value frequency,
                            Value relative_humidity,
                            Value atmospheric_pressure) {

    // Range checks rely on scalar branching and are only meaningful (and
    // compilable) for scalar Value types. Vectorized/JIT callers (e.g. the
    // acoustic path integrator, where each lane may carry a different
    // frequency) skip validation and rely on the formula below, which is
    // branch-free.
    if constexpr (!dr::is_jit_v<Value>) {
        if (temperature < -73.0f) {
            throw std::invalid_argument("Temperature must be above -73 °C for accuracy of +/-50% (and success)");
        }
        else if (frequency < 50.0f) {
            throw std::invalid_argument("Frequency in Hz. Must be greater than 50 Hz.");
        }
        else if (atmospheric_pressure > 200000.0f)
        {
            throw std::invalid_argument("Atmospheric pressure in Pascal. Must be less than 200 kPa.");
        }
        else if (frequency/atmospheric_pressure < 0.0004f || frequency/atmospheric_pressure > 10.0f) {
            throw std::invalid_argument(" Frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa for accuracy of +/-50%. (and success)");
        }
    }

    constexpr float p_r  = 101325.0f; // reference atmospheric pressure
    constexpr float T_0  = 293.15f;   // reference temperature (20°C)
    constexpr float T_01 = 273.16f;   // triple point temperature of water

    Value T = temperature + 273.15f;

    // saturation vapour pressure ratio p_sat/p_r (ISO 9613-1)
    Value p_sat_ratio = dr::pow(Value(10.f),
        Value(-6.8346f) * dr::pow(Value(T_01) / T, Value(1.261f)) + Value(4.6151f));

    // molar concentration of water vapor as a percentage (Eq. B.1)
    Value h = (relative_humidity * 100.f) * p_sat_ratio * (atmospheric_pressure / p_r);

    // Oxygen relaxation frequency
    Value f_rO = (atmospheric_pressure / p_r) *
        (24.f + 4.04e4f * h * (0.02f + h) / (0.391f + h));

    // Nitrogen relaxation frequency
    Value f_rN = (atmospheric_pressure / p_r) * dr::pow(T / T_0, Value(-0.5f)) *
        (9.f + 280.f * h * dr::exp(Value(-4.17f) *
            (dr::pow(T / T_0, Value(-1.f / 3.f)) - 1.f)));

    // air attenuation
    Value f2 = frequency * frequency;
    Value alpha = 8.686f * f2 *
        (1.84e-11f * (p_r / atmospheric_pressure) * dr::sqrt(T / T_0) +
        dr::pow(T / T_0, Value(-2.5f)) *
            (0.01275f * dr::exp(Value(-2239.1f) / T) / (f_rO + f2 / f_rO) +
            0.1068f * dr::exp(Value(-3352.f) / T) / (f_rN + f2 / f_rN)));

    // 10*log10(e) = 10/ln(10) ≈ 4.342944819f
    return alpha / 4.342944819f;
}

/**
 * \brief Apply pure tone attenuation to an energy time curve (ETC).
 *
 * Multiplies each time bin of the ETC with a frequency-dependent exponential
 * decay factor derived from the distance the sound has travelled and the
 * air attenuation coefficient computed for each frequency band, following
 * ISO 9613-1:1993.
 *
 * From Python, this is a drop-in post-processing step for the output of
 * ``mitsuba.render()``: it accepts a ``TensorXf`` of arbitrary shape (not
 * just a flat/2-D array) directly, and returns a ``TensorXf`` of that exact
 * same shape and type, as long as its total size is a multiple of
 * ``len(frequencies)``.
 *
 * \param etc
 *      Input energy time curve as a 2-D array of shape
 *      (n_time_bins, n_frequencies). From Python, the output of
 *      ``mitsuba.render()`` (a ``TensorXf`` of arbitrary shape, e.g. also
 *      including a frequency axis) can be passed directly, e.g.
 *      ``apply_pure_tone_attenuation(etc=mitsuba.render(scene, sensor=microphone, integrator=integrator), ...)``.
 * \param sampling_rate
 *      Sampling rate in Hz used to convert sample indices to times.
 * \param speed_of_sound_ms
 *      Speed of sound in m/s (use the return value of speed_of_sound()).
 * \param temperature
 *      Temperature in degree Celsius.
 * \param frequencies
 *      Center frequencies in Hz, one value per frequency band. Must have
 *      the same number of entries as \c etc has columns.
 * \param relative_humidity
 *      Relative humidity in the range of 0 to 1.
 * \param atmospheric_pressure
 *      Atmospheric pressure in Pascal.
 *
 * \return
 *      A new vector containing the attenuated ETC with the same layout as
 *      the input (row-major, n_time_bins × n_frequencies). From Python,
 *      when \c etc was a ``TensorXf`` (e.g. straight from
 *      ``mitsuba.render()``), the result is a ``TensorXf`` of that same
 *      shape, ready to be used like any other rendered output (plotted,
 *      saved, compared, etc.).
 */
template <typename Value>
std::vector<Value> apply_pure_tone_attenuation(
        const std::vector<Value>& etc,
        Value sampling_rate,
        Value speed_of_sound_ms,
        Value temperature,
        const std::vector<Value>& frequencies,
        Value relative_humidity,
        Value atmospheric_pressure) {

    size_t n_frequencies = frequencies.size();
    if (n_frequencies == 0 || etc.size() % n_frequencies != 0) {
        throw std::invalid_argument(
            "etc size must be a multiple of frequencies.size().");
    }
    size_t n_time_bins = etc.size() / n_frequencies;

    // Precompute the energy decay coefficient per frequency band
    std::vector<Value> decay(n_frequencies);
    for (size_t f = 0; f < n_frequencies; ++f) {
        decay[f] = energy_attenuation_coefficient<Value>(
            temperature, frequencies[f], relative_humidity, atmospheric_pressure);
    }

    std::vector<Value> etc_attenuated(n_time_bins * n_frequencies);

    for (size_t t = 0; t < n_time_bins; ++t) {
        // Time and distance for this bin
        Value time     = static_cast<Value>(t) / sampling_rate;
        Value distance = time * speed_of_sound_ms;
        if (distance > 10000.0f) {
            throw std::invalid_argument(
                "Distance must be smaller than 10 km for accuracy of +/-50% (and success).\n"
                "Calculated distance was: " + std::to_string(distance) + "\n"
                "From speed: " + std::to_string(speed_of_sound_ms) +
                " m/s and time: " + std::to_string(time) + " s. With s = t * v.");
        }

        for (size_t f = 0; f < n_frequencies; ++f) {
            // Exponential decay:  exp(-distance * m_f)
            Value attenuation = dr::exp(-distance * decay[f]);
            etc_attenuated[t * n_frequencies + f] =
                etc[t * n_frequencies + f] * attenuation;
        }
    }

    return etc_attenuated;
}

NAMESPACE_END(acoustic)
NAMESPACE_END(mitsuba)