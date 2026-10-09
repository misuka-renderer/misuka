#pragma once

#include <mitsuba/mitsuba.h>
#include <mitsuba/core/logger.h>
#include <drjit/math.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

NAMESPACE_BEGIN(mitsuba)
NAMESPACE_BEGIN(acoustic)

template <typename Value>
inline bool is_missing_value(const Value &value) {
    if constexpr (dr::is_jit_v<Value>) {
        // JIT/AD arrays (e.g. Float under an *_ad_* variant): these
        // parameters are always broadcast scalars in practice (one
        // physical value per render, never per-sample), so a dr::all()
        // reduction to a host bool is cheap and correct.
        return dr::all(dr::isnan(value));
    } else if constexpr (std::is_floating_point_v<Value>) {
        return std::isnan(static_cast<double>(value));
    } else {
        return false;
    }
}

// -----------------------------------------------------------------------
//! @{ \name Standard atmospheric conditions
// -----------------------------------------------------------------------

/**
 * \brief "Standard medium" reference atmospheric conditions.
 *
 * Used by the acoustic_path/acoustic_ad integrators to fill in whichever
 * fields of an ``acoustic_medium`` dict were left unspecified, so that a
 * (possibly empty) ``acoustic_medium`` always describes a complete,
 * physically valid medium. Room-temperature values, internally consistent
 * (\ref acoustic_medium_standard_saturation_vapor_pressure is within 0.2%
 * of the Magnus-formula estimate at
 * \ref acoustic_medium_standard_temperature, see
 * speed_of_sound_ideal_gas()).
 */
constexpr float acoustic_medium_standard_temperature               = 20.0f;    ///< degree Celsius
constexpr float acoustic_medium_standard_relative_humidity         = 0.6f;     ///< in the range 0 to 1
constexpr float acoustic_medium_standard_atmospheric_pressure      = 101325.0f; ///< Pascal
constexpr float acoustic_medium_standard_saturation_vapor_pressure = 2333.0f;  ///< Pascal
constexpr float acoustic_medium_standard_co2_ppm                   = 428.73f;  ///< parts per million

//! @}
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
//! @{ \name Speed of sound calculation
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
Value speed_of_sound_simple(const Value temperature) {
    // Range check relies on scalar branching and is only meaningful (and
    // compilable) for scalar Value types; see energy_attenuation_coefficient()
    // for the established pattern.
    if constexpr (!dr::is_jit_v<Value>) {
        if (temperature < -20.0f || temperature > 50.0f) {
            throw std::invalid_argument("Temperature out of range for simple method (-20°C to 50°C).");
        }
    }
    return Value(343.2f) * dr::sqrt((temperature + Value(273.15f)) / Value(293.15f));
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
 *    Atmospheric pressure in Pascal, must be positive.
 * \param saturation_vapor_pressure
 *    Saturation vapor pressure in Pascal. Missing values (see
 *    is_missing_value()) are estimated from \c temperature via the Magnus
 *    formula.
 *
 * \return
 *      The speed of sound in meters per second
 */
template <typename Value>
Value speed_of_sound_ideal_gas(const Value temperature,
                               const Value relative_humidity,
                               const Value atmospheric_pressure,
                               const Value saturation_vapor_pressure = Value(std::numeric_limits<float>::quiet_NaN())) {
    if constexpr (!dr::is_jit_v<Value>) {
        if (relative_humidity < 0.0f || relative_humidity > 1.0f) {
            throw std::invalid_argument("Relative humidity must be in the range of 0 to 1.");
        }
        if (atmospheric_pressure <= 0.0f) {
            throw std::invalid_argument("Atmospheric pressure must be positive.");
        }
    }

    Value temperature_kelvin = temperature + Value(273.15f);

    // Ideal gas calculation based on Ostashev and Wilson
    float R = 8.314f; // J/(mol*K)
    float gamma_a = 1.400f;
    float gamma_w = 1.330f;
    float mu_a = 28.97f * 1e-3f; //kg/mol
    float mu_w = 18.02f * 1e-3f; //kg/mol
    float R_a = R / mu_a;

    // A missing saturation_vapor_pressure is estimated from temperature
    // using the Magnus formula (see e.g. O. A. Alduchov and R. E. Eskridge,
    // "Improved Magnus Form Approximation of Saturation Vapor Pressure,"
    // J. Appl. Meteor., 1996).
    Value e_s;
    if (is_missing_value(saturation_vapor_pressure)) {
        e_s = Value(6.1094f) * dr::exp((Value(17.625f) * temperature) /
                                       (temperature + Value(243.04f)));
        e_s = Value(100) * e_s; // hPa → Pa
    } else {
        e_s = saturation_vapor_pressure;
    }

    Value e = e_s * relative_humidity; // water vapor partial pressure
    float alpha = mu_a / mu_w;
    float delta = (1.0f - (1.0f / gamma_a)) / (1.0f - (1.0f / gamma_w));
    float nu = (gamma_a - 1.0f) / (gamma_w - 1.0f);
    Value C = (e / atmospheric_pressure) /
              (alpha * (Value(1.0f) - e / atmospheric_pressure));

    return dr::sqrt(gamma_a * R_a * temperature_kelvin *
                    (Value(1.0f) + (alpha * (1.0f + delta - nu) - 1.0f) * C));
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
 *    Atmospheric pressure in Pascal, must be positive and in the range
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
Value speed_of_sound_cramer(const Value temperature,
                            const Value relative_humidity,
                            Value atmospheric_pressure,
                            Value co2_ppm = Value(std::numeric_limits<float>::quiet_NaN())) {
    if constexpr (!dr::is_jit_v<Value>) {
        if (relative_humidity < 0.0f || relative_humidity > 1.0f) {
            throw std::invalid_argument("Relative humidity must be in the range of 0 to 1.");
        }
        if (atmospheric_pressure <= 0.0f) {
            throw std::invalid_argument("Atmospheric pressure must be positive.");
        }
    }

    // Resolve missing values to their defaults before validating ranges, so
    // that the value actually used in the calculation is the one checked.
    if (is_missing_value(atmospheric_pressure)) {
        atmospheric_pressure = Value(101325.0f); // standard atmosphere
    }
    if (is_missing_value(co2_ppm)) {
        co2_ppm = Value(428.73f); // NOAA GML global mean, doi.org/10.15138/9N0H-ZH07
    }

    if constexpr (!dr::is_jit_v<Value>) {
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
    }

    Value x_c = co2_ppm * 1e-6f; // Convert ppm to mole fraction
    Value T = temperature + Value(273.15f); // Convert to Kelvin
    Value p = atmospheric_pressure; // in Pa

    Value p_sv = dr::exp(1.2811805e-5f * T * T - 1.9509874e-2f * T +
                         Value(34.04926034f) - 6.3536311e3f / T);

    Value x_w = relative_humidity * p_sv / p; // Mole fraction of water vapor

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
 * \brief Calculation methods for the speed of sound
 *
 * Differentiable: under an ``*_ad_*`` variant, gradients set on
 * \c temperature, \c relative_humidity, \c atmospheric_pressure,
 * \c saturation_vapor_pressure or \c co2_ppm propagate through to the
 * returned speed of sound.
 *
 * This function calculates the speed of sound in air, using one of the
 * following methods:
 *
 * "simple": following ISO 9613-1 (Formula A.5),
 * \f$c = 343.2 \cdot \sqrt{(T + 273.15) / 293.15}\f$. Only uses \c
 * temperature (\f$T\f$), which must be in the range of -20°C to 50°C.
 *
 * "ideal_gas": speed of sound of a humid-air mixture treated as an
 * ideal gas, based on chapter 6.3 in V. E. Ostashev and D. K. Wilson,
 * Acoustics in Moving Inhomogeneous Media, 2nd ed. London: CRC Press,
 * 2015. doi: 10.1201/b18922,
 * \f$c = \sqrt{\gamma_a R_a T_K (1 + (\alpha (1 + \delta - \nu) - 1)
 * C)}\f$, where \f$T_K\f$ is \c temperature in Kelvin, \f$R_a\f$
 * the specific gas constant of dry air, \f$\gamma_a, \gamma_w\f$ the
 * heat capacity ratios of dry air and water vapor, \f$\alpha\f$ the
 * ratio of their molar masses, and \f$C\f$ the water vapor mole
 * fraction term derived from \c relative_humidity, \c
 * atmospheric_pressure and \c saturation_vapor_pressure; see
 * speed_of_sound_ideal_gas() for the exact constants.
 *
 * "cramer": O. Cramer, "The variation of the specific heat ratio and
 * the speed of sound in air with temperature, pressure, humidity, and
 * CO2 concentration," The Journal of the Acoustical Society of
 * America, vol. 93, no. 5, pp. 2510-2516, May 1993,
 * doi: 10.1121/1.405827, an empirical quadratic fit, the sum of:
 *
 * <ul>
 *   <li>a temperature-only term \f$(a_0 + a_1 T + a_2 T^2)\f$</li>
 *   <li>a water-vapor term \f$(a_3 + a_4 T + a_5 T^2) x_w\f$</li>
 *   <li>a pressure term \f$(a_6 + a_7 T + a_8 T^2) p\f$</li>
 *   <li>a CO2 term \f$(a_9 + a_{10} T + a_{11} T^2) x_c\f$</li>
 *   <li>squared terms \f$a_{12} x_w^2 + a_{13} p^2 + a_{14} x_c^2\f$</li>
 *   <li>a cross term \f$a_{15} x_c\, p\, x_w\f$</li>
 * </ul>
 *
 * where \f$x_w\f$ is the water vapor mole fraction (derived from \c
 * relative_humidity and \f$p\f$), \f$p\f$ is \c atmospheric_pressure
 * and \f$x_c\f$ is the CO2 mole fraction (derived from \c co2_ppm);
 * the 16 empirical coefficients \f$a_0 \ldots a_{15}\f$ are given in
 * speed_of_sound_cramer(). Requires \c temperature in the range of
 * 0°C to 30°C and \c atmospheric_pressure in the range of 75,000 Pa
 * to 102,000 Pa.
 *
 * \param temperature
 *      The temperature in degree Celsius.
 * \param relative_humidity
 *      Relative humidity in the range of 0 to 1.
 * \param atmospheric_pressure
 *      Atmospheric pressure in Pascal, must be positive. For "cramer",
 *      a missing value (see is_missing_value()) defaults to 101,325 Pa
 *      (standard atmosphere).
 * \param saturation_vapor_pressure
 *      Saturation vapor pressure in Pascal. Only used by the "ideal_gas"
 *      method. A missing value (see is_missing_value()) is estimated from
 *      \c temperature via the Magnus formula (see e.g. O. A. Alduchov and
 *      R. E. Eskridge, "Improved Magnus Form Approximation of Saturation
 *      Vapor Pressure," J. Appl. Meteor., 1996).
 * \param co2_ppm
 *      CO2 concentration in parts per million (ppm). Only used by the
 *      "cramer" method, must be in the range of 0 ppm to 10,000 ppm. A
 *      missing value (see is_missing_value()) defaults to 428.73 ppm, the
 *      global monthly mean for 2026-07 reported by NOAA GML
 *      (https://doi.org/10.15138/9N0H-ZH07, retrieved 2026-08-28).
 * \param method
 *      The method to use for the calculation: "simple" (default),
 *      "ideal_gas" or "cramer".
 *
 * \return
 *      The speed of sound in meters per second
 */
template <typename Value>
Value speed_of_sound(const Value temperature,
                     const Value relative_humidity,
                     Value atmospheric_pressure,
                     const Value saturation_vapor_pressure,
                     const Value co2_ppm,
                     const std::string& method = "simple") {

    // input validation - at least temperature must be provided
    if (is_missing_value(temperature)) {
        throw std::invalid_argument("Temperature must be provided.");
    }

    if (method == "simple") {
        return speed_of_sound_simple<Value>(temperature);
    } else if (method == "ideal_gas") {
        return speed_of_sound_ideal_gas<Value>(temperature, relative_humidity,
                                               atmospheric_pressure, saturation_vapor_pressure);
    } else if (method == "cramer") {
        return speed_of_sound_cramer<Value>(temperature, relative_humidity,
                                            atmospheric_pressure, co2_ppm);
    } else {
        throw std::invalid_argument("Invalid method specified for speed of sound calculation. "
                                    "Valid options are 'simple', 'ideal_gas' or 'cramer'.");
    }
}

/**
* \brief Pure tone energy attenuation coefficient following ISO 9613-1:1993.
* Calculates the energy attenuation coefficient in air for a given
* frequency, temperature, relative humidity and atmospheric pressure.
* The attenuation coefficient in dB/m is
* \f$\alpha = 8.686 f^2 (\alpha_{cl} + \alpha_{vib})\f$, consisting of
* a classical absorption term \f$\alpha_{cl}\f$ and a molecular
* relaxation term \f$\alpha_{vib}\f$:
* \f$\alpha_{cl} = 1.84 \cdot 10^{-11} (p_r / p_a) \sqrt{T / T_0}\f$
* \f$\alpha_{vib} = (T / T_0)^{-5/2}(\alpha_O + \alpha_N)\f$
* where \f$\alpha_O\f$ and \f$\alpha_N\f$ are the oxygen and nitrogen
* relaxation contributions. The relaxation frequencies depend on
* atmospheric pressure, temperature and water vapor concentration.
* Here \f$f\f$ is \c frequency, \f$T\f$ is temperature in Kelvin,
* \f$T_0 = 293.15\f$ K and \f$p_r = 101325\f$ Pa are the reference
* temperature and pressure, and \f$p_a\f$ is \c atmospheric_pressure.
* The water vapor concentration is derived from \c relative_humidity
* and the saturation vapor pressure.
* The returned coefficient is converted from dB/m to the natural
* energy decay coefficient in 1/m via
* \f$\alpha / (10 / \ln 10)\f$.
* Validity ranges according to ISO 9613-1:
*
*<ul>
*  <li>\c temperature must be greater than -73 °C for an accuracy of
*      +/-50% and is in the range of -20 °C to 50 °C for an accuracy of +/-10%.</li>
*  <li>\c frequency must be greater than 50 Hz.</li>
*  <li>\c atmospheric_pressure must be less than 200 kPa.</li>
*</ul>
*
* \param temperature
*      Temperature in degree Celsius.
* \param frequency
*      Frequency in Hz.
* \param relative_humidity
*      Relative humidity in the range of 0 to 1.
* \param atmospheric_pressure
*      Atmospheric pressure in Pascal.
* \return
*      Energy decay coefficient in 1/m.
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
            throw std::invalid_argument("Temperature must be above -73 °C (ISO 9613-1 validity range).");
        }
        else if (frequency < 50.0f) {
            throw std::invalid_argument("Frequency must be at least 50 Hz (ISO 9613-1 validity range).");
        }
        else if (atmospheric_pressure > 200000.0f)
        {
            throw std::invalid_argument("Atmospheric pressure must be less than 200 kPa (ISO 9613-1 validity range).");
        }
        else if (frequency/atmospheric_pressure < 0.0004f || frequency/atmospheric_pressure > 10.0f) {
            throw std::invalid_argument("Frequency-to-pressure ratio must be between 4e-4 Hz/Pa and 10 Hz/Pa (ISO 9613-1 validity range).");
        }
    }

    constexpr float p_r  = 101325.0f; // reference atmospheric pressure
    constexpr float T_0  = 293.15f;   // reference temperature (20°C)
    constexpr float T_01 = 273.16f;   // triple point temperature of water

    Value T = temperature + 273.15f;

    // saturation vapor pressure ratio p_sat/p_r (ISO 9613-1)
    Value p_sat_ratio = dr::pow(Value(10.f),
        Value(-6.8346f) * dr::pow(Value(T_01) / T, Value(1.261f)) + Value(4.6151f));

    // molar concentration of water vapor as a percentage (Eq. B.1)
    Value h = (relative_humidity * 100.f) * p_sat_ratio / (atmospheric_pressure / p_r);

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

NAMESPACE_END(acoustic)
NAMESPACE_END(mitsuba)