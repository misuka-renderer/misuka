#pragma once

#include <mitsuba/mitsuba.h>
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
 * \brief Calculation methods and automatic method selector for the speed of sound
 *
 * This Function calculates the speed of sound in air 
 *
 * \param temperature
 *      The temperature in degree Celsius.
 * \param relative_humidity
 *     Relative humidity in the range of 0 to 1.
 * \param atmospheric_pressure
 *    Atmospheric pressure in Pascal
 * \param saturation_vapor_pressure
 *    Saturation vapor pressure in Pascal.
 * \param co2_ppm
 *   CO2 concentration in parts per million (ppm).
 * \param method
 *  The method to use for the calculation. Possible values are:
 *   - "auto" (default): automatically selects the method based on the types of input parameters.
 *   - "simple" The calculation follows ISO 9613-1 (Formula A.5).
 *   - "ideal_gas": calculates based on chapter 6.3 in V. E. Ostashev and D. K. Wilson, Acoustics in Moving Inhomogeneous Media, 2nd ed. London: CRC Press, 2015. doi: 10.1201/b18922.
 *   - "cramer": calculates using Cramers method described in O. Cramer, “The variation of the specific heat ratio and the speed of sound in air with temperature, pressure, humidity, and CO2 concentration,” The Journal of the Acoustical Society of America, vol. 93, no. 5, pp. 2510-2516, May 1993, doi: 10.1121/1.405827.
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
    }

    if (selected_method == "simple") {
        if (temperature < -20.0f || temperature > 50.0f) {
            throw std::invalid_argument("Temperature out of range for simple method (-20°C to 50°C).");
        }
        return 343.2f * dr::sqrt((temperature + 273.15f) / 293.15f);
    }

    else if (relative_humidity < 0.0f || relative_humidity > 1.0f) {
        throw std::invalid_argument("Relative humidity must be in the range of 0 to 1.");
    }
    else if (atmospheric_pressure < 0.0f) {
        throw std::invalid_argument("Atmospheric pressure must be non-negative.");
    }

    else if (selected_method == "ideal_gas") {

        float temperature_kelvin = static_cast<float>(temperature) + 273.15f;

        // Ideal gas calculation based on Ostashev and Wilson
        float R = 8.314f; // J/(mol*K)
        float gamma_a = 1.400f;
        float gamma_w = 1.330f;
        float mu_a = 28.97f * 1e-3f; //kg/mol
        float mu_w = 18.02f * 1e-3f; //kg/mol
        float R_a = R / mu_a;
        float p = static_cast<float>(atmospheric_pressure);

        // saturation_vapor_pressure: -1 bedeutet "nicht angegeben"
        Value e_s;
        if (saturation_vapor_pressure < Value(0)) {
            e_s = Value(6.1094f) * dr::exp((Value(17.625f) * temperature) /
                                           (temperature + Value(243.04f)));
            e_s = Value(100) * e_s; // hPa → Pa
        } else {
            e_s = saturation_vapor_pressure;
        }

        float e = p * static_cast<float>(relative_humidity);
        float alpha = mu_a / mu_w;
        float delta = (1.0f - (1.0f / gamma_a)) / (1.0f - (1.0f / gamma_w));
        float nu = (gamma_a - 1.0f) / (gamma_w - 1.0f);
        float C = (e / static_cast<float>(atmospheric_pressure)) /
                  (alpha * (1.0f - e / static_cast<float>(atmospheric_pressure)));

        return dr::sqrt(gamma_a * R_a * temperature_kelvin *
                        (1.0f + (alpha * (1.0f + delta - nu) - 1.0f) * C));
    } 
    else if (selected_method == "cramer") {
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
        if (is_missing_value(atmospheric_pressure)) {
            atmospheric_pressure = Value(101325.0f);
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

    } else {
        throw std::invalid_argument("Invalid method specified for speed of sound calculation. Valid options are 'auto', 'simple', 'cramer', 'ideal_gas' or no argument.");
    }
}

/**
 * \brief Apply air attenuation to an energy time curve (ETC).
 *
 * Multiplies each time bin of the ETC with a frequency-dependent exponential
 * decay factor derived from the distance the sound has travelled.
 *
 * \param etc
 *      Input energy time curve as a 2-D array of shape
 *      (n_time_bins, n_frequencies).
 * \param sampling_rate
 *      Sampling rate in Hz used to convert sample indices to times.
 * \param speed_of_sound_ms
 *      Speed of sound in m/s (use the return value of speed_of_sound()).
 * \param air_attenuation_decay
 *      Energy decay coefficients m in 1/m, one value per frequency band.
 *      Must have the same number of entries as \p etc has columns.
 * \param n_time_bins
 *      Number of time bins (rows) in \p etc.
 * \param n_frequencies
 *      Number of frequency bands (columns) in \p etc.
 *
 * \return
 *      A new vector containing the attenuated ETC with the same layout as
 *      the input (row-major, n_time_bins × n_frequencies).
 */
template <typename Value>
std::vector<Value> apply_air_attenuation(
        const std::vector<Value>& etc,
        Value sampling_rate,
        Value speed_of_sound_ms,
        const std::vector<Value>& air_attenuation_decay,
        size_t n_time_bins,
        size_t n_frequencies) {

    if (etc.size() != n_time_bins * n_frequencies) {
        throw std::invalid_argument(
            "etc size does not match n_time_bins * n_frequencies.");
    }
    if (air_attenuation_decay.size() != n_frequencies) {
        throw std::invalid_argument(
            "air_attenuation_decay size does not match n_frequencies.");
    }

    std::vector<Value> etc_attenuated(n_time_bins * n_frequencies);

    for (size_t t = 0; t < n_time_bins; ++t) {
        // Time and distance for this bin
        Value time     = static_cast<Value>(t) / sampling_rate;
        Value distance = time * speed_of_sound_ms;

        for (size_t f = 0; f < n_frequencies; ++f) {
            // Exponential decay:  exp(-distance * m_f)
            Value attenuation = dr::exp(-distance * air_attenuation_decay[f]);
            etc_attenuated[t * n_frequencies + f] =
                etc[t * n_frequencies + f] * attenuation;
        }
    }

    return etc_attenuated;
}

NAMESPACE_END(acoustic)
NAMESPACE_END(mitsuba)