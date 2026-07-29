#include <mitsuba/core/acoustic.h>
#include <mitsuba/python/python.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

MI_PY_EXPORT(acoustic) {
    MI_PY_IMPORT_TYPES()

    m.def("speed_of_sound",
          [](float temperature,
             float relative_humidity,
             float atmospheric_pressure,
             float saturation_vapor_pressure,
             float co2_ppm,
             const std::string &method) {
              return acoustic::speed_of_sound<float>(temperature,
                                                      relative_humidity,
                                                      atmospheric_pressure,
                                                      saturation_vapor_pressure,
                                                      co2_ppm,
                                                      method);
          },
          "temperature"_a,
          "relative_humidity"_a = std::numeric_limits<float>::quiet_NaN(),
          "atmospheric_pressure"_a = std::numeric_limits<float>::quiet_NaN(),
          "saturation_vapor_pressure"_a = -1.f,
          "co2_ppm"_a = std::numeric_limits<float>::quiet_NaN(),
          "method"_a = std::string("auto"),
          "Return the speed of sound in air. Chooses calculation method based on input parameters.");

    m.def("apply_pure_tone_attenuation",
          [](const std::vector<float> &etc,
             float sampling_rate,
             float speed_of_sound_ms,
             float temperature,
             const std::vector<float> &frequencies,
             float relative_humidity,
             float atmospheric_pressure,
             size_t n_time_bins,
             size_t n_frequencies) {
              return acoustic::apply_pure_tone_attenuation<float>(etc,
                                                             sampling_rate,
                                                             speed_of_sound_ms,
                                                             temperature,
                                                             frequencies,
                                                             relative_humidity,
                                                             atmospheric_pressure,
                                                             n_time_bins,
                                                             n_frequencies);
          },
          "etc"_a,
          "sampling_rate"_a,
          "speed_of_sound_ms"_a,
          "temperature"_a,
          "frequencies"_a,
          "relative_humidity"_a,
          "atmospheric_pressure"_a,
          "n_time_bins"_a,
          "n_frequencies"_a,
          "Apply air attenuation to an energy time curve (ETC), computing the "
          "air attenuation decay coefficients from temperature, frequencies, "
          "relative humidity and atmospheric pressure (ISO 9613-1:1993).");
}
