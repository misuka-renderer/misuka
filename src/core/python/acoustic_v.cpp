#include <drjit/dynamic.h>
#include <mitsuba/core/acoustic.h>
#include <mitsuba/python/python.h>
#include <nanobind/stl/string.h>

MI_PY_EXPORT(acoustic) {
    MI_PY_IMPORT_TYPES()

    m.def("speed_of_sound",
          [](Float temperature,
             Float relative_humidity,
             Float atmospheric_pressure,
             Float saturation_vapor_pressure,
             Float co2_ppm,
             const std::string &method) {
              return acoustic::speed_of_sound<Float>(temperature,
                                                      relative_humidity,
                                                      atmospheric_pressure,
                                                      saturation_vapor_pressure,
                                                      co2_ppm,
                                                      method);
          },
          "temperature"_a,
          "relative_humidity"_a = std::numeric_limits<float>::quiet_NaN(),
          "atmospheric_pressure"_a = std::numeric_limits<float>::quiet_NaN(),
          "saturation_vapor_pressure"_a = std::numeric_limits<float>::quiet_NaN(),
          "co2_ppm"_a = std::numeric_limits<float>::quiet_NaN(),
          "method"_a = std::string("simple"),
          D(acoustic, speed_of_sound));

    m.def("energy_attenuation_coefficient",
          [](Float temperature,
             Float frequency,
             Float relative_humidity,
             Float atmospheric_pressure) {
              return acoustic::energy_attenuation_coefficient<Float>(
                  temperature, frequency, relative_humidity, atmospheric_pressure);
          },
          "temperature"_a,
          "frequency"_a,
          "relative_humidity"_a,
          "atmospheric_pressure"_a,
          D(acoustic, energy_attenuation_coefficient));
}
