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
          D(acoustic, speed_of_sound));

    m.def("apply_pure_tone_attenuation",
          [](nb::object etc,
             float sampling_rate,
             float speed_of_sound_ms,
             float temperature,
             const std::vector<float> &frequencies,
             float relative_humidity,
             float atmospheric_pressure) {
              // Accept plain lists as well as array-like objects (numpy
              // arrays, drjit/mitsuba tensors such as the TensorXf returned
              // by mi.render): flatten row-major via numpy, any trailing
              // size-1 channel dimension disappears naturally under ravel().
              nb::object np = nb::module_::import_("numpy");
              nb::object arr = np.attr("asarray")(etc);
              nb::object shape = arr.attr("shape");
              std::vector<float> etc_flat =
                  nb::cast<std::vector<float>>(arr.attr("ravel")().attr("tolist")());

              std::vector<float> result =
                  acoustic::apply_pure_tone_attenuation<float>(etc_flat,
                                                          sampling_rate,
                                                          speed_of_sound_ms,
                                                          temperature,
                                                          frequencies,
                                                          relative_humidity,
                                                          atmospheric_pressure);

              nb::object out = np.attr("array")(result).attr("reshape")(shape);

              if (nb::hasattr(etc, "numpy")) {
                  // input was a drjit/mitsuba tensor (e.g. the TensorXf
                  // returned by mi.render): reconstruct the same tensor
                  // type with the original shape via its own Python
                  // constructor, so this is a drop-in replacement.
                  return etc.type()(out);
              }

              // Plain list/tuple/numpy array input: return a numpy array
              // reshaped to match the input.
              return out;
          },
          "etc"_a,
          "sampling_rate"_a,
          "speed_of_sound_ms"_a,
          "temperature"_a,
          "frequencies"_a,
          "relative_humidity"_a,
          "atmospheric_pressure"_a,
          D(acoustic, apply_pure_tone_attenuation));
}
