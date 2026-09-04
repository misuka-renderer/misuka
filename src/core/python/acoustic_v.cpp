#include <drjit/dynamic.h>
#include <drjit/tensor.h>
#include <mitsuba/core/acoustic.h>
#include <mitsuba/python/python.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

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
          "method"_a = std::string("auto"),
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

    m.def("apply_pure_tone_attenuation",
          [](nb::object etc,
             Float sampling_rate,
             Float speed_of_sound_ms,
             Float temperature,
             const std::vector<Float> &frequencies,
             Float relative_humidity,
             Float atmospheric_pressure) -> nb::object {
              // A TensorXf (e.g. straight from mi.render(), possibly
              // gradient-tracked under an AD variant) is handled natively:
              // .array() gives its flat buffer *without* detaching it from
              // any AD graph, and the result is rebuilt into a TensorXf of
              // the same shape the same way -- gradients survive the round
              // trip end to end.
              if (nb::isinstance<TensorXf>(etc)) {
                  const TensorXf &tensor = nb::cast<const TensorXf &>(etc);
                  auto result = acoustic::apply_pure_tone_attenuation<Float>(
                      tensor.array(), sampling_rate, speed_of_sound_ms,
                      temperature, frequencies, relative_humidity,
                      atmospheric_pressure);
                  return nb::cast(TensorXf(result, tensor.ndim(), tensor.shape().data()));
              }

              // Plain list/tuple/numpy array input: never carries an AD
              // graph to begin with, so flatten/reshape via numpy as
              // before (handles arbitrary input shapes uniformly; any
              // trailing size-1 channel dimension disappears naturally
              // under ravel()).
              nb::object np = nb::module_::import_("numpy");
              nb::object arr = np.attr("asarray")(etc);
              nb::object shape = arr.attr("shape");
              auto etc_flat = nb::cast<mitsuba::DynamicBuffer<Float>>(arr.attr("ravel")());

              auto result = acoustic::apply_pure_tone_attenuation<Float>(
                  etc_flat, sampling_rate, speed_of_sound_ms, temperature,
                  frequencies, relative_humidity, atmospheric_pressure);

              return np.attr("asarray")(nb::cast(result)).attr("reshape")(shape);
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
