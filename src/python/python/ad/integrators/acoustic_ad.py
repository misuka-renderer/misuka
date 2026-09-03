from __future__ import annotations # Delayed parsing of type annotations
from typing import Optional, Tuple, Callable, Any, Union

import drjit as dr
import mitsuba as mi
import gc

from .common import RBIntegrator, mis_weight

# "Standard medium" reference atmospheric conditions -- mirrors the
# acoustic_medium_standard_* constants in include/mitsuba/core/acoustic.h.
# Used to fill in whichever fields of an 'acoustic_medium' dict were left
# unspecified, so a (possibly empty) 'acoustic_medium' always describes a
# complete, physically valid medium. Room-temperature values, internally
# consistent (ACOUSTIC_MEDIUM_STANDARD_SATURATION_VAPOR_PRESSURE is within
# 0.2% of the Magnus-formula estimate at
# ACOUSTIC_MEDIUM_STANDARD_TEMPERATURE).
ACOUSTIC_MEDIUM_STANDARD_TEMPERATURE = 25.0                # degree Celsius
ACOUSTIC_MEDIUM_STANDARD_RELATIVE_HUMIDITY = 0.6           # in the range 0 to 1
ACOUSTIC_MEDIUM_STANDARD_ATMOSPHERIC_PRESSURE = 101825.0   # Pascal
ACOUSTIC_MEDIUM_STANDARD_SATURATION_VAPOR_PRESSURE = 3167.0  # Pascal
ACOUSTIC_MEDIUM_STANDARD_CO2_PPM = 400.0                   # parts per million

class AcousticADIntegrator(RBIntegrator):
    r"""
    .. _integrator-acoustic_ad:

    Acoustic AD Integrator (:monosp:`acoustic_ad`)
    ---------------------------------------------------

    .. pluginparameters::

     * - speed_of_sound
       - |float|
       - Speed of sound in meters per second. If set explicitly, this value
         is always used, regardless of ``acoustic_medium``. If both
         ``speed_of_sound`` and ``acoustic_medium`` are given, a warning is
         logged. (Default: 343.0, unless overridden by ``acoustic_medium``)

     * - acoustic_medium
       - |dict|
       - Optional dictionary describing the propagation medium (air). See
         :py:func:`mitsuba.acoustic.speed_of_sound` and
         :py:func:`mitsuba.acoustic.apply_pure_tone_attenuation` for the
         recognized fields and their meaning. The medium fields
         (``temperature``, ``relative_humidity``, ``atmospheric_pressure``,
         ``saturation_vapor_pressure``, ``co2_ppm``) are exposed as
         differentiable parameters via :py:func:`mitsuba.traverse`. Any
         field left unspecified (and every field, if ``acoustic_medium`` is
         given as an empty dict) falls back to a standard/reference medium:
         25°C, 60% relative humidity, 101,825 Pa, 3,167 Pa saturation vapor
         pressure, 400 ppm CO2. Since this always yields a complete medium,
         ``apply_attenuation`` always succeeds and
         ``speed_of_sound_method: "auto"`` always resolves to ``"cramer"``
         (the only method that uses every field) once ``acoustic_medium``
         is given at all, however (in)complete.

     * - max_time
       - |float|
       - Stopping criterion for the maximum propagation time in seconds.
         Paths whose accumulated travel distance exceeds ``max_time *
         speed_of_sound`` are terminated.

     * - max_depth
       - |int|
       - Specifies the longest path depth (where -1
         corresponds to :math:`\infty`). A value of 1 will only render directly
         audible sound sources. 2 will lead to first-order reflections, and so
         on. (Default: -1)

     * - rr_depth
       - |int|
       - Russian roulette path termination is not yet supported. Setting this
         to any value other than the default raises an error. Use
         ``max_energy_loss`` instead. (Default: 100000)

     * - max_energy_loss
       - |float|
       - Maximum energy loss in dB before a path is terminated. Set to -1 to
         disable this criterion. (Default: 60.0)

     * - hide_emitters
       - |bool|
       - Hide directly visible emitters, i.e. skip the direct (line-of-sight)
         contribution from sound sources. (Default: no, i.e. |false|)

     * - is_detached
       - |bool|
       - Whether the sampling strategy should be detached from the optimized
         parameters. (Default: |true|)

     * - track_time_derivatives
       - |bool|
       - Whether to track derivatives with respect to time/distance.
         (Default: |true|)

    This is the base class for differentiable acoustic integrators. It extends
    the :ref:`acoustic path tracer <integrator-acoustic_path>` with automatic
    differentiation (AD) support, enabling gradient-based optimization of scene
    parameters such as material properties.

    Like the acoustic path tracer, it simulates sound propagation by tracing
    paths from the sensor (microphone) to the emitters (sound sources), and
    computes an energy-based impulse response (echogram) by accumulating path
    contributions into time bins determined by the total path length and the
    speed of sound.

    This class is not meant to be used in practice, but mostly exists for
    debugging purposes and as a reference implementation.
    Instead, use :ref:`acoustic_prb <integrator-acoustic_prb>` for
    differentiable rendering of static scenes or
    :ref:`acoustic_prb_threepoint <integrator-acoustic_prb_threepoint>` for
    non-static scenes.

    .. note:: This integrator does not handle participating media or polarized
       rendering. It requires a ``Microphone`` sensor with a ``Tape`` film
       type.

    .. tabs::
        .. code-tab:: python

            'type': 'acoustic_ad',
            'max_time': 1.0,
            'speed_of_sound': 343.0,
            'max_depth': -1,

        .. code-tab:: python
            :name: integrator-acoustic_ad-medium

            'type': 'acoustic_ad',
            'max_time': 1.0,
            'acoustic_medium': {
                'temperature': 20.0,
                'relative_humidity': 0.5,
                'atmospheric_pressure': 101325.0,
                'saturation_vapor_pressure': 3200.0,
                'co2_ppm': 400,
                'speed_of_sound_method': 'auto',
            },
            'max_depth': -1,
    """

    def __init__(self, props):
        super().__init__(props)

        self.max_time    = props.get("max_time")
        if self.max_time is None or self.max_time <= 0.:
            raise ValueError("\"max_time\" must be set to a value greater than zero!")


        speed_of_sound_prop = props.get("speed_of_sound", 343.)
        speed_of_sound_explicit = props.has_property("speed_of_sound")

        # An 'acoustic_medium' dict (see acoustic.h) is only used to derive
        # the speed of sound when 'speed_of_sound' was not set explicitly.
        # Named 'acoustic_medium' (not 'medium') to avoid confusion with
        # mitsuba's existing Medium plugin (participating media).
        #
        # 'acoustic_medium_present' is set by the dict parser whenever the
        # 'acoustic_medium' key was given at all, even as an empty dict --
        # this is what distinguishes "acoustic_medium omitted" (unaffected
        # by any of this, self.has_medium False, unchanged
        # pre-acoustic_medium behavior) from "acoustic_medium given, however
        # (in)complete" (a standard/reference medium, see
        # ACOUSTIC_MEDIUM_STANDARD_* below, fills in whichever fields were
        # left unspecified, so every field is always a concrete value below
        # -- no per-field "was this provided" checks are needed anywhere
        # past this point).
        self.has_medium = props.get("acoustic_medium_present", False)
        medium_temperature = props.get("acoustic_medium_temperature", ACOUSTIC_MEDIUM_STANDARD_TEMPERATURE)
        medium_relative_humidity = props.get("acoustic_medium_relative_humidity", ACOUSTIC_MEDIUM_STANDARD_RELATIVE_HUMIDITY)
        medium_atmospheric_pressure = props.get("acoustic_medium_atmospheric_pressure", ACOUSTIC_MEDIUM_STANDARD_ATMOSPHERIC_PRESSURE)
        medium_saturation_vapor_pressure = props.get("acoustic_medium_saturation_vapor_pressure", ACOUSTIC_MEDIUM_STANDARD_SATURATION_VAPOR_PRESSURE)
        medium_co2_ppm = props.get("acoustic_medium_co2_ppm", ACOUSTIC_MEDIUM_STANDARD_CO2_PPM)
        self.speed_of_sound_method = props.get("acoustic_medium_speed_of_sound_method", "auto")

        if not speed_of_sound_explicit and self.has_medium:
            # Every medium field always has a concrete value (real or
            # standard-default, see above), so "auto" has nothing left to
            # infer from -- it always resolves to "cramer", the most
            # complete of the three models (the only one that uses all of
            # temperature/relative_humidity/atmospheric_pressure/co2_ppm).
            if self.speed_of_sound_method == "auto":
                self.speed_of_sound_method = "cramer"
                mi.Log(mi.LogLevel.Warn,
                       "speed_of_sound: no method specified, defaulting to "
                       f"\"{self.speed_of_sound_method}\".")
        elif speed_of_sound_explicit and self.has_medium:
            mi.Log(mi.LogLevel.Warn,
                   "Both \"speed_of_sound\" and \"acoustic_medium\" were "
                   f"specified: the explicit \"speed_of_sound\" value ({speed_of_sound_prop}) "
                   "is used, \"acoustic_medium\" is ignored for the speed of sound.")

        # Live (mi.Float, not a plain Python float) so gradients set on them
        # via mi.traverse() + an optimizer survive into speed_of_sound() /
        # the attenuation coefficient computed in sample() -- see
        # traverse()/parameters_changed() below.
        self.medium_temperature = mi.Float(medium_temperature)
        self.medium_relative_humidity = mi.Float(medium_relative_humidity)
        self.medium_atmospheric_pressure = mi.Float(medium_atmospheric_pressure)
        self.medium_saturation_vapor_pressure = mi.Float(medium_saturation_vapor_pressure)
        self.medium_co2_ppm = mi.Float(medium_co2_ppm)

        if not speed_of_sound_explicit and self.has_medium:
            self.update_speed_of_sound()
        else:
            self.speed_of_sound = speed_of_sound_prop

        if self.speed_of_sound is None or self.speed_of_sound <= 0.:
            raise ValueError("Property \"speed_of_sound\" must be set to a value greater than zero!")

        # Air attenuation (ISO 9613-1) needs the medium's temperature,
        # relative_humidity and atmospheric_pressure. All three are always
        # concrete once self.has_medium is set (see above), so attenuation
        # is simply on-by-default whenever a medium was given, and forced
        # off when it wasn't (there's nothing to compute it from).
        self.apply_attenuation = self.has_medium and props.get("acoustic_medium_apply_attenuation", True)


        self.is_detached = props.get("is_detached", True)

        self.track_time_derivatives = props.get("track_time_derivatives", True)

        max_depth = props.get('max_depth', -1)
        if max_depth < 0 and max_depth != -1:
            raise ValueError("\"max_depth\" must be set to -1 (infinite) or a value >= 0")

        # Map -1 (infinity) to 2^32-1 bounces
        self.max_depth = max_depth if max_depth != -1 else 0xffffffff

        self.rr_depth = props.get('rr_depth', 100000)
        if self.rr_depth <= 0:
            raise ValueError("\"rr_depth\" must be set to a value greater than zero!")
        if self.rr_depth != 100000:
            raise NotImplementedError("Russian Roulette is not yet implemented! Please use energy loss stopping criterion instead.")

        max_energy_loss = props.get('max_energy_loss', 60.)
        if max_energy_loss <= 0. and max_energy_loss != -1.:
            raise ValueError("\"max_energy_loss\" must be set to -1 (disabled) or a value > 0 (in dB)")

        self.throughput_threshold = \
            0. if max_energy_loss == -1. else 10 ** (-max_energy_loss / 10.)

    def update_speed_of_sound(self):
        """Re-derive self.speed_of_sound from the (live) medium fields,
        using the method resolved once at construction time (see __init__
        and the speed_of_sound_method docs above). Called at construction
        and again from parameters_changed() whenever an optimizer updates
        one of the traversed medium parameters. method= is always one of
        "simple"/"ideal_gas"/"cramer" here (never "auto"), so this does not
        re-trigger auto-detection or its log message."""
        if self.speed_of_sound_method == "simple":
            self.speed_of_sound = mi.acoustic.speed_of_sound(
                temperature=self.medium_temperature, method="simple")
        elif self.speed_of_sound_method == "ideal_gas":
            self.speed_of_sound = mi.acoustic.speed_of_sound(
                temperature=self.medium_temperature,
                relative_humidity=self.medium_relative_humidity,
                atmospheric_pressure=self.medium_atmospheric_pressure,
                saturation_vapor_pressure=self.medium_saturation_vapor_pressure,
                method="ideal_gas")
        elif self.speed_of_sound_method == "cramer":
            self.speed_of_sound = mi.acoustic.speed_of_sound(
                temperature=self.medium_temperature,
                relative_humidity=self.medium_relative_humidity,
                atmospheric_pressure=self.medium_atmospheric_pressure,
                co2_ppm=self.medium_co2_ppm,
                method="cramer")
        else:
            raise ValueError(
                "Invalid method specified for speed of sound calculation. "
                "Valid options are 'auto', 'simple', 'cramer', 'ideal_gas' or no argument.")

    def traverse(self, callback):
        # Only the atmospheric medium parameters are exposed: they are the
        # physically meaningful optimization targets (e.g. inferring
        # atmospheric conditions from an observed echogram). An explicitly
        # set 'speed_of_sound' bypasses 'acoustic_medium' entirely (see
        # __init__) and has no medium fields to expose here.
        if self.has_medium:
            callback.put_parameter("medium_temperature", self.medium_temperature, mi.ParamFlags.Differentiable)
            callback.put_parameter("medium_relative_humidity", self.medium_relative_humidity, mi.ParamFlags.Differentiable)
            callback.put_parameter("medium_atmospheric_pressure", self.medium_atmospheric_pressure, mi.ParamFlags.Differentiable)
            callback.put_parameter("medium_saturation_vapor_pressure", self.medium_saturation_vapor_pressure, mi.ParamFlags.Differentiable)
            callback.put_parameter("medium_co2_ppm", self.medium_co2_ppm, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        if self.has_medium:
            self.update_speed_of_sound()
            # Prevents the JIT from baking these in as compile-time
            # literals across optimizer iterations, matching e.g.
            # roughplastic.cpp's parameters_changed().
            dr.make_opaque(self.medium_temperature, self.medium_relative_humidity,
                            self.medium_atmospheric_pressure,
                            self.medium_saturation_vapor_pressure, self.medium_co2_ppm,
                            self.speed_of_sound)

    def to_string(self):
        md = self.max_depth if self.max_depth != 0xffffffff else -1
        if self.throughput_threshold == 0.:
            max_el = "disabled"
        else:
            max_el = f"{ -10.0 * (dr.log(self.throughput_threshold) / dr.log(10.0)) } dB"

        return (
            f"{type(self).__name__}[\n"
            f"  speed_of_sound = {self.speed_of_sound}\n"
            f"  max_time = {self.max_time}\n"
            f"  max_depth = {md}\n"
            f"  rr_depth = {self.rr_depth}\n"
            f"  max_energy_loss = {max_el}\n"
            f"  hide_emitters = {self.hide_emitters}\n"
            f"]"
        )


    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> mi.TensorXf:
        mi.Log(mi.LogLevel.Info, "Rendering in normal mode")

        if not develop:
            raise RuntimeError("develop=True must be specified when "
                               "invoking AD integrators")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aov_names()
            )

            # Generate a set of rays starting at the sensor,
            # Generate a set of rays starting at the sensor,
            # pos.x stores normalized frequency index, ray.wavelengths stores frequency
            ray, weight, position_sample = self.sample_rays(scene, sensor, sampler)

            # Prepare an ImageBlock as specified by the film
            block = film.create_block()

            # Launch the Monte Carlo sampling process in primal mode
            self.sample(
                scene=scene,
                sampler=sampler,
                sensor=sensor,
                ray=ray,
                block=block,
                position_sample=position_sample,
                active=mi.Bool(True)
            )

            # Explicitly delete any remaining unused variables
            del sampler, ray, weight, position_sample#, L, valid
            gc.collect()

            # Perform the weight division and return an image tensor
            film.put_block(block)
            self.primal_image = film.develop()

            return self.primal_image

    def sample_rays(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Float]:
        """
        Sample a set of primary rays for a given sensor

        Returns a tuple containing

        - the set of sampled rays
        - a ray weight (usually 1 if the sensor's response function is sampled
          perfectly)
        - the continuous positions on the Imageblock associated with each ray.
          The first value contains the frequency index, the second value is not used.
        """

        film = sensor.film()
        film_size = film.crop_size()

        if film.sample_border():
            raise NotImplementedError("Acoustic sampling does not support border sampling")

        if not (film.crop_offset()[0] == 0 and film.crop_offset()[1] == 0):
            raise NotImplementedError("Acoustic sampling does not support crop offset")

        spp = sampler.sample_count()

        # Compute frequency indices for the rays
        idx = dr.arange(mi.UInt32, film_size.x * spp)

        # Try to avoid a division by an unknown constant if we can help it
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)

        # Compute the position on the image plane
        # (frequency index, time index)
        # time index is computed in sample() and not used here
        pos = mi.Vector2i(idx, 0 * idx)
        pos_f = mi.Vector2f(pos)

        # Re-scale the position to [0, 1]^2
        if not dr.allclose(film.crop_size(), film.size()):
            raise RuntimeError("Acoustic sampling does not support cropping")
        if not dr.allclose(film.crop_offset(), mi.Vector2i(0, 0)):
            raise RuntimeError("Acoustic sampling does not support cropping")

        scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
        offset = -mi.ScalarVector2f(film.crop_offset()) * scale
        pos_adjusted = dr.fma(pos_f, scale, offset)

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = 0.0 # unused

        frequency_sample = 0.
        if mi.has_flag(film.flags(), mi.FilmFlags.Spectral):
            raise NotImplementedError("Full spectral rendering not implemented.")
        else:
            pass # wavelength determined from the wavelength bin (stored in pos.x.).


        with dr.resume_grad():
            ray, weight = sensor.sample_ray_differential(
                time=time,
                sample1=frequency_sample, #unused for now,
                sample2=pos_adjusted, # pos.x() stores normalized frequency index in [0,1], pos.y() is not used
                sample3=aperture_sample
            )

        return ray, weight, pos_adjusted


    def prepare(self,
                sensor: mi.Sensor,
                seed: int = 0,
                spp: int = 0,
                aovs: list = []):

        film = sensor.film()
        sampler = sensor.sampler().clone()

        if spp != 0:
            sampler.set_sample_count(spp)

        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)

        film_size = film.crop_size()

        if film.sample_border():
            raise RuntimeError("Acoustic sampling does not support border sampling")
            film_size += 2 * film.rfilter().border_size()

        wavefront_size = film_size.x * spp

        if wavefront_size >= 2**32:
            raise RuntimeError(
                "The total number of Monte Carlo samples required by this "
                "rendering task (%i) exceeds 2^32-1 = 4294967295. Please use "
                "fewer samples per pixel or render using multiple passes."
                % wavefront_size)

        sampler.seed(seed, wavefront_size)
        film.prepare(aovs)

        return sampler, spp

    @dr.syntax
    def sample(self,
               scene: mi.Scene,
               sampler: mi.Sampler,
               sensor: mi.Sensor,
               ray: mi.Ray3f,
               block: mi.ImageBlock,
               position_sample: mi.Point2f, # in [0,1]^2
               active: mi.Bool,
               **_ # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum, mi.Bool]:
        mi.Log(mi.LogLevel.Debug, "running sample().")

        film = sensor.film()
        n_frequencies = mi.ScalarVector2f(film.crop_size()).x
        n_channels = film.base_channels_count()

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # dr.replace_grad breaks code in non ad variants.
        ad_variant = dr.replace_grad(mi.Float(1.0), mi.Float(2.0)).shape != (0,)
        mi.Log(mi.LogLevel.Debug, f"ad_variant: {ad_variant}")

        # --------------------- Configure loop state ----------------------

        distance     = mi.Float(0.0)
        max_distance = self.max_time * self.speed_of_sound

        # Air attenuation (ISO 9613-1) decay coefficient for this path's
        # frequency. Constant along the whole path (only distance varies),
        # so it is computed once upfront; 0 when attenuation is disabled,
        # which makes the exp(-distance * decay) factor a no-op (== 1).
        attenuation_decay = mi.Float(0.0)
        if self.apply_attenuation:
            attenuation_decay = mi.acoustic.energy_attenuation_coefficient(
                temperature=self.medium_temperature, frequency=ray.wavelengths[0],
                relative_humidity=self.medium_relative_humidity,
                atmospheric_pressure=self.medium_atmospheric_pressure)

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(ray)
        depth = mi.UInt32(0)  # Depth of current vertex
        β = mi.Spectrum(1)                                 # Path throughput weight
        η = mi.Float(1)                                    # Index of refraction
        active = mi.Bool(active)                           # Active SIMD lanes

        # Variables caching information from the previous bounce
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(0.) if self.hide_emitters else mi.Float(1.)
        prev_bsdf_delta = mi.Bool(True)

        while dr.hint(active,
                      max_iterations=self.max_depth,
                      label="Acoustic AD Path Tracing"):
            active_next = mi.Bool(active)

            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.
            si = scene.ray_intersect(ray,
                                        ray_flags=mi.RayFlags.All,
                                        coherent=(depth == 0))

            # Calculate path segment length. si.t includes a spawn-ray epsilon,
            # which moves the origin towards the intersection point and reduces
            # si.t slightly. Use true geometric distance instead:
            τ = dr.select(depth == 0, dr.norm(si.p - ray.o), dr.norm(si.p - prev_si.p))

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------

            # Hide the direct sound if necessary
            if self.hide_emitters:
                active_next &= ~((depth == 0) & ~si.is_valid())

            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            si_pdf = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            if self.is_detached:
                dr.disable_grad(si_pdf)

            mis = mis_weight(
                prev_bsdf_pdf,
                si_pdf
            )

            # Intensity of current emitter weighted by importance (def. by prev bsdf hits)
            Le = β * mis * ds.emitter.eval(si, active_next)

            # Store (direct) intensity to the image block
            T      = distance + τ
            Le     *= dr.exp(-T * attenuation_decay)
            active_next &= si.is_valid()
            Le_pos = mi.Point2f(position_sample.x * n_frequencies, # rescale from [0, 1] to [0, n_frequencies]
                                block.size().y * T / max_distance)
            block.put(pos=Le_pos,
                      values=film.prepare_sample(Le[0], si.wavelengths, n_channels),
                      active=active_next & (Le[0] > 0.))

            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next &= (depth + 1 < self.max_depth)

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            # If so, randomly sample an emitter without derivative tracking.
            with dr.suspend_grad(when=not self.is_detached):
                ds, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)

            # Retrace the ray towards the emitter because ds is directly sampled
            # from the emitter shape instead of tracing a ray against it.
            # This contradicts the definition of "sampling of *directions*"
            # Only retrace when gradients are needed (AD mode); in primal mode
            # skip the retrace to match the C++ acoustic_path integrator exactly.
            # si.spawn ray offsets the origin by a small epsilon to avoid
            # self-intersection. This introduced minor timing differences.
            if dr.grad_enabled(si.p):
                si_em       = scene.ray_intersect(si.spawn_ray(ds.d), active=active_em)
                ds_attached = mi.DirectionSample3f(scene, si_em, ref=si)
                ds_attached.pdf, ds_attached.delta, ds_attached.uv, ds_attached.n = (ds.pdf, ds.delta, si_em.uv, si_em.n)
                ds = ds_attached

            if self.is_detached:
                # The sampled emitter direction and the pdf must be detached
                # Recompute `em_weight = em_val / ds.pdf` with only `em_val` attached
                dr.disable_grad(ds.d, ds.pdf)
                em_val    = scene.eval_emitter_direction(si, ds, active_em)

                if dr.hint(ad_variant, mode='scalar'):
                    em_weight = dr.replace_grad(em_weight, dr.select((ds.pdf != 0), em_val / ds.pdf, 0))

            active_em &= (ds.pdf != 0.0)

            # Evaluate BSDF * cos(theta) differentiably (and detach the bsdf pdf)
            wo = si.to_local(ds.d)
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
            if self.is_detached:
                dr.disable_grad(bsdf_pdf_em)
            mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))

            Lr_dir = β * mis_em * bsdf_value_em * em_weight

            # Store (emission sample) intensity to the image block
            # Use true geometric distance instead of ds.dist (which is
            # shortened by the spawn-ray epsilon offset)
            τ_dir      = dr.norm(ds.p - si.p)
            T_dir      = T + τ_dir
            Lr_dir     *= dr.exp(-T_dir * attenuation_decay)
            Lr_dir_pos = mi.Point2f(position_sample.x * n_frequencies, # rescale from [0, 1] to [0, n_frequencies]
                                    block.size().y * T_dir / max_distance)
            block.put(pos=Lr_dir_pos,
                      values=film.prepare_sample(Lr_dir[0], si.wavelengths, n_channels),
                      active=active_em & (Lr_dir[0] > 0.))

            # ------------------ Detached BSDF sampling -------------------

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                sampler.next_1d(),
                                                sampler.next_2d(),
                                                active_next)

            if self.is_detached:
                # The sampled bsdf direction and the pdf must be detached
                # Recompute `bsdf_weight = bsdf_val / bsdf_sample.pdf` with only `bsdf_val` attached
                dr.disable_grad(bsdf_sample.wo, bsdf_sample.pdf)
                bsdf_val    = bsdf.eval(bsdf_ctx, si, bsdf_sample.wo, active_next)

                if dr.hint(ad_variant, mode='scalar'):
                    bsdf_weight = dr.replace_grad(bsdf_weight, dr.select(
                    (bsdf_sample.pdf != 0), bsdf_val / bsdf_sample.pdf, 0))

            # ---- Update loop variables based on current interaction -----

            wo_world = si.to_world(bsdf_sample.wo)
            if self.is_detached:
                # The direction in *world space* is detached
                wo_world = dr.detach(wo_world)

            ray = si.spawn_ray(wo_world)
            η *= bsdf_sample.eta
            β *= bsdf_weight

            # Information about the current vertex needed by the next iteration

            prev_si = si
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
            distance = T

            # -------------------- Stopping criterion ---------------------

            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β)
            active_next &= (β_max != 0)
            active_next &= β_max >= self.throughput_threshold
            active_next &= distance <= max_distance

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(β_max * η**2, .95)

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue


            depth[si.is_valid()] += 1
            active = active_next

        return (
            block,
            (depth != 0),  # Ray validity flag for alpha blending
        )

    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:
        mi.Log(mi.LogLevel.Info, "Rendering in forward mode")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aov_names()
            )

            # Generate a set of rays starting at the sensor
            ray, weight, position_sample = self.sample_rays(scene, sensor, sampler)

            # Prepare an ImageBlock as specified by the film
            block = film.create_block()

            with dr.resume_grad():
                # Launch the Monte Carlo sampling process in primal mode (1)
                # Contrary to the light case we already need the input gradient as we return δH dot L to avoid storing L which is a function.
                self.sample(
                    scene=scene,
                    sampler=sampler.clone(),
                    sensor=sensor,
                    ray=ray,
                    block=block,
                    position_sample=position_sample,
                    active=mi.Bool(True)
                )

                film.put_block(block)
                result_img = film.develop()

                # Propagate the gradients to the image tensor
                dr.forward_to(result_img)

        return dr.grad(result_img)


    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:
        mi.Log(mi.LogLevel.Info, "Rendering in backward mode")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aov_names()
            )

            # Generate a set of rays starting at the sensor
            ray, weight, position_sample = self.sample_rays(scene, sensor, sampler)

            # Prepare an ImageBlock as specified by the film
            block = film.create_block()

            with dr.resume_grad():
                # Launch the Monte Carlo sampling process in primal mode (1)
                # Contrary to the light case we already need the input gradient as we return δH dot L to avoid storing L which is a function.
                self.sample(
                    scene=scene,
                    sampler=sampler.clone(),
                    sensor=sensor,
                    ray=ray,
                    block=block,
                    position_sample=position_sample,
                    active=mi.Bool(True)
                )

                film.put_block(block)
                result_img = film.develop()

                # Differentiate sample splatting and weight division steps to
                # retrieve the adjoint radiance
                dr.set_grad(result_img, grad_in)
                dr.enqueue(dr.ADMode.Backward, result_img)
                dr.traverse(dr.ADMode.Backward)

                # We don't need any of the outputs here
                del ray, weight, block, sampler, position_sample

                gc.collect()

                # Run kernel representing side effects of the above
                dr.eval()


mi.register_integrator("acoustic_ad", lambda props: AcousticADIntegrator(props))
