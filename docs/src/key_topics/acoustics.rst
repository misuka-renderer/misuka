.. _sec-acoustics:

Acoustic rendering
==================

misuka turns `Mitsuba 3 <https://mitsuba-renderer.org/>`_ into a room-acoustic renderer. The geometry, samplers,
scene format, and the `Dr.Jit <https://drjit.readthedocs.io/en/v1.4.0/>`_
JIT / autodiff engine all carry over unchanged. What changes is *how* transport
is evaluated and *what* the renderer records. Instead of solving light energy transport for a steady
state and integrating it into a 2D image, misuka resolves sound energy transport in time and
records an **energy-time curve (ETC)**: the energy arriving at a receiver as a
function of propagation time, resolved per frequency band.

This page explains the concepts a user needs in order to work with misuka, and the differences to Mitsuba 3.
The plugins themselves are documented in the
:ref:`plugin reference <sec-integrators>`, and examples live in the
:doc:`rendering <../rendering_tutorials>` and
:doc:`inverse rendering <../inverse_rendering_tutorials>` tutorials.

The ``acoustic`` variant family
--------------------------------

Mitsuba 3 is provides a set of different system *variants*, each a ``(backend × spectrum)`` combination.
The ``backend`` part determines *where and how* the computation runs, while the ``spectrum`` part defines what a "spectrum" represents.
For example, ``cuda_rgb`` performs RGB rendering on NVIDIA GPUs via CUDA, while ``metal_spectral`` renders on Apple Silicon and produces multichannel images that represent continuous wavelength bands, and ``llvm_mono`` renders monochromatic (greyscale) images on the CPU.

misuka adds an ``_acoustic`` variant family.
These variants work in the **frequency (Hz) domain** rather than with wavelengths (nm).
Therefore, acoustic scenes **must** be rendered with an acoustic variant and are not compatible with ``rgb``, ``mono`` or ``spectral`` variants.
See :ref:`Frequency-domain spectra <sec-spectra-acoustic>`.

The opposite also holds:
Rendering an image with an optical integrator using an acoustic variant will produce unexpected results, unless handled explicitly.
For example, using an acoustic spectrum with an optical variant will interpret frequency nodes in Hz as wavelengths in nm.
Conversely, using an acoustic variant to render an ``rgb`` image will produce very noisy images because Mitsuba's rgb-specific variance reduction techniques are not natively compatible with the acoustic spectrum type.
See the :doc:`Forward Rendering: Shoebox Room <../rendering_tutorials>` tutorial for an example how to render images of acoustic scenes, evaluated at specific frequencies.

The ``pip`` package ships with ``acoustic_ad`` variants for all backends (``cuda``, ``metal`` and ``llvm``).
We recommend setting the variants in the following way.
When selecting multiple variants, the first available one will be used.

.. code-block:: python

    import misuka as mi

    mi.set_variant('cuda_ad_acoustic', 'metal_ad_acoustic', 'llvm_ad_acoustic')

    print(f'Using variant: {mi.variant()}')

Mitsuba's backend prefixes apply (``scalar_``, ``llvm_``, , ``cuda_``, ``metal_`` and their ``_ad_`` autodiff forms), so ``llvm_ad_acoustic`` is a vectorized CPU variant with automatic differentiation, ``cuda_ad_acoustic`` and ``metal_ad_acoustic`` are its GPU counterparts on NVIDIA GPUs and Apple Silicon, respectively.
See the `Mitsuba variants guide <https://mitsuba.readthedocs.io/en/v3.9.0/src/key_topics/variants.html>`_ for the underlying variant system, and the :ref:`developer guide <sec-compiling>` for enabling additional variants if you compile misuka yourself.


Time-resolved transport
-----------------------

Acoustic simulation in misuka is a geometric (ray-based) simulation. A path from
a sound source to the receiver is attenuated at each surface interaction by the
frequency-dependent absorption and scattering of the
:ref:`acoustic material <bsdf-acousticbsdf>`. Three points characterize how this
relates to light transport:

- **The same transport equation, with a propagation delay.** misuka solves the
  *room acoustic rendering equation* :cite:`Siltanen2007`, which is the
  rendering equation with the incident term evaluated at a retarded time, i.e.
  the current time offset by the propagation time:
  
  .. math::
      L(\mathbf{x} \to \omega, t) = L_\mathrm{e}(\mathbf{x} \to \omega, t) +
      \int_{\mathcal{H}^2} f_\mathrm{r}(\mathbf{x}, \omega' \to \omega)\,
      L\!\left(\mathbf{y} \to \omega',\, t - \frac{\|\mathbf{x} - \mathbf{y}\|}{c}\right)
      \cos\theta'\, \mathrm{d}\omega'

  Letting the speed of sound :math:`c \to \infty` removes the delay and recovers
  the light-transport case exactly. What travels along a ray is therefore a
  radiance-like quantity, invariant along the ray, just as in Mitsuba; the
  :math:`1/r^2` falloff enters only through the integration over all rays: an 
  emitter that is further away occupies a smaller solid angle and is therefore hit
  by fewer rays.
- **Time is resolved rather than integrated out.** Light transport is solved for
  the steady state. misuka instead tracks the total geometric path length,
  converts it into a propagation *time*, and bins the contribution accordingly.
  The ETC is thus the energy impulse response of the transport operator, and
  integrating it over all time recovers the steady-state result a conventional
  renderer would produce.
- **Energetic, not wave-based.** Contributions are summed as intensities, with
  no pressure phase. Optical rendering makes the same approximation, but it is a
  stronger one in acoustics: audible wavelengths are not always small compared 
  to room geometry, so diffraction and interference effects can become significant
  at lower frequencies / larger wavelengths. These effects are not modelled. 

Ray-geometry intersection, importance sampling, the Dr.Jit computation graph,
and gradient propagation are all inherited from Mitsuba and Dr.Jit and behave as
usual.

The Energy Time Curve (ETC)
---------------------------

The ETC is the primary output of all acoustic ray tracing methods.
It can be interpreted as the envelope of the squared impulse response.

It is produced by the :ref:`tape <film-tape>` film paired with a
:ref:`microphone <sensor-microphone>` sensor, and records how much energy
reaches the receiver in each **time bin**, separately for each **frequency**.

- The **rendered frequencies** are listed explicitly on the ``tape`` film via its
  ``frequencies`` parameter (e.g. octave-band center frequencies). Each entry is
  one band of the output.
- The **time bins** (``time_bins``) discretize propagation time. Together with
  the integrator's ``max_time`` they set the temporal resolution: each bin
  covers ``max_time / time_bins`` seconds, and a path contributes to the bin
  matching its total propagation time.

A render therefore returns a tensor of shape ``(time_bins, frequencies, 1)``,
with energy against time along the first axis and one column per frequency band.
This replaces the ``(height, width, channels)`` image a conventional film would
produce. The microphone has no image resolution, only a single receiver point.

.. code-block:: python

    max_time      = 0.1     # seconds
    sampling_rate = 10000   # time bins per second

    microphone = mi.load_dict({
        "type": "microphone",
        "origin": [2.0, 1.0, 1.2],
        "direction": [3.0, 6.0, 1.2],
        "film": {
            "type": "tape",
            "frequencies": "125, 250, 500, 1000, 2000, 4000",
            "time_bins": int(max_time * sampling_rate),
        },
    })

    integrator = mi.load_dict({
        "type": "acoustic_path",
        "max_time": max_time,
        "speed_of_sound": 343,
        "max_depth": -1,        # unlimited reflections
    })

    etc = mi.render(scene, sensor=microphone, integrator=integrator, spp=2**16)

What carries over, and what is replaced
---------------------------------------

If you already know Mitsuba 3, the mental model is: **keep the scene, swap the
acoustic components.**

.. list-table::
    :header-rows: 1
    :widths: 30 35 35

    * - Role
      - Mitsuba (optical)
      - misuka (acoustic)
    * - Spectrum
      - RGB / wavelength spectra in nanometers
      - frequency spectra in Hz
    * - Material
      - optical BSDFs (``diffuse``, ``conductor``, ...)
      - :ref:`acousticbsdf <bsdf-acousticbsdf>` (absorption + scattering)
    * - Sensor
      - camera (``perspective``, ...)
      - :ref:`microphone <sensor-microphone>` (receiver point)
    * - Film
      - image film (``hdrfilm``, ...)
      - :ref:`tape <film-tape>` (ETC storage)
    * - Integrator
      - path / PRB light-transport integrators
      - :ref:`acoustic_path <integrator-acoustic_path>` (forward) and the
        differentiable :ref:`acoustic_prb <integrator-acoustic_prb>` integrators

Geometry, transforms, samplers, reconstruction filters, the XML/dict scene
format, and the whole Dr.Jit / autodiff functionality are unchanged and remain
documented at `mitsuba.readthedocs.io <https://mitsuba.readthedocs.io/en/v3.9.0>`_ and `drjit.readthedocs.io <https://drjit.readthedocs.io/en/v1.4.0/>`_.

Differentiable acoustics
------------------------

Under an ``_ad_`` variant, misuka differentiates the ETC with respect to scene
parameters: material absorption/scattering, source and receiver positions, and
geometry.

The :ref:`acoustic_prb <integrator-acoustic_prb>` integrator
implements **Time-Resolved Path Replay Backpropagation** :cite:`acoustic_prb`,
which propagates gradients efficiently across many reflections without
storing the full path history.
It is the basis for all optimization tasks and is suited for material optimization
with static (not optimized) geometry.
The :ref:`acoustic_prb_threepoint <integrator-acoustic_prb_threepoint>` produces
unbiased gradients also when optimizing geometry. See :ref:`plugin reference <sec-integrators>`
for further details.