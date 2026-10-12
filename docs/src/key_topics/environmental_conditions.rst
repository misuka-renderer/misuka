.. _key_topics-environmental_conditions:

Environmental conditions in acoustic rendering
================================================


.. note::

    **For example usage see the tutorials** :doc:`../tutorials_acoustic/rendering/atmospheric_rendering` **and** 
    :doc:`../tutorials_acoustic/inverse_rendering/atmosphere_optimization`
    
The acoustic integrators (:ref:`acoustic_path <integrator-acoustic_path>`, :ref:`acoustic_ad <integrator-acoustic_ad>`, :ref:`acoustic_prb <integrator-acoustic_prb>` and their three-point variants) simulate sound propagation through air, whose state -- temperature, humidity, pressure, CO2 concentration -- affects two things:

* the **speed of sound**, which determines how a path's traveled distance maps to a time bin, and
* the **frequency-dependent air attenuation** (ISO 9613-1), which determines how much energy a path loses over that distance.

Every integrator exposes both through a single ``acoustic_medium`` plugin parameter (a dictionary of atmospheric fields plus ``speed_of_sound_method`` and ``apply_attenuation``); see its own :ref:`plugin documentation <integrator-acoustic_path>` for the parameter reference. This page documents the underlying physics and formulas in full detail.

.. _key_topics-environmental_conditions-speed_of_sound:

Speed of sound
--------------

Three calculation methods are available, selected via ``speed_of_sound_method``. All three are differentiable: under an ``*_ad_*`` variant, gradients set on ``temperature``, ``relative_humidity``, ``atmospheric_pressure``, ``saturation_vapor_pressure`` or ``co2_ppm`` propagate through to the returned speed of sound. Note that in the PRB integrators (``acoustic_prb``, ``acoustic_prb_threepoint``), these gradients only flow through air attenuation: the effect of the speed of sound on time-bin placement is not differentiated.

``"simple"`` (default)
^^^^^^^^^^^^^^^^^^^^^^^

Following ISO 9613-1 (Formula A.5):

.. math::
    :name: eq_speed_simple

    c = 343.2 \cdot \sqrt{(T + 273.15) / 293.15}

Only uses temperature :math:`T` (in °C), which must be in the range of -20°C to 50°C. This is the default method since temperature is, in practice, measured far more often than humidity or atmospheric pressure -- a default of one of the other two methods would still fall back to inaccurate standard-medium values for whichever quantities were not actually measured, without any accuracy benefit over ``"simple"``.

``"ideal_gas"``
^^^^^^^^^^^^^^^^

Speed of sound of a humid-air mixture treated as an ideal gas, based on chapter 6.3 in V. E. Ostashev and D. K. Wilson, *Acoustics in Moving Inhomogeneous Media*, 2nd ed. London: CRC Press, 2015. doi: 10.1201/b18922:

.. math::
    :name: eq_speed_ideal_gas

    c = \sqrt{\gamma_a R_a T_K \left(1 + (\alpha (1 + \delta - \nu) - 1) C\right)}

where :math:`T_K` is temperature in Kelvin, :math:`R_a` the specific gas constant of dry air, :math:`\gamma_a, \gamma_w` the heat capacity ratios of dry air and water vapor, :math:`\alpha` the ratio of their molar masses, and :math:`C` the water vapor mole fraction term derived from relative humidity, atmospheric pressure and saturation vapor pressure. A missing saturation vapor pressure is estimated from temperature via the Magnus formula (O. A. Alduchov and R. E. Eskridge, "Improved Magnus Form Approximation of Saturation Vapor Pressure," J. Appl. Meteor., 1996). Relative humidity must be in the range of 0 to 1, atmospheric pressure must be positive.

``"cramer"``
^^^^^^^^^^^^^

\ O. Cramer, "The variation of the specific heat ratio and the speed of sound in air with temperature, pressure, humidity, and CO2 concentration," The Journal of the Acoustical Society of America, vol. 93, no. 5, pp. 2510-2516, May 1993, doi: 10.1121/1.405827 -- an empirical quadratic fit, the sum of:

* a temperature-only term :math:`(a_0 + a_1 T + a_2 T^2)`
* a water-vapor term :math:`(a_3 + a_4 T + a_5 T^2) x_w`
* a pressure term :math:`(a_6 + a_7 T + a_8 T^2) p`
* a CO2 term :math:`(a_9 + a_{10} T + a_{11} T^2) x_c`
* squared terms :math:`a_{12} x_w^2 + a_{13} p^2 + a_{14} x_c^2`
* a cross term :math:`a_{15} x_c\, p\, x_w`

where :math:`x_w` is the water vapor mole fraction (derived from relative humidity and :math:`p`), :math:`p` is atmospheric pressure and :math:`x_c` is the CO2 mole fraction (derived from CO2 concentration in ppm); the 16 empirical coefficients :math:`a_0 \ldots a_{15}` are Cramer's published constants. Requires temperature in the range of 0°C to 30°C and atmospheric pressure in the range of 75,000 Pa to 102,000 Pa. A missing atmospheric pressure defaults to 101,325 Pa (standard atmosphere); a missing CO2 concentration defaults to 428.73 ppm, the global monthly mean for 2026-07 reported by NOAA GML (https://doi.org/10.15138/9N0H-ZH07, retrieved 2026-08-28).

.. _key_topics-environmental_conditions-attenuation:

Air attenuation
---------------

Sound loses energy as it travels through air, more so at higher frequencies. The integrators compute this loss following ISO 9613-1:1993: the pure-tone energy attenuation coefficient in dB/m is

.. math::
    :name: eq_attenuation

    \alpha = 8.686\, f^2 (\alpha_{cl} + \alpha_{vib})

consisting of a classical absorption term :math:`\alpha_{cl}` and a molecular relaxation term :math:`\alpha_{vib}`:

.. math::
    :name: eq_attenuation_terms

    \begin{align}
    \alpha_{cl}  &= 1.84 \cdot 10^{-11} (p_r / p_a) \sqrt{T / T_0} \\
    \alpha_{vib} &= (T / T_0)^{-5/2}(\alpha_O + \alpha_N)
    \end{align}

where :math:`\alpha_O` and :math:`\alpha_N` are the oxygen and nitrogen relaxation contributions, whose relaxation frequencies depend on atmospheric pressure, temperature and the water vapor concentration derived from relative humidity. Here :math:`f` is frequency, :math:`T` is temperature in Kelvin, :math:`T_0 = 293.15` K and :math:`p_r = 101325` Pa are the reference temperature and pressure, and :math:`p_a` is atmospheric pressure. The coefficient is converted from dB/m to the natural (1/m) energy decay coefficient used during rendering via :math:`\alpha_f = \alpha / (10 / \ln 10)`.

Validity ranges (per ISO 9613-1): temperature must be greater than -73°C (for an accuracy of +/-50%, +/-10% in the range of -20°C to 50°C), frequency must be greater than 50 Hz, atmospheric pressure must be less than 200 kPa, and the frequency-to-pressure ratio must be between :math:`4 \times 10^{-4}` Hz/Pa and 10 Hz/Pa.

During rendering, each path contribution of frequency :math:`f` is scaled by :math:`\exp(-d \, \alpha_f)`, where :math:`d` is the path's traveled distance -- applied per-vertex to the true accumulated geometric distance in the ``acoustic_path``, ``acoustic_ad`` and ``acoustic_prb`` integrators.

.. _key_topics-environmental_conditions-python_api:

Python API
----------

The underlying formulas are also available standalone, for use outside of a full render: :py:func:`mitsuba.acoustic.speed_of_sound` and :py:func:`mitsuba.acoustic.energy_attenuation_coefficient`, both differentiable in the same parameters as described above.
