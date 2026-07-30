.. only:: not latex

    .. image:: images/misuka_logo.png
        :width: 60%
        :align: center

Getting started
===============

misuka is a research-oriented, differentiable **room-acoustic renderer** for
forward and inverse sound-transport simulation. It is a fully compatible
extension to `Mitsuba 3 <https://mitsuba.readthedocs.io/en/v3.9.0/>`_: it reuses
Mitsuba's scene format, geometry, samplers, and the `Dr.Jit
<https://drjit.readthedocs.io/en/v1.4.0/>`_ JIT compiler / autodiff engine, and
adds acoustic plugins (an absorbing/scattering material, several acoustic path tracers,
a microphone sensor, and an Energy-Time-Curve film). It implements `Time-Resolved
Path Replay Backpropagation <https://dl.acm.org/doi/pdf/10.1145/3730900>`_ for
efficient gradient estimation with respect to material properties, source/receiver
positions, and scene geometry.

Because misuka is an extension, the light-transport engine, scene description
language, and Python API are documented upstream. This site documents only
**acoustic rendering functionality**. Follow the links above for everything
misuka inherits from Mitsuba 3 and Dr.Jit.

Installation
------------

misuka can be installed via :monosp:`pip` from `PyPI
<https://pypi.org/project/misuka/>`_. This is the recommended method of installation.

.. code-block:: bash

    pip install misuka

This command will also install :monosp:`Dr.Jit` on your system if not already available.

See the :ref:`developer guide <sec-compiling>` for complete instructions on building
from the git source tree.

Requirements
^^^^^^^^^^^^

- ``Python >= 3.9``
- (optional) For computation on the GPU: ``Nvidia driver >= 535``
- (optional) For vectorized / parallel computation on the CPU: ``LLVM >= 11.1``
- (optional) For computation on Apple Silicon GPUs: macOS with a Metal-capable GPU

Hello World!
------------

The example below builds a simple shoebox room with a spherical sound source,
places a microphone, and renders an **Energy Time Curve (ETC)**. The ETC represents
the temporal energy distribution in the squared impulse response.

.. code-block:: python

    import misuka as mi

    mi.set_variant('cuda_ad_acoustic', 'metal_ad_acoustic', 'llvm_ad_acoustic')
    print('Using variant:', mi.variant())

    from misuka import ScalarTransform4f as tf

    # A 6 x 8 x 4 m shoebox room with a spherical sound source.
    room_dim     = [6.0, 8.0, 4.0]
    source_pos   = [3.0, 6.0, 1.2]
    receiver_pos = [2.0, 1.0, 1.2]

    scene_dict = {
        'type': 'scene',
        # Omnidirectional sound source.
        'emitter': {
            'type': 'sphere',
            'radius': 1,
            'center': source_pos,
            'emitter': {'type': 'area', 'radiance': {'type': 'uniform', 'value': 50}},
        },

        'room': {
            'type': 'cube',
            'to_world': tf().scale(room_dim),
            'flip_normals': True,
            'bsdf': {
                'type': 'acousticbsdf',
                'absorption': {'type': 'spectrum', 'value': [(100, 0.1), (500, 0.2), (20000, 0.3)]},
                'scattering': {'type': 'spectrum', 'value': [(100, 0.2), (500, 0.5), (20000, 0.8)]},
            },
        },
    }

    scene = mi.load_dict(scene_dict)

    # A microphone that records an ETC into a `tape` film.
    max_time      = 2
    sampling_rate = 2000
    n_time_bins   = int(max_time * sampling_rate)
    frequencies   = [100, 500, 20000]

    microphone = mi.load_dict({
        'type': 'microphone',
        'origin': receiver_pos,
        'direction': source_pos,
        'film': {
            'type': 'tape',
            'frequencies': ','.join(map(str, frequencies)),
            'time_bins': n_time_bins,
        },
    })

    integrator = mi.load_dict({
        'type': 'acoustic_path',
        'max_time': max_time,
    })

    # Render the ETC. Increase spp to reduce variance.
    spp = 2**20 # around 1 million rays, power of 2 for better performance.
    etc = mi.render(scene, sensor=microphone, integrator=integrator, spp=spp)

For a fully working version, including separate materials for each wall, a visual
preview of the room and a plot of the ETC, see the :doc:`rendering tutorials <src/rendering_tutorials>`.

License
-------

misuka is licensed under the `PolyForm Noncommercial License 1.0.0
<https://polyformproject.org/licenses/noncommercial/1.0.0>`_, which permits academic
and private use. Files inherited from Mitsuba 3 remain under the original BSD-3-Clause
license. See `LICENSE
<https://github.com/misuka-renderer/misuka/blob/master/LICENSE>`_.
If you are interested in using misuka commercially, please contact
a.jueterbock@tu-berlin.de.

Citation
--------

When using misuka in academic projects, please cite:

.. code-block:: bibtex

    @article{misuka,
        title   = {{misuka}: An Open-Source Differentiable Room Acoustic Renderer},
        author  = {J\"uterbock, Tobias and Finnendahl, Ugo and Worchel, Markus and
                   Wujecki, Daniel and Alexa, Marc and Weinzierl, Stefan},
        journal = {Proceedings of Meetings on Acoustics},
        volume  = {58},
        number  = {1},
        pages   = {022004:1--022004:13},
        year    = {2026},
        doi     = {10.1121/2.0002193},
    }

When using Time-Resolved Path Replay Backpropagation, please also cite:

.. code-block:: bibtex

    @article{acoustic_prb,
        title   = {Differentiable Geometric Acoustic Path Tracing Using
                   Time-Resolved Path Replay Backpropagation},
        author  = {Finnendahl, Ugo and Worchel, Markus and J\"uterbock, Tobias and
                   Wujecki, Daniel and Brinkmann, Fabian and Weinzierl, Stefan and
                   Alexa, Marc},
        journal = {ACM Transactions on Graphics},
        volume  = {44},
        number  = {4},
        pages   = {82:1--82:17},
        year    = {2025},
        doi     = {10.1145/3730900},
    }

misuka is built on `Mitsuba 3 <https://mitsuba.readthedocs.io/en/v3.9.0/>`_. When
appropriate, please also cite the underlying renderer following its
`citation guidelines <https://mitsuba.readthedocs.io/en/v3.9.0/#citation>`_.

.. .............................................................................

.. toctree::
   :hidden:

   self

.. toctree::
    :maxdepth: 1
    :caption: Tutorials
    :hidden:

    src/rendering_tutorials
    src/inverse_rendering_tutorials

.. toctree::
    :maxdepth: 1
    :caption: Guides
    :hidden:

    src/key_topics
    src/developer_guide

.. toctree::
    :maxdepth: 1
    :caption: References
    :hidden:

    src/plugin_reference
    src/api_reference

.. toctree::
    :maxdepth: 1
    :caption: Miscellaneous
    :hidden:

    src/optix_setup
    release_notes
    zz_bibliography
