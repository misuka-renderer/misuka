.. _sec-emitters:

Emitters
========

*Emitters* are the sound sources of a scene. misuka reuses Mitsuba's emitter
plugins unchanged, so an emitter's spectrum is interpreted in the frequency
domain like every other spectrum in an acoustic variant (see
:ref:`Frequency-domain spectra <sec-spectra-acoustic>`).

**In practice, use a spherical emitter.** Attach an :ref:`area <emitter-area>`
emitter to a :ref:`sphere <shape-sphere>` shape and give the sphere a radius that
is small compared to the room, but not so small that rays rarely find it. This is
what the tutorials and the getting-started example do.

.. tabs::
    .. code-tab:: xml

        <scene version="3.0.0">
            <!-- .. scene contents .. -->

            <shape type="sphere">
                <float name="radius" value="0.2"/>
                <point name="center" x="3" y="6" z="1.2"/>

                <emitter type="area">
                    <spectrum name="radiance" value="100:1, 20000:1"/>
                </emitter>
            </shape>
        </scene>

    .. code-tab:: python

        'type': 'scene',

        # .. scene contents ..

        'source': {
            'type': 'sphere',
            'radius': 0.2,
            'center': [3, 6, 1.2],
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'spectrum', 'value': [(100, 1), (20000, 1)]},
            },
        },

The :ref:`point <emitter-point>` emitter currently loses the direct sound. A
contribution that reaches the receiver without touching a surface can only be
recorded when the traced ray hits the emitter itself, and a point emitter has no
geometry to hit. Reflected contributions still arrive, because emitter sampling
runs at every surface interaction, so a point source yields an ETC whose first
arrival is a first-order reflection instead of the direct path. In light
transport this is deliberate. A delta emitter has zero area and is meant to be
reached exclusively through emitter sampling, which is also where its entire
contribution belongs. An acoustic point emitter is planned.

Emitters that model a distant environment, such as ``constant``, ``envmap`` and
``directional``, have no acoustic counterpart and are documented upstream in the
`Mitsuba emitter reference
<https://mitsuba.readthedocs.io/en/v3.9.0/src/generated/plugins_emitters.html>`_.
