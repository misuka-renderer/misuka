Release notes
=============

[unreleased]
------------

- Add Acoustic Path Integrator `[PR #3] <https://github.com/misuka-renderer/misuka/pull/3>`_
- Add Acoustic BSDF `[f5c57d4] <https://github.com/misuka-renderer/misuka/commit/f5c57d495597ad400a283f3620481aef802b5c7f>`_
- `acoustic_ad` and `acoustic_prb`: 
    - Use infinite depth by default `[f33301e] <https://github.com/misuka-renderer/misuka/commit/f33301ec13115c9132df09a6828a8f9b6fcbaa71>`_
    - Use true geometric distances for path length calculation, avoiding the epsilon offset introduced by `si.spawn_ray()` `[757f880] <https://github.com/misuka-renderer/misuka/commit/757f8807a9444a853d1886f9442dc79da6e50c9f>`_

misuka 0.0.0
------------

*August 24, 2025*

- Initial source code release, branched off Mitsuba 3.6.4. `[f4bb42e] <https://github.com/misuka-renderer/misuka/commit/f4bb42e43ab68df3b0c8a5ecd4df04106e2bb582>`_
