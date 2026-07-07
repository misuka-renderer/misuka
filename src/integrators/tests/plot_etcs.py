"""Plot acoustic energy-time curves (ETCs).

Plots every ``.exr`` ETC found next to this script -- the acoustic integrator
tests (``test_acoustic_path.py``, ``test_acoustic_ad_integrators.py``) dump the
rendered ETC and a copy of the reference ETC here whenever a primal test fails
-- together with the committed reference ETCs in
``resources/data_acoustic/tests/integrators``.

Run directly::

    python plot_etcs.py
"""

from pathlib import Path

import mitsuba as mi
mi.set_variant('cuda_acoustic', 'llvm_acoustic', 'scalar_acoustic')

import numpy as np
import pyfar as pf
import matplotlib.pyplot as plt

from mitsuba.scalar_acoustic.test.util import find_resource

# Sampling rate must match the config's sampling_rate used to render the ETCs.
SAMPLING_RATE = 1000.0

# Directory next to this script (where failing tests dump ETCs) and the
# committed reference ETCs in the data_acoustic submodule.
tests_dir = Path(__file__).resolve().parent
data_dir = Path(find_resource('resources/data_acoustic/tests/integrators'))


def load_etc(path):
    """Load an ETC .exr as a pyfar Signal.

    The tape ETC has frequency bins along the first axis and time bins along the
    second. Squeeze a possible trailing channel axis, then transpose so each
    frequency band becomes a pyfar channel (time is the last axis).
    """
    etc = np.squeeze(np.array(mi.TensorXf(mi.Bitmap(str(path)))))
    return pf.Signal(etc.T, SAMPLING_RATE)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # (directory, label prefix) pairs.
    sources = [
        (tests_dir, ''),
        (data_dir, 'data_acoustic: '),
    ]

    n_plotted = 0
    for directory, label_prefix in sources:
        for exr in sorted(directory.glob('*.exr')):
            sig = load_etc(exr)
            pf.plot.time(sig, dB=True, ax=ax, label=f'{label_prefix}{exr.stem}')
            n_plotted += 1

    if n_plotted == 0:
        print(f'No .exr files found in {tests_dir} or {data_dir}')
        return

    ax.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
