"""
Overview
--------

This file defines a set of unit tests to assess the correctness of the
acoustic adjoint integrators  `acoustic_ad`, `acoustic_ad_threepoint`,
`acoustic_prb`, `acoustic_prb_threepoint`. All integrators will be tested for
their implementation of primal rendering, adjoint forward
rendering and adjoint backward rendering.

- For primal rendering, the output ETC will be compared to a ground truth
  ETC precomputed in the `resources/data_acoustic/tests/integrators` directory.
- Adjoint backward rendering will be compared against finite differences.
- Adjoint forward rendering will be compared against finite differences.

Those tests will be run on a set of configurations (scene description + metadata)
also provided in this file. More tests can easily be added by creating a new
configuration type and add it to the *_CONFIGS_LIST below.

By executing this script with python directly it is possible to regenerate the
reference data (e.g. for a new configurations). Please see the following command:

``python3 test_acoustic_ad_integrators.py --help``

"""

from mitsuba.scalar_acoustic import ScalarTransform4f as T
import drjit as dr
import mitsuba as mi

import pytest
import os
import argparse

from os.path import join, exists

from mitsuba.scalar_rgb.test.util import fresolver_append_path
from mitsuba.scalar_rgb.test.util import find_resource

output_dir = find_resource('resources/data_acoustic/tests/integrators')


# -------------------------------------------------------------------
#                          initialization tests
# -------------------------------------------------------------------

# List of integrators to test

INTEGRATORS = [
    'acoustic_ad',
    'acoustic_ad_threepoint',
    'acoustic_prb',
    'acoustic_prb_threepoint'
]

@pytest.mark.parametrize('integrator_name', INTEGRATORS)
def test01_initialization(variants_all_jit_acoustic, integrator_name):
    integrator = mi.load_dict({'type': integrator_name,
                               'max_time': 1.0,
                               'speed_of_sound': 343.0}, parallel=False)
    assert isinstance(integrator, mi.CppADIntegrator)

    with pytest.raises(ValueError):
        mi.load_dict({'type': integrator_name, 'max_time': -1})

@pytest.mark.parametrize('integrator_name', INTEGRATORS)
def test02_constructor_default_values(variants_all_jit_acoustic, integrator_name):
    """Test that default property values are set correctly."""
    integrator = mi.load_dict({'type': integrator_name, 'max_time': 1.0})

    assert integrator.speed_of_sound == 343.0
    assert integrator.max_time == 1.0
    assert integrator.is_detached
    assert not integrator.hide_emitters
    assert integrator.track_time_derivatives
    assert integrator.max_depth == 0xffffffff  # -1 maps to 2^32-1
    assert integrator.rr_depth == 100000
    assert dr.allclose(integrator.throughput_threshold, 10 ** (-60.0 / 10.0))

@pytest.mark.parametrize('integrator_name', INTEGRATORS)
def test03_constructor_custom_values(variants_all_jit_acoustic, integrator_name):
    """Test that custom property values are accepted and stored."""
    integrator = mi.load_dict({
        'type': integrator_name,
        'max_time': 5.0,
        'speed_of_sound': 100.0,
        'max_depth': 10,
        'rr_depth': 50,
        'is_detached': False,
        'hide_emitters': True,
        'track_time_derivatives': False,
        'max_energy_loss': 60.0,
    })

    assert integrator.max_time == 5.0
    assert integrator.speed_of_sound == 100.0
    assert integrator.max_depth == 10
    assert integrator.rr_depth == 50
    assert not integrator.is_detached
    assert integrator.hide_emitters
    assert not integrator.track_time_derivatives
    assert dr.allclose(integrator.throughput_threshold, 10 ** (-60.0 / 10.0))

@pytest.mark.parametrize('integrator_name', INTEGRATORS)
def test04_constructor_max_time_missing(variants_all_jit_acoustic, integrator_name):
    """max_time is required and must raise ValueError if missing."""
    with pytest.raises(ValueError):
        mi.load_dict({'type': integrator_name})

@pytest.mark.parametrize('integrator_name', INTEGRATORS)
def test05_constructor_max_time_zero(variants_all_jit_acoustic, integrator_name):
    """max_time=0 must raise ValueError."""
    with pytest.raises(ValueError):
        mi.load_dict({'type': integrator_name, 'max_time': 0.0})

@pytest.mark.parametrize('integrator_name', INTEGRATORS)
def test06_constructor_speed_of_sound_invalid(variants_all_jit_acoustic, integrator_name):
    """speed_of_sound <= 0 must raise ValueError."""
    with pytest.raises(ValueError):
        mi.load_dict({'type': integrator_name, 'max_time': 1.0, 'speed_of_sound': 0.0})
    with pytest.raises(ValueError):
            mi.load_dict({'type': integrator_name, 'max_time': 1.0, 'speed_of_sound': -1.0})

@pytest.mark.parametrize('integrator_name', INTEGRATORS)
def test07_constructor_max_depth_invalid(variants_all_jit_acoustic, integrator_name):
    """max_depth < -1 must raise an exception, but -1 and 0 are valid."""
    with pytest.raises(Exception):
        mi.load_dict({'type': integrator_name, 'max_time': 1.0, 'max_depth': -2})

    # max_depth=-1 (infinite) is valid
    integrator = mi.load_dict({'type': integrator_name, 'max_time': 1.0, 'max_depth': -1})
    assert integrator.max_depth == 0xffffffff

    # max_depth=0 is valid
    integrator = mi.load_dict({'type': integrator_name, 'max_time': 1.0, 'max_depth': 0})
    assert integrator.max_depth == 0

@pytest.mark.parametrize('integrator_name', INTEGRATORS)
def test08_constructor_rr_depth_invalid(variants_all_jit_acoustic, integrator_name):
    """rr_depth <= 0 must raise an exception."""
    with pytest.raises(Exception):
        mi.load_dict({'type': integrator_name, 'max_time': 1.0, 'rr_depth': 0})
    with pytest.raises(Exception):
            mi.load_dict({'type': integrator_name, 'max_time': 1.0, 'rr_depth': -1})

@pytest.mark.parametrize('integrator_name', INTEGRATORS)
def test09_constructor_max_energy_loss_invalid(variants_all_jit_acoustic, integrator_name):
    """max_energy_loss < 0 (and not -1) must raise ValueError."""
    with pytest.raises(ValueError):
        mi.load_dict({'type': integrator_name, 'max_time': 1.0, 'max_energy_loss': -2.0})
    with pytest.raises(ValueError):
        mi.load_dict({'type': integrator_name, 'max_time': 1.0, 'max_energy_loss': 0.0})

    # -1 (disabled) is valid
    integrator = mi.load_dict({'type': integrator_name, 'max_time': 1.0, 'max_energy_loss': -1.0})
    assert integrator.throughput_threshold == 0.0
    integrator = mi.load_dict({'type': integrator_name, 'max_time': 1.0, 'max_energy_loss': 123})
    assert integrator.throughput_threshold == 10 ** (-123 / 10.0)


# -------------------------------------------------------------------
#                          Test configs
# -------------------------------------------------------------------


class ConfigBase:
    """
    Base class to configure test scene and define the parameter to update
    """
    requires_discontinuities = False

    def __init__(self) -> None:
        self.spp = 2**20
        self.speed_of_sound = 340
        self.max_time = 0.2
        self.max_depth = -1
        self.rr_depth = 100000
        self.max_energy_loss = 20
        self.sampling_rate = 1000.0
        self.frequencies = '250, 500'
        self.error_mean_threshold = 0.05
        self.error_max_threshold = 0.5
        self.error_mean_threshold_bwd = 0.05
        self.ref_fd_epsilon = 1e-3
        self.emitter_radius = 0.5

        self.integrator_dict = {
            'max_depth': self.max_depth,
            'speed_of_sound': self.speed_of_sound,
            'max_depth': self.max_depth,
            'max_time': self.max_time,
            'max_energy_loss': self.max_energy_loss,
            'rr_depth': self.rr_depth,
        }

        self.sensor_dict = {
            'type': 'microphone',
            'origin': [-2, 0, 0],
            'direction': [1, 0, 0],
            'kappa': 0,
            'film': {
                'type': 'tape',
                'rfilter': {'type': 'gaussian', 'stddev': 0.25},
                'sample_border': False,
                'pixel_format': 'MultiChannel',
                'component_format': 'float32',
            }
        }

        # Set the config name based on the type
        import re
        self.name = re.sub(r'(?<!^)(?=[A-Z])', '_', self.__class__.__name__[:-6]).lower()

    def initialize(self):
        """
        Initialize the configuration, loading the Mitsuba scene and storing a
        copy of the scene parameters to compute gradients for.
        """

        self.sensor_dict['film']['frequencies'] = self.frequencies
        self.sensor_dict['film']['time_bins'] = int(self.sampling_rate * self.max_time)
        self.scene_dict['sensor'] = self.sensor_dict

        @fresolver_append_path
        def create_scene():
            return mi.load_dict(self.scene_dict)
        self.scene = create_scene()
        self.params = mi.traverse(self.scene)

        if hasattr(self, 'key'):
            self.params.keep([self.key])
            self.initial_state = type(self.params[self.key])(self.params[self.key])

    def update(self, theta):
        """
        This method update the scene parameter associated to this config
        """
        self.params[self.key] = self.initial_state + theta
        dr.set_label(self.params, 'params')
        self.params.update()
        dr.eval()

    def __repr__(self) -> str:
        return f'{self.name}[\n' \
            f'  integrator = {self.integrator_dict},\n' \
            f'  spp = {self.spp},\n' \
            f'  key = {self.key if hasattr(self, "key") else "None"}\n' \
            f']'


class SphericalEmitterRadianceConfig(ConfigBase):
    """
    Radiance of a spherical emitter
    """

    def __init__(self) -> None:
        super().__init__()
        self.key = 'spherical_emitter.emitter.radiance.value'
        self.scene_dict = {
            'type': 'scene',
            'spherical_emitter': {
                'type': 'sphere',
                'radius': self.emitter_radius,
                'center': [2, 0, 0],
                'emitter': {
                    'type': 'area',
                    'radiance': {
                        'type': 'uniform',
                        'value': 1,
                    },
                },
            },
        }

class ShoeboxAbsorptionConfig(ConfigBase):
    """
    Absorption coefficient of the shoebox room.
    """

    def __init__(self) -> None:
        super().__init__()
        self.key = 'shoebox.bsdf.absorption.values'
        self.scene_dict = {
            'type': 'scene',
            'shoebox': {
                'type': 'cube',
                'flip_normals': True,
                'to_world': T().scale([7, 5, 3]),
                'bsdf': {
                    'type': 'acousticbsdf',
                    'specular_lobe_width': 0.001,
                    'absorption': {
                        'type': 'spectrum',
                        'value': [(250, 0.4), (500, 0.6)],
                    },
                    'scattering': {
                        'type': 'spectrum',
                        'value': [(250, 0.1), (500, 0.9)],
                    },
                },
            },
            'spherical_emitter': {
                'type': 'sphere',
                'radius': self.emitter_radius,
                'center': [2, 0, 0],
                'emitter': {
                    'type': 'area',
                    'radiance': {
                        'type': 'uniform',
                        'value': 1,
                    },
                },
            },
        }


class ShoeboxScatteringConfig(ConfigBase):
    """
    Scattering coefficient of the shoebox room.
    """

    def __init__(self) -> None:
        super().__init__()
        self.key = 'shoebox.bsdf.scattering.values'
        self.scene_dict = {
            'type': 'scene',
            'shoebox': {
                'type': 'cube',
                'flip_normals': True,
                'to_world': T().scale([7, 5, 3]),
                'bsdf': {
                    'type': 'acousticbsdf',
                    'specular_lobe_width': 0.001,
                    'absorption': {
                        'type': 'spectrum',
                        'value': [(250, 0.4), (500, 0.6)],
                    },
                    'scattering': {
                        'type': 'spectrum',
                        'value': [(250, 0.1), (500, 0.9)],
                    },
                },
            },
            'spherical_emitter': {
                'type': 'sphere',
                'radius': self.emitter_radius,
                'center': [2, 0, 0],
                'emitter': {
                    'type': 'area',
                    'radiance': {
                        'type': 'uniform',
                        'value': 1,
                    },
                },
            },
        }



# -------------------------------------------------------------------
#            Test configs with discontinuities
# -------------------------------------------------------------------

# -------------------------------------------------------------------
#                           List configs
# -------------------------------------------------------------------
BASIC_CONFIGS_LIST = [
    ShoeboxAbsorptionConfig,
    ShoeboxScatteringConfig,
    SphericalEmitterRadianceConfig,
]

DISCONTINUOUS_CONFIGS_LIST = [
]

# List of configs that fail on integrators with depth less than three
INDIRECT_ILLUMINATION_CONFIGS_LIST = [
]

# List of integrators to test
# (Name, handles discontinuities, has render_backward, has render_forward)
INTEGRATORS_RENDER = [
    ('acoustic_ad',             False,  True,   True),
    ('acoustic_prb',            False,  True,   False),
    ('acoustic_ad_threepoint',  True,   True,   True),
    ('acoustic_prb_threepoint', True,   True,   False)
]

CONFIGS_PRIMAL   = []
CONFIGS_BACKWARD = []
CONFIGS_FORWARD  = []
for integrator_name, handles_discontinuities, has_render_backward, has_render_forward in INTEGRATORS_RENDER:
    todos = BASIC_CONFIGS_LIST + (DISCONTINUOUS_CONFIGS_LIST if handles_discontinuities else [])
    for config in todos:
        if (('direct' in integrator_name or 'projective' in integrator_name) and
                config in INDIRECT_ILLUMINATION_CONFIGS_LIST):
            continue

        CONFIGS_PRIMAL.append((integrator_name, config))
        if has_render_backward:
            CONFIGS_BACKWARD.append((integrator_name, config))
        if has_render_forward:
            CONFIGS_FORWARD.append((integrator_name, config))



# -------------------------------------------------------------------
#                           Unit tests
# -------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize('integrator_name, config', CONFIGS_PRIMAL)
def test10_rendering_primal(variants_all_ad_acoustic, integrator_name, config):
    config = config()
    config.initialize()

    config.integrator_dict['type'] = integrator_name
    integrator = mi.load_dict(config.integrator_dict, parallel=False)
    filename = join(output_dir, f"test_{config.name}_primal_ref.exr")
    etc_primal_ref = mi.TensorXf(mi.Bitmap(filename))
    etc = integrator.render(config.scene, seed=0, spp=config.spp)

    # FIXME: once the integrators normalize by spp, remove this normalization.
    etc /= dr.max(dr.abs(etc))
    etc_primal_ref /= dr.max(dr.abs(etc_primal_ref))

    error = dr.abs(etc - etc_primal_ref) / dr.maximum(dr.abs(etc_primal_ref), 2e-2)
    error_mean = dr.mean(error, axis=None)
    error_max = dr.max(error, axis=None)

    if error_mean > config.error_mean_threshold or error_max > config.error_max_threshold:
        print(f"Failure in config: {config.name}, {integrator_name}")
        print(f"-> error mean: {error_mean} (threshold={config.error_mean_threshold})")
        print(f"-> error max: {error_max} (threshold={config.error_max_threshold})")
        print(f'-> reference image: {filename}')
        filename = join(os.getcwd(), f"test_{integrator_name}_{config.name}_primal.exr")
        filename_ref = join(os.getcwd(), f"test_{integrator_name}_{config.name}_ref.exr")
        print(f'-> write current image: {filename}')
        mi.util.write_bitmap(filename, etc)
        mi.util.write_bitmap(filename_ref, etc_primal_ref)
        pytest.fail("ETC values exceeded configuration's tolerances!")


@pytest.mark.slow
@pytest.mark.skipif(os.name == 'nt', reason='Skip those memory heavy tests on Windows')
@pytest.mark.parametrize('integrator_name, config', CONFIGS_FORWARD)
def test11_rendering_forward(variants_all_ad_acoustic, integrator_name, config):
    mi.set_log_level(mi.LogLevel.Debug)
    config = config()
    config.initialize()

    config.integrator_dict['type'] = integrator_name
    integrator = mi.load_dict(config.integrator_dict)


    filename = join(output_dir, f"test_{config.name}_fwd_ref.exr")
    etc_fwd_ref = mi.TensorXf(mi.Bitmap(filename))

    theta = mi.Float(0.0)
    dr.enable_grad(theta)
    dr.set_label(theta, 'theta')
    config.update(theta)
    # We call dr.forward() here to propagate the gradient from the latent variable into
    # the scene parameter. This prevents dr.forward_to() in integrator.render_forward()
    # to trace gradients outside the dr.Loop().
    dr.forward(theta, dr.ADFlag.ClearEdges)

    dr.set_label(config.params, 'params')
    etc_fwd = integrator.render_forward(
        config.scene, seed=0, spp=config.spp, params=theta) / config.spp
    etc_fwd = dr.detach(etc_fwd)

    error = dr.abs(etc_fwd - etc_fwd_ref) / dr.maximum(dr.abs(etc_fwd_ref), 2e-1)
    error_mean = dr.mean(error, axis=None)
    error_max = dr.max(error, axis=None)

    if error_mean > config.error_mean_threshold or error_max > config.error_max_threshold:
        print(f"Failure in config: {config.name}, {integrator_name}")
        print(f"-> error mean: {error_mean} (threshold={config.error_mean_threshold})")
        print(f"-> error max: {error_max} (threshold={config.error_max_threshold})")
        print(f'-> reference image: {filename}')
        filename = join(os.getcwd(), f"test_{integrator_name}_{config.name}_image_fwd.exr")
        print(f'-> write current image: {filename}')
        mi.util.write_bitmap(filename, etc_fwd)
        filename = join(os.getcwd(), f"test_{integrator_name}_{config.name}_image_error.exr")
        print(f'-> write error image: {filename}')
        mi.util.write_bitmap(filename, error)
        pytest.fail("Gradient values exceeded configuration's tolerances!")


@pytest.mark.slow
@pytest.mark.skipif(os.name == 'nt', reason='Skip those memory heavy tests on Windows')
@pytest.mark.parametrize('integrator_name, config', CONFIGS_BACKWARD)
def test12_rendering_backward(variants_all_ad_acoustic, integrator_name, config):
    mi.set_log_level(mi.LogLevel.Debug)

    config = config()
    config.initialize()
    config.integrator_dict['type'] = integrator_name
    integrator = mi.load_dict(config.integrator_dict)

    filename = join(output_dir, f"test_{config.name}_fwd_ref.exr")
    etc_fwd_ref = mi.TensorXf(mi.Bitmap(filename))

    grad_in = 0.001
    etc_adj = dr.full(mi.TensorXf, grad_in, etc_fwd_ref.shape)

    theta = mi.Float(0.0)
    dr.enable_grad(theta)
    dr.set_label(theta, 'theta')
    config.update(theta)

    integrator.render_backward(config.scene, grad_in=etc_adj, seed=0, spp=config.spp, params=theta)

    grad = dr.grad(theta) / dr.width(etc_fwd_ref)
    print(f"grad: {grad}")
    grad_ref = dr.mean(etc_fwd_ref, axis=None) * grad_in
    print(f"grad_ref: {grad_ref}")

    error = dr.abs(grad - grad_ref) / dr.maximum(dr.abs(grad_ref), 1e-3)
    if error > config.error_mean_threshold_bwd:
        print(f"Failure in config: {config.name}, {integrator_name}")
        print(f"-> grad:     {grad}")
        print(f"-> grad_ref: {grad_ref}")
        print(f"-> error: {error} (threshold={config.error_mean_threshold_bwd})")
        print(f"-> ratio: {grad / grad_ref}")
        pytest.fail("Gradient values exceeded configuration's tolerances!")


# -------------------------------------------------------------------
#                      Generate reference images
# -------------------------------------------------------------------

if __name__ == "__main__":
    """
    Generate reference primal/forward ETCs for all configs.
    """
    parser = argparse.ArgumentParser(prog='GenerateConfigReferenceETCs')
    parser.add_argument('--spp', default=2**30, type=int,
                        help='Samples per pixel. Default value: 2**30.')
    args = parser.parse_args()

    mi.set_variant('cuda_acoustic', 'llvm_acoustic')

    if not exists(output_dir):
        os.makedirs(output_dir)

    for config in BASIC_CONFIGS_LIST + DISCONTINUOUS_CONFIGS_LIST:
        config = config()
        print(f"name: {config.name}")

        config.initialize()

        integrator_path = mi.load_dict({
            'type': 'acoustic_path',
            'speed_of_sound': config.speed_of_sound,
            'max_depth': config.max_depth,
            'max_time': config.max_time,
            'max_energy_loss': config.max_energy_loss,
        })

        # Primal render
        etc_ref = integrator_path.render(config.scene, seed=0, spp=args.spp)

        filename = join(output_dir, f"test_{config.name}_primal_ref.exr")
        mi.util.write_bitmap(filename, etc_ref)

        # Finite difference
        theta = mi.Float(-0.5 * config.ref_fd_epsilon)
        config.update(theta)
        etc_1 = integrator_path.render(config.scene, seed=0, spp=args.spp)
        dr.eval(etc_1)

        theta = mi.Float(0.5 * config.ref_fd_epsilon)
        config.update(theta)
        etc_2 = integrator_path.render(config.scene, seed=0, spp=args.spp)
        dr.eval(etc_2)

        etc_fd = (etc_2 - etc_1) / config.ref_fd_epsilon

        filename = join(output_dir, f"test_{config.name}_fwd_ref.exr")
        mi.util.write_bitmap(filename, etc_fd)
