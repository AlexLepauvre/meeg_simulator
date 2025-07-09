import numpy as np
import pandas as pd
import pytest
import multisim
from multisim import Simulator



def test_basic_import():
    assert hasattr(multisim, "__version__")


def test_simulator_init_minimal():
    """Test basic instantiation with minimal, valid parameters."""
    X = pd.DataFrame(np.random.randn(10, 1), columns=["cond"])
    effects = [
        {
            "condition": "cond",
            "windows": [ [0.1, 0.3] ],
            "effect_size": 1.0
        }
    ]
    sim = Simulator(
        X=X,
        noise_std=0.1,
        n_channels=4,
        n_subjects=2,
        tmin=-0.1,
        tmax=0.5,
        sfreq=100,
        effects=effects
    )
    assert sim.n_epochs == 10
    assert sim.n_exp_conditions == 1
    assert sim.n_samples > 0
    assert len(sim.data) == 2
    assert isinstance(sim.data[0], np.ndarray)


def test_invalid_tmax_tmin():
    X = pd.DataFrame(np.random.randn(5, 1), columns=["cond"])
    effects = [
        {"condition": "cond", "windows": [ [0.1, 0.2] ], "effect_size": 1.0}
    ]
    with pytest.raises(ValueError, match="tmax must be greater than tmin"):
        Simulator(
            X=X,
            noise_std=0.1,
            n_channels=2,
            n_subjects=1,
            tmin=0.5,
            tmax=0.5,
            sfreq=100,
            effects=effects
        )


def test_invalid_sfreq():
    X = pd.DataFrame(np.random.randn(5, 1), columns=["cond"])
    effects = [
        {"condition": "cond", "windows": [ [0.1, 0.2] ], "effect_size": 1.0}
    ]
    with pytest.raises(ValueError, match="sfreq must be positive"):
        Simulator(
            X=X,
            noise_std=0.1,
            n_channels=2,
            n_subjects=1,
            tmin=0.0,
            tmax=0.5,
            sfreq=0,
            effects=effects
        )


def test_no_samples():
    X = pd.DataFrame(np.random.randn(5, 1), columns=["cond"])
    effects = [
        {"condition": "cond", "windows": [ [0.1, 0.2] ], "effect_size": 1.0}
    ]
    with pytest.raises(ValueError, match="Derived n_samples must be"):
        Simulator(
            X=X,
            noise_std=0.1,
            n_channels=2,
            n_subjects=1,
            tmin=0.0,
            tmax=0.01,
            sfreq=1,
            effects=effects
        )


def test_invalid_ch_cov_shape():
    X = pd.DataFrame(np.random.randn(5, 1), columns=["cond"])
    effects = [
        {"condition": "cond", "windows": [ [0.1, 0.2] ], "effect_size": 1.0}
    ]
    bad_cov = np.eye(3)  # wrong size
    with pytest.raises(ValueError, match="ch_cov must have shape"):
        Simulator(
            X=X,
            noise_std=0.1,
            n_channels=2,
            n_subjects=1,
            tmin=0.0,
            tmax=0.5,
            sfreq=100,
            effects=effects,
            ch_cov=bad_cov
        )


def test_kernel():
    X = pd.DataFrame(np.random.randn(5, 1), columns=["cond"])
    effects = [
        {"condition": "cond", "windows": [ [0.1, 0.2] ], "effect_size": 1.0}
    ]
    bad_kern = np.eye(2)  # wrong shape
    with pytest.raises(ValueError, match="kern must be a 1-D non-empty array"):
        Simulator(
            X=X,
            noise_std=0.1,
            n_channels=2,
            n_subjects=1,
            tmin=0.0,
            tmax=0.5,
            sfreq=100,
            effects=effects,
            kern=bad_kern
        )


def test_invalid_effects_format():
    X = pd.DataFrame(np.random.randn(5, 1), columns=["cond"])
    effects = [
        ["cond", [0.1, 0.2], 1.0]
    ]
    with pytest.raises(TypeError, match=f"Effect #{0} is not a dict."):
        Simulator(
            X=X,
            noise_std=0.1,
            n_channels=2,
            n_subjects=1,
            tmin=0.0,
            tmax=0.5,
            sfreq=100,
            effects=effects
        )


def test_invalid_effects_missing_condition():
    X = pd.DataFrame(np.random.randn(5, 1), columns=["cond"])
    effects = [
        {"windows": [ [0.1, 0.2] ], "effect_size": 1.0}
    ]
    with pytest.raises(ValueError, match="must have 'condition' and 'windows'"):
        Simulator(
            X=X,
            noise_std=0.1,
            n_channels=2,
            n_subjects=1,
            tmin=0.0,
            tmax=0.5,
            sfreq=100,
            effects=effects
        )


def test_invalid_effects_unknown_condition():
    X = pd.DataFrame(np.random.randn(5, 1), columns=["cond"])
    effects = [
        {"condition": "nonexistent", "windows": [ [0.1, 0.2] ], "effect_size": 1.0}
    ]
    with pytest.raises(ValueError, match="unknown condition"):
        Simulator(
            X=X,
            noise_std=0.1,
            n_channels=2,
            n_subjects=1,
            tmin=0.0,
            tmax=0.5,
            sfreq=100,
            effects=effects
        )


def test_malformed_window():
    X = pd.DataFrame(np.random.randn(5, 1), columns=["cond"])
    effects = [
        {"condition": "cond", "windows": [], "effect_size": 1.0}
    ]
    with pytest.raises(ValueError, match=f"Effect #{0}: 'windows' malformed."):
        Simulator(
            X=X,
            noise_std=0.1,
            n_channels=2,
            n_subjects=1,
            tmin=0.0,
            tmax=0.5,
            sfreq=100,
            effects=effects
        )


def test_invalid_effects_both_amp_and_size():
    X = pd.DataFrame(np.random.randn(5, 1), columns=["cond"])
    effects = [
        {
            "condition": "cond",
            "windows": [ [0.1, 0.2] ],
            "effect_size": 1.0,
            "effect_amp": 0.5
        }
    ]
    with pytest.raises(ValueError, match="either.*effect_size.*not both"):
        Simulator(
            X=X,
            noise_std=0.1,
            n_channels=2,
            n_subjects=1,
            tmin=0.0,
            tmax=0.5,
            sfreq=100,
            effects=effects
        )


def test_invalid_effects_missing_amp_and_size():
    X = pd.DataFrame(np.random.randn(5, 1), columns=["cond"])
    effects = [
        {
            "condition": "cond",
            "windows": [ [0.1, 0.2] ]
        }
    ]
    with pytest.raises(ValueError, match="must provide effect_size or effect_amp"):
        Simulator(
            X=X,
            noise_std=0.1,
            n_channels=2,
            n_subjects=1,
            tmin=0.0,
            tmax=0.5,
            sfreq=100,
            effects=effects
        )

