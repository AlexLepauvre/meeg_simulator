MultiSim: A toolbox for simulating multivariate EEG/MEG data
=============================================================

**MultiSim** is a Python package for simulating multivariate EEG/MEG datasets with user-defined experimental effects.  
It enables principled testing and validation of decoding pipelines, source reconstruction methods, and statistical analyses.

Features
--------

- Simulate multivariate EEG/MEG data with known ground-truth effects
- Inject effects at specified time windows and across specified conditions
- Control signal-to-noise ratio, spatial covariance, temporal smoothing, and between-subject variability
- Generate multiple subjects for group-level statistical analysis
- Export to `MNE-Python <https://mne.tools/>`_ and `EEGLAB <https://sccn.ucsd.edu/eeglab/>`_ formats

Installation
------------

You can install the package using pip:

.. code-block:: bash

   pip install multisim

To install with full dependencies (e.g., for notebooks, visualization, or exporting to other formats):

.. code-block:: bash

   pip install multisim[full]

Usage Example
-------------

.. code-block:: python

   from multisim import Simulator
   import numpy as np

   # Define experimental design
   X = np.random.randn(100, 2)  # 100 trials, 2 experimental conditions

   # Define time window of effect and its location in design matrix
   t_win = np.array([[0.2, 0.5]])  # Effect between 200–500 ms
   effects = np.array([1])         # Effect linked to second condition

   # Simulate data
   sim = Simulator(
      X=X,
      noise_std=0.1,
      n_channels=64,
      n_subjects=20,
      tmin=-0.2,
      tmax=0.8,
      sfreq=250,
      t_win=t_win,
      effects=effects,
      effect_size=[0.5],  # Mahalanobis effect size
   )

   data = sim.data  # List of simulated subjects

Documentation
-------------

Documentation and tutorials are available at:

https://alexlepauvre.github.io/meeg_simulator/

License
-------

.. literalinclude:: ../../LICENSE
   :language: none

Citation
--------

If you use this toolbox in your research, please cite the accompanying paper (in prep).

.. toctree::
   :maxdepth: 2
   :hidden:

   Simulator class <api>
   Tutorials <tutorial/index>

