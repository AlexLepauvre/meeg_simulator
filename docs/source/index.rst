MultiSim: A toolbox for simulating multivariate EEG/MEG data
=============================================================

Motivation and Overview
--------
One critical challenge in designing analysis pipelines for MEG/EEG data is confirming that the pipeline is both sensitive (i.e., it detects real effects) and specific (i.e., it avoids false alarms). In most cases, it is not known a priori whether or when experimental effects are present in the data, and it is therefore not possible to assess the sensitivity and specificity of analysis pipelines based on the data one is trying to analyse. 

**MultiSim** is a Python package for simulating multivariate EEG/MEG datasets with user-defined experimental effects.  
It enables principled testing and validation of decoding pipelines, source reconstruction methods, and statistical analyses.

Features
--------
Specifically, the toolbox allows to:
- Specify a between-trial design (e.g., two conditions, Condition A and Condition B).
- Inject multivariate effects at particular time windows (e.g., Condition A is active from 100–200 ms, Condition B from 300–400 ms).
- Control signal-to-noise ratio, spatial covariance, temporal smoothing, and between-subject variability
- Generate multiple subjects for group-level statistical analysis
- Export to `MNE-Python <https://mne.tools/>`_ and `EEGLAB <https://sccn.ucsd.edu/eeglab/>`_ formats
- Validate that the pipeline recovers the known effects accurately.

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

