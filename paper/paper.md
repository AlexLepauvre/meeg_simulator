---
title: 'MultiSim: A Python Toolbox for Simulating Datasets with time resolved multivariate effects'
tags:
  - Python
  - neuroscience
  - EEG
  - MEG
  - iEEG
  - LFP
  - simulation
authors:
  - name: Alex Lepauvre
    orcid: 0000-0002-4191-1578
    corresponding: true
    affiliation: "1, 2"
  - name: Qian Chu
    orcid: 0000-0003-2308-6102
    affiliation: "1, 3, 4, 5"
  - name: Lucia Melloni
    orcid: 0000-0001-8743-5071
    affiliation: "1, 6, 7"
  - name: Peter Zeidman
    orcid: 0000-0003-3610-6619
    affiliation: "8"
affiliations:
  - name: Neural Circuits, Consciousness and Cognition Research Group, Max Planck Institute for Empirical Aesthetics, Frankfurt am Main, Germany
    index: 1
    ror: 000rdbk18
  - name: Donders Institute for Brain, Cognition and Behaviour, Radboud University Nijmegen, Nijmegen, The Netherlands
    index: 2
    ror: 016xsfp80
  - name: Max Planck – University of Toronto Centre for Neural Science and Technology
    index: 3
  - name: Krembil Brain Institute, Toronto Western Hospital, University Health Network, Toronto, ON, Canada
    index: 4
    ror: 042xt5161
  - name: Institute of Biomedical Engineering, University of Toronto, Toronto, ON, Canada
    index: 5
    ror: 03dbr7087
  - name: Department of Neurology, New York University Grossman School of Medicine, New York, NY, USA
    index: 6
    ror: 0190ak572
  - name: Predictive Brain Department, Research Center One Health Ruhr, University Alliance Ruhr, Faculty of Psychology, Ruhr University Bochum, Bochum, Germany
    index: 7
    ror: 04tsk2644
  - name: Wellcome Centre for Human Neuroimaging, Institute of Neurology, University College London, London, UK
    index: 8
    ror: 02jx3x895
date: 1 July 2025
bibliography: paper.bib
header-includes:
  - \usepackage{amsmath}
  - \providecommand{\bm}[1]{\boldsymbol{#1}}
---

# Summary

In MEG/EEG research, validating analysis pipelines is hampered by the lack of ground-truth neural signals in real data. SimMEG fills this gap by generating realistic, time-locked multivariate effects of known magnitude that you can inject into simulated sensor data. You can then run any pipeline—e.g. decoding, sensor-level statistics, or source estimation—against these datasets to benchmark sensitivity and specificity.

Key benefits include:

- Testing whether your pipeline reliably detects effects of a chosen size.  
- Providing demonstrable, reproducible benchmarks for reviewers or collaborators.  
- Offering a controlled teaching environment for newcomers.  

Below, we describe the rationale (Statement of needs), and the data-generation method (Methods), a hands-on example (Results), and potential extensions (Discussion).  

# Statement of needs

Multivariate pattern analysis (MVPA) is now routine in cognitive neuroscience for probing how the brain represents information [@ritchie2019decoding;@haynes2006decoding;@kriegeskorte2008representational;@haxby2001distributed;@poldrack2009decoding]. Applied to high-temporal-resolution electrophysiology signals such as electro and magneto-encephalography (EEG and MEG respectively), decoding techniques reveal the millisecond-by-millisecond unfolding of mental representations [@cichy2014resolving;@king2014characterizing;@king2016brain;@cogitate2025adversarial;@kok2017prior]. Strikingly, despite the ubiquity of MVPA techniques, to our knowledge, no method exists to test the sensitivity and specificity of decoding analysis pipelines, nor to estimate, before data collection, how many trials and how many participants are required to detect an effect of a given size.

MultiSim addresses this gap by letting investigators simulate time-resolved multi-channel signals with parameters matching that of their recording setups, and specify multivariate effects with known timing, spatialization and strength, while controlling channel covariance, sensory noise and between subjects variability. Our algorithm produces multi-subject data sets in which ground truth effects are known. By running their pipeline on these data, researchers obtain a direct read-out of its true-positive rate (can it recover the injected effects?) and false-positive rate (does it raise alarms when nothing is present). In addition, our simulator can be used to perform computational power analysis, to determine the number of trials and subjects, by iterating over these parameters.

This toolbox promotes best-practice MVPA by giving researchers a tailored benchmark for their specific experimental designs, a testbed for developing new decoding methods, and a principled way to check that planned studies are properly powered—ultimately enabling more reliable and efficient investigations of brain function.

# Code Quality and Documentation

SimMEG is hosted on GitHub. Examples and API documentation are available on the platform [here](https://alexlepauvre.github.io/meeg_simulator/). We provide installation guides, algorithm introductions, and examples of using the package with [Jupyter Notebook](https://alexlepauvre.github.io/meeg_simulator/tutorial/index.html). We further provide the full mathematical details of our simulation [here](https://alexlepauvre.github.io/meeg_simulator/tutorial/06-mathematical_details.html). The package is available on Linux, macOS and Windows for Python >=3.12
It can be installed with pip install simMEG. To ensure high code quality, all implementations adhere to the PEP8 code style [REF], enforced by ruff [REF], the code formatter black and the static analyzer prospector. The documentation is provided through docstrings using the NumPy conventions and build using Sphinx.

# Acknowledgements

# References

# Supplementary
