Attractor Networks Change Point Detection

Author: Eugene Tan

===============
Version History
===============
v1.0 - 9/6/2023 (Initial Upload of code)
v1.1 - 22/11/2023 (Added new code for public use)
v1.2
- Cleaned up comments and removed redundant code from AttractorNetworks_PublicUse.py
- Reimplemented comparisons against naive moving statistics
- Made usage of code with imported data clearer.
- Allowed additional parameters (cutoff ratio, bandwidth, significance level) to be more easily tunable

========
Overview
========
This code repository contains Python implementations of a new attractor networks change poitn detection algorithm. For a given input time series, a portion of the data is used to construct an attractor network (a Markov transition network representation of the underlying system's attractor). In the testing stage, incoming observations are compared against expected transition probabilities in the attractor network. The resulting level of surprise of each observation is calculated using an information theory definition of surprise. The calculated surprise metric is subsequently used to classify each new observation as abnormal or normal

================
Running the Code
================
1. Ensure all relevant Python dependencies are installed. Please check all imported packages in the scripts for a list of dependencies.
2. AttractorNetworks.py, ChaoticSystems.py, NetworkReconstructions.py, SurrogatesGenerator.py, all contain helper functions used to construct and attractor network and calculate surprise.
3. A demo of the algorithm for analysing change points in the Chua chaotic oscillator is provided in AttractorNetworks.py (runtime is approximately 5 minutes, depending on machine)

Note: Each file is setup as in the form of a notebook with cells demarcated with "# %%". User are advised to run AttractorNetworks.py cell by cell in Visual Studio Code.

N.B. There is also a version of the code that you can use to run the analysis on your own dataset for change point detection.

======================
Overview of Code Files
======================

AttractorNetworks.py - Contains bulk of helper functions for constructing attractor networks

ChaoticSystems.py - RK4 integrator implementation for integrating various dynamical systems

NetworkReconstruction.py - Code for calculating the dynamics component of the attractor network, and calculating surprise

SurrogatesGenerator.py - Implementation of the iterated Amplitude Adjusted Fourier Surrogates used in constructing artificial chaotic time series with abnormal behaviour (see paper)

AttractorNetworks_PublicUse.py - Version of the code that you can use to input your own dataset to run change point detection on.
