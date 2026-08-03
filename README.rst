.. image:: https://readthedocs.org/projects/py-oma/badge/?version=latest
    :target: https://py-oma.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://app.codacy.com/project/badge/Grade/4c292ef58452482097d0ae49a3ed10f9
    :target: https://app.codacy.com/gh/pyOMA-dev/pyOMA/dashboard
.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :target: https://www.gnu.org/licenses/gpl-3.0
    :alt: License: GPL v3
.. image:: https://zenodo.org/badge/768642315.svg
  :target: https://doi.org/10.5281/zenodo.14936576

============================================
pyOMA - Operational Modal Analysis in Python
============================================

.. image:: https://raw.githubusercontent.com/pyOMA-dev/pyOMA/refs/heads/master/doc/_static/logo.png
  :width: 110
  :align: left

pyOMA is an open-source toolbox for Operational Modal Analysis (OMA) developed
by Simon Marwitz, Volkmar Zabel et al. at the Institute of Structural Mechanics (ISM)
of the Bauhaus-Universität Weimar. Operational Modal Analysis is a methodology for
the identification of structural modal properties from ambient (output-only)
vibration measurements. It is written in Python 3.9+.


 * **Documentation:** https://py-oma.readthedocs.io
 * **Source Code:** https://github.com/pyOMA-dev/pyOMA
 * **Citing in your work:** https://doi.org/10.5281/zenodo.14936576

--------------------------------
About Operational Modal Analysis
--------------------------------

In a broader sense OMA consists of a series of processes:

.. image:: https://raw.githubusercontent.com/pyOMA-dev/pyOMA/refs/heads/master/doc/_static/concept_map.png
  :width: 800
  :alt: blockdiagram


.. list-table::

      * - Vibration Testing
        - Acquiring vibration signals (acceleration, velocity, ...) from mechanical structures under ambient or forced excitation (wind, traffic, microtremors, shakers, ...), from single or multiple simultaneous measurement setups, as one-off campaigns or continuous monitoring installations
      * - Signal Processing
        - Filters, Windows, Decimation, Spectral Estimation, Correlation Function Estimation — interactively (GUI/Jupyter) or scripted
      * - System Identification
        - SSI-cov, SSI-data, pLSCF, PRCE and ERA; SSI-cov, SSI-data and pLSCF additionally support cross-validation and uncertainty quantification; PreGER and PoGER provide pre- and post-identification multi-setup merging
      * - Modal Analysis
        - Estimation of modal parameters (frequencies, damping ratios, mode shapes) from identified systems. Manually, using stabilization diagrams or automatically using multi-stage clustering methods.
      * - Post Processing
        - Plotting and comparison of mode shapes, PoSER/PoGER/PreGER multi-setup merging, uncertainty quantification, SHM


---------------------
Applications of pyOMA
---------------------

The toolbox is currently used on a daily basis to analyze the continuously
acquired vibration measurements of a structural health monitoring system (since 2015).
Further uses include various academic and commercial measurement campaigns
on civil engineering structures including bridges, towers/masts, wide-span floors, etc.

.. [Ref1] Simon Marwitz et al. “An Experimental Evaluation of Two Potential Improvements for 3D Laser Vibrometer Based Operational Modal Analysis”. In: Experimental Mechanics 57.8 (July 2017), pp. 1311–1325.

.. [Ref2] Simon Marwitz et al. “Modalanalyse von Monitoringdaten eines Sendeturms”. In: Bautechnik 95.4 (Mar. 2018), pp. 288–295.

.. [Ref3] Simon Marwitz et al. “Operational Modal Analysis with a 3D Laser Vibrometer without External Reference”. In: Rotating Machinery, Hybrid Test Methods, Vibro-Acoustics & Laser Vibrometry. Ed. by James De Clerck et al. Vol. 8. Proceedings of the 34th IMAC, A Conference and Exposition on Structural Dynamics 2016. Society of Experimental Mechanics. Springer International Publishing, Jan. 25, 2016, pp. 75–85.

.. [Ref4] Simon Marwitz et al. “Automatisierte Modalanalyse und Langzeitmonitoring eines rotationssymmetrischen Turmtragwerks”. In: Berichte der Fachtagung Baustatik-Baupraxis 13. Ed. by Günther Meschke et al. Vol. 13. Baustatik Baupraxis e. V.. Ruhr-Universität Bochum: Lehrstuhl für Statik und Dynamik der Ruhr-Universität Bochum, Mar. 2017, pp. 165–172.

.. [Ref5] Simon Marwitz et al. “Cross-Evaluation of two Measures for the Assessment of Estimated State-Space Systems in Operational Modal Analysis”. In: Proceedings of the 7th International Operational Modal Analysis Conference. Shaker Verlag GmbH, Germany, May 11, 2017, pp. 253–256.

.. [Ref6] Simon Marwitz et al. “Betrachtung von Unsicherheiten in der Modalanalyse mit der Stochastic Subspace Identification am Beispiel eines seilabgespannten Masts”. In: Tagungsband der 15. D-A-CH Tagung Erdbebeningenieurwesen und Baudynamik. Sept. 21, 2017.

.. [Ref7] Simon Marwitz et al. “Modale Identifikation aus Langzeit-Dehnungsmessungen an einem Sendeturm”. In: Tagungsband der VDI Baudynamik Tagung. Apr. 17, 2018.

.. [Ref8] Simon Marwitz et al. “Relations between the quality of identified modal parameters and measured data obtained by structural monitoring”. In: Conference Proceedings of ISMA2018 - USD2018. Sept. 17, 2018.

.. [Ref9] Zabel, V. et al. "Bestimmung von modalen Parametern seilabgespannter Rohrmasten". In:  Berichte der Fachtagung Baustatik-Baupraxis, Institut für Baustatik und Baudynamik. 2020.

.. [Ref10] Marwitz, S. et al. " Cross-Validation in Stochastic Subspace Identification". In: Proceedings of the IOMAC 2025. 2025.

.. [Ref11] Marwitz, S. "Quantification and Reduction of Polymorphic Uncertainties in Operational Modal Analysis". PhD Thesis, Bauhaus-Universität Weimar, 2026.





-------
Install
-------

Requirements
============

- Python ≥ 3.9 — https://www.python.org/ or https://www.anaconda.com/download
- NumPy, SciPy, Matplotlib (installed automatically)

Optional extras:

- Jupyter widgets (interactive notebook GUI): ``pip install "pyoma-toolbox[jupyter]"``
- Desktop PyQt6 GUI: ``pip install "pyoma-toolbox[gui]"``
- 3-D pyvista/VTK mode-shape backend — translucent surfaces and
  hardware-accelerated animation, used by default when installed:
  ``pip install "pyoma-toolbox[pyvista]"``

For the full experience — both interactive frontends and the 3-D backend
(recommended):

.. code-block:: bash

   pip install "pyoma-toolbox[gui, jupyter, pyvista]"

After installing the ``gui`` extra, the ``pyoma`` command starts the
desktop GUI directly - no script needed:

.. code-block:: bash

   pyoma

Install from source
===================

.. code-block:: bash

   git clone https://github.com/pyOMA-dev/pyOMA.git
   cd pyOMA
   pip install -e .

--------------------------
Get started with a project
--------------------------

For a GUI-only start with nothing pre-written, see "Quickest start" on
the `gui_usage <https://py-oma.readthedocs.io/en/latest/gui_usage.html>`_
page - just run ``pyoma``. The rest of this section covers the scripted
five-step workflow (geometry → signals → pre-processing → identification
→ stabilisation).

**Option A — Jupyter notebook (recommended)**

 #. Clone the repository and ``pip install -e .`` as shown above.
 #. Open ``scripts/single_setup_analysis.ipynb`` in JupyterLab.
 #. Run all cells — the notebook uses the bundled example data and requires no
    additional configuration.
 #. To analyse your own data, set ``EXAMPLE_DATA`` and ``SETUP_DIR`` to point
    to your files and assign ``PreProcessSignals.load_measurement_file`` to a
    callable that reads your measurement format.

**Option B — Plain Python script (PyQt6 GUI)**

 #. Install with ``pip install -e ".[gui]"``.
 #. Run ``scripts/single_setup_analysis.py`` from the repository root, or
    launch the interactive GUI directly with the ``pyoma`` command
    installed alongside the package.

Full step-by-step guide: https://py-oma.readthedocs.io/en/latest/getting_started.html

Full documentation: https://py-oma.readthedocs.io

------------
Getting help
------------

 #. In case of errors check that:

  * input files are formatted correctly

  * arguments are of the right type and order

  * search the internet for similar errors

 #. Open an issue at https://github.com/pyOMA-dev/pyOMA/issues

-----------------
Project Structure
-----------------

::

    pyOMA
    ├── pyOMA
    │   ├── core
    │   │  ├── PreProcessingTools.py   # GeometryProcessor, PreProcessSignals, SignalPlot
    │   │  ├── ModalBase.py            # base class for all identification methods
    │   │  ├── SSICovRef.py            # BRSSICovRef, PogerSSICovRef
    │   │  ├── SSIData.py              # SSIData, SSIDataMC, SSIDataCV
    │   │  ├── VarSSIRef.py            # SSI-cov with uncertainty (variance) estimation
    │   │  ├── MultiSetupSSI.py        # PreGERSSI, VarPreGERSSI (PreGER multi-setup)
    │   │  ├── PLSCF.py                # poly-reference Least-Squares Complex Frequency
    │   │  ├── VarPLSCF.py             # pLSCF with uncertainty (variance) estimation
    │   │  ├── PRCE.py                 # poly-reference Complex Exponential
    │   │  ├── ERA.py                  # Eigensystem Realisation Algorithm
    │   │  ├── StabilDiagram.py        # StabilCalc, StabilCluster, StabilPlot
    │   │  ├── PlotMSH.py              # ModeShapePlot
    │   │  ├── PostProcessingTools.py  # MergePoSER, compare_modes
    │   │  └── Helpers.py              # ConfigFile, utility functions
    │   ├── GUI
    │   │  ├── MultiSetupGUI.py       # main entry point (also: `pyoma` launcher)
    │   │  ├── GeometryProcessorGUI.py
    │   │  ├── PreProcessSignalsGUI.py
    │   │  ├── ChanDofEditorGUI.py
    │   │  ├── ModalAnalysisGUI.py    # hosts the per-method widgets below
    │   │  │   ├── SSICovRefGUI.py
    │   │  │   ├── SSIDataGUI.py
    │   │  │   ├── VarSSIRefGUI.py
    │   │  │   ├── PLSCFGUI.py
    │   │  │   ├── VarPLSCFGUI.py
    │   │  │   └── PRCEGUI.py
    │   │  ├── StabilGUI.py
    │   │  ├── PlotMSHGUI.py
    │   │  ├── JupyterGUI.py          # ipywidgets for Jupyter
    │   │  └── HelpersGUI.py          # shared GUI utilities

Additionally some further files are provided with it:

::

    ├── doc
    ├── input_files
    ├── scripts
    ├── tests
    │   ├── test_helpers.py
    │   ├── test_preprocessing.py
    │   ├── test_modal_methods.py
    │   ├── test_stabildiagram.py
    │   ├── test_multi_setup.py
    │   └── files
    │       └── ...
    ├── LICENSE
    ├── README.rst
    └── pyproject.toml


Current development is focused on the ``core`` package which contains all the algorithms.

The ``input_files`` packages provides templates for input files for automated and structured analysis of a dataset consisting of multiple measurements.

The ``scripts`` package contains templates for certain recurring tasks, as well as commonly used functions, derived from the core and GUI packages.

The ``tests`` package contains common use cases and files, which could be run to test if any changes in the modules result in breaking existing functionality.

The documentation is generated from the git repository by `Sphinx <https://www.sphinx-doc.org/>`_  automatically and available on `<https://py-oma.readthedocs.io/>`_



------------
Contributing
------------

Where to start:

 * New to the codebase? The `GUI development guide
   <https://py-oma.readthedocs.io/en/latest/gui_development.html>`_ walks through
   adding a GUI part for an OMA method, and the `API reference
   <https://py-oma.readthedocs.io/en/latest/api_reference.html>`_ documents the
   core and GUI classes.
 * Fork the project on GitHub and start development
 * Open a Pull Request to get your changes merged into the project
 * Run the test suite before submitting: ``pip install -e ".[dev]" && pytest``
 * Ensure the documentation can be built: navigate to the ``doc`` folder and run ``make clean && make html`` to catch any RST syntax errors.
 * Install the pre-commit hooks once after cloning, so GUI-related checks
   (Qt Designer ``.ui``/generated file sync, GUI smoke tests) run
   automatically before each commit: ``pip install -e ".[dev]" && pre-commit install``
