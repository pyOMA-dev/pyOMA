Examples
========

The following worked examples use the bundled steel-frame test structure dataset
(``tests/files/``).  Each is available as a plain Python script in ``scripts/``
and as an interactive Jupyter notebook rendered here.

For a full description of the test structure, the 3D scanning laser vibrometer
measurement technique, and the 15-setup scan arrangement, see
:doc:`example_data`.

.. toctree::
   :maxdepth: 1

   example_data
   _collections/single_setup_analysis
   _collections/multi_setup_analysis
   _collections/multi_setup_analysis_poger

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Notebook
     - Workflow
   * - :doc:`_collections/single_setup_analysis`
     - One measurement setup → SSI-cov/ref → stabilisation → mode shapes
   * - :doc:`_collections/multi_setup_analysis`
     - 15 setups → per-setup SSI → PoSER merge → global mode shapes
   * - :doc:`_collections/multi_setup_analysis_poger`
     - 15 setups → joint Hankel matrix → single SSI (PoGER) → mode shapes
