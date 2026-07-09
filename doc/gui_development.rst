GUI Development
=================

This page explains how pyOMA's PyQt6 desktop GUI (``pyOMA/GUI/``) is built,
so you can modify an existing window or add a new one. For what the windows
look like and how to use them, see :doc:`gui_usage`. The API reference for
every module is at :doc:`gui_reference`.

.. contents:: On this page
   :local:
   :depth: 2


Two independent GUI stacks
----------------------------

``pyOMA/GUI/`` contains two frontends that share no code beyond the core
algorithm classes they wrap:

* **PyQt6 desktop** — everything covered on this page:
  ``StabilGUI.py``, ``PlotMSHGUI.py``, ``GeometryProcessorGUI.py``,
  ``PreProcessSignalsGUI.py``, ``ModalAnalysisGUI.py``, ``ChanDofEditorGUI.py``,
  and the five per-method widgets (``SSICovRefGUI.py``, ``SSIDataGUI.py``,
  ``PLSCFGUI.py``, ``PRCEGUI.py``, ``VarSSIRefGUI.py``), plus shared helpers
  in ``HelpersGUI.py``.
* **ipywidgets/Jupyter** — ``JupyterGUI.py`` only. It does not import Qt and
  is out of scope for this page; see :ref:`gui_usage-jupyter`.


The ``.ui`` → ``build_ui.py`` → ``generated/`` pipeline
------------------------------------------------------------

Every window's layout is a Qt Designer file under ``pyOMA/GUI/ui/*.ui``,
compiled into a Python module under ``pyOMA/GUI/generated/ui_*.py``:

.. code-block:: bash

   # Open a form in Qt Designer, e.g.:
   designer pyOMA/GUI/ui/plscf.ui

   # After saving, regenerate the compiled Python module:
   python scripts/build_ui.py

**Never hand-edit files under** ``pyOMA/GUI/generated/`` — they carry a
"DO NOT EDIT" header and are fully determined by their ``.ui`` source.
``python scripts/build_ui.py --check`` verifies the two are in sync without
writing anything; this runs both in CI (``.github/workflows/gui-checks.yml``)
and as a pre-commit hook.


The hand-written module pattern
-----------------------------------

Each hand-written class inherits from both the appropriate Qt base class and
the generated ``Ui_*`` mixin, calls ``setupUi(self)``, then wires
signals/slots manually. The smallest complete example,
:class:`~pyOMA.GUI.PLSCFGUI.PLSCFWidget`:

.. code-block:: python

   class PLSCFWidget(QWidget, Ui_PLSCFWidget):
       def __init__(self, prep_signals, instance=None, parent=None):
           super().__init__(parent)
           self.prep_signals = prep_signals
           self.setupUi(self)
           self._wire_buttons()
           self.set_instance(instance if instance is not None else PLSCF(prep_signals))

       def _wire_buttons(self):
           self.btn_build_half_spectra.clicked.connect(self._on_build_half_spectra)
           self.btn_compute_modal_params.clicked.connect(self._on_compute_modal_params)

Top-level windows (``QMainWindow`` subclasses meant to be opened directly,
e.g. ``StabilGUI``, ``GeometryProcessorGUI``) additionally call ``self.show()``
at the end of ``__init__``; widgets meant to be embedded elsewhere (the
per-method widgets, hosted as tabs inside ``ModalAnalysisGUI``) and dialogs
(``ChanDofEditorGUI``) do not — their caller decides when to show them.


Shared helpers — HelpersGUI.py
------------------------------------

* :class:`~pyOMA.GUI.HelpersGUI.MyMplCanvas` — embeds a matplotlib
  ``Figure`` in a Qt widget (``FigureCanvasQTAgg`` wrapper). Supports Qt
  Designer "promoted widget" placeholders: Designer/``uic`` can only
  instantiate it with a bare ``parent`` at form-construction time, so a
  figure that doesn't exist yet (e.g. one owned by a plot object passed in
  later) is attached afterwards via ``set_figure()``.
* :class:`~pyOMA.GUI.HelpersGUI.DelayedDoubleSpinBox` — a ``QDoubleSpinBox``
  that debounces its ``valueChangedDelayed`` signal by 1.5 s after the last
  change, so an expensive recompute isn't triggered on every scroll tick.
* ``my_excepthook`` — installed as ``sys.excepthook`` so an unhandled
  exception in a Qt callback doesn't hard-crash the app.
* ``_parse_int_list`` — parses a comma-separated ``QLineEdit`` (e.g.
  "0,1,2") into a list of ints, or ``None`` for blank text ("use all");
  shared by every method widget's block-selection fields.


The ``start_*_gui()`` idiom
--------------------------------

Every top-level window has a module-level launcher function used by the
example scripts and tests:

.. code-block:: python

   def start_stabil_gui(stabil_plot, modal_data, geometry_data=None, ...):
       global app
       app = QApplication.instance() or QApplication(sys.argv)
       form = StabilGUI(stabil_plot, cmpl_plot, msh_plot)
       loop = QEventLoop()
       form.destroyed.connect(loop.quit)
       loop.exec()

This makes each window usable both standalone (``python scripts/single_setup_analysis.py``)
and from an already-running ``QApplication`` — e.g. ``StabilGUI`` constructs
and shows a ``ModeShapeGUI`` directly (not through its launcher) when a
mode-shape panel is requested.


The per-identification-method widget pattern
--------------------------------------------------

``SSICovRefGUI``, ``SSIDataGUI``, ``PLSCFGUI``, ``PRCEGUI``, and
``VarSSIRefGUI`` all follow one contract, and are the template to copy when
adding GUI support for a not-yet-covered method (e.g. ``ERA``, currently
missing — see the TODO on :doc:`index`):

* Constructor: ``(prep_signals, instance=None, parent=None)``. If *instance*
  is omitted, a fresh unbuilt object of the wrapped core class is created.
* ``set_instance(instance)`` adopts an existing (possibly partially or fully
  computed) instance and refreshes every field/button-enabled-state from it
  — used both by the constructor and by "Load State...".
* Each button runs exactly one step of the wrapped object's
  build → compute pipeline, directly on ``self.instance`` (mutating it in
  place), then re-runs the same field/state refresh as ``set_instance``.

``PLSCFGUI.py`` (2 build steps: ``build_half_spectra`` →
``compute_modal_params``) is the smallest complete example to read first;
``VarSSIRefGUI.py`` (4 steps, including a sensitivities-preparation step used
only for uncertainty quantification) is the most elaborate.

These widgets are hosted as tabs inside :class:`~pyOMA.GUI.ModalAnalysisGUI.ModalAnalysisGUI`,
which eagerly constructs one instance of every registered widget against the
same ``prep_signals`` and switches between them with a method-selector combo
box + ``QStackedWidget``.


Adding a new GUI window: step by step
-------------------------------------------

1. Design the form in Qt Designer, save it under ``pyOMA/GUI/ui/<name>.ui``.
2. ``python scripts/build_ui.py`` — generates ``pyOMA/GUI/generated/ui_<name>.py``.
3. Write the hand-written module (or, for a new identification method, a new
   widget class following the per-method pattern above, registered in
   ``ModalAnalysisGUI.py``'s method list).
4. Add tests to ``tests/test_gui.py``, marked ``@pytest.mark.gui``: construct
   the class directly (not via its ``start_*_gui()`` launcher, which blocks
   in a ``QEventLoop``) and register it with ``qtbot.addWidget(...)`` for
   automatic teardown.
5. Add a capture function to ``scripts/generate_gui_screenshots.py`` and run
   it to produce the new window's documentation screenshot.
6. Add the new module to :doc:`gui_reference`'s ``autosummary`` list, and a
   section + figure to :doc:`gui_usage`.
7. Before committing, run the same checks CI and pre-commit enforce::

    python scripts/build_ui.py --check
    python scripts/generate_gui_screenshots.py --check
    pytest -m gui tests/


Headless testing convention
--------------------------------

Every Qt-related test, CI job, and the screenshot script itself runs with::

    QT_QPA_PLATFORM=offscreen

set *before any Qt import* (see ``tests/conftest.py``), which renders widgets
to an in-memory framebuffer instead of a real display — this is what makes
``pytest -m gui`` and ``scripts/generate_gui_screenshots.py`` work in CI and
on machines with no display server. Pair it with
``matplotlib.use('Agg')`` (also set before any matplotlib import) so
embedded figures don't try to open an interactive backend.
