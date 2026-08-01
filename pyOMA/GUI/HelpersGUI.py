# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""Shared helper widgets and utilities for the pyOMA GUIs."""
import os
import sys

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PyQt6.QtWidgets import QDoubleSpinBox, QSizePolicy, QFileDialog, QMessageBox
from PyQt6.QtCore import pyqtSignal, QTimer


def save_figure_dialog(parent, canvas, caption="Choose a filename to save to"):
    """Prompt for a filename and save *canvas* to it.

    Backend-neutral: a matplotlib ``FigureCanvas`` offers its full filetype
    list and saves via ``savefig``; a pyvista ``QtInteractor`` (which has no
    such methods) offers a PNG and saves via ``screenshot``. Returns the
    chosen filename, or ``''`` if the user cancelled.
    """
    is_mpl = hasattr(canvas, 'get_supported_filetypes_grouped')
    if is_mpl:
        sorted_filetypes = sorted(canvas.get_supported_filetypes_grouped().items())
        filters = ';;'.join(
            '%s (%s)' % (name, " ".join('*.%s' % ext for ext in exts))
            for name, exts in sorted_filetypes)
        from matplotlib import rcParams
        startpath = os.path.expanduser(rcParams.get('savefig.directory', ''))
        start = os.path.join(startpath, canvas.get_default_filename())
    else:
        start = 'geometry.png'
        filters = 'PNG image (*.png);;All Files (*)'

    fname, _ext = QFileDialog.getSaveFileName(
        parent, caption=caption, directory=start, filter=filters)
    if fname:
        if is_mpl:
            canvas.figure.savefig(fname)
        else:
            if not os.path.splitext(fname)[1]:
                fname += '.png'
            canvas.screenshot(fname)
    return fname


class UnsavedChangesMixin:
    """closeEvent() prompts to save if self._dirty is True.

    Subclasses must set self._dirty = True on every mutating action,
    and implement self._do_save() (call the GUI's own save_state/
    save method) and self._dirty = False after a successful save.
    """

    def _build_save_prompt(self):
        """Construct (but don't yet exec()) the close-time save prompt.

        A plain QMessageBox instance (rather than the static .question()
        convenience method) is needed so the "close without saving" button
        can be relabelled to "Continue" - .question() only offers the
        built-in standard-button texts, with no way to rename one.
        """
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved changes")
        box.setText("Save your work before closing?")
        box.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel)
        box.button(QMessageBox.StandardButton.Discard).setText("Continue")
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        return box

    def _prompt_save_on_close(self, event):
        if not getattr(self, '_dirty', False):
            return True
        answer = self._build_save_prompt().exec()
        if answer == QMessageBox.StandardButton.Cancel:
            event.ignore()
            return False
        if answer == QMessageBox.StandardButton.Save:
            self._do_save()
        return True


def _parse_int_list(text):
    """Parse a comma-separated list of ints; blank text means "use all"."""
    text = text.strip()
    if not text:
        return None
    return [int(tok) for tok in text.split(',') if tok.strip()]


def my_excepthook(type_, value, tback):
    '''
    make qt application not crash on errors
    log the exception here
    then call the default handler
    '''
    sys.__excepthook__(type_, value, tback)


class MyMplCanvas(FigureCanvasQTAgg):
    """Embeddable Matplotlib canvas widget for PyQt6 GUIs.

    A thin wrapper around :class:`FigureCanvasQTAgg` that creates a figure
    with a single axes and handles size-policy setup automatically.  Also
    usable as a Designer "promoted widget" placeholder for an externally
    supplied figure (see :meth:`set_figure`) — Designer/uic can only ever
    instantiate this with ``parent`` at form-construction time, so a figure
    that doesn't exist yet (e.g. one owned by a plot object passed in later)
    must be attached afterwards rather than through the constructor.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    width : float, optional
        Figure width in inches.  Default is 5.
    height : float, optional
        Figure height in inches.  Default is 2.5.
    dpi : int, optional
        Figure resolution in dots per inch.  Default is 100.
    figure : matplotlib.figure.Figure, optional
        Use this existing figure instead of creating a new one with its own
        axes.  When given, ``width``/``height``/``dpi`` are ignored and
        :meth:`compute_initial_figure` is not called.
    """

    def __init__(self, parent=None, width=5, height=2.5, dpi=100, figure=None):
        if figure is None:
            figure = Figure(figsize=(width, height), dpi=dpi)
            self.axes = figure.add_subplot(111)
            self.compute_initial_figure()

        FigureCanvasQTAgg.__init__(self, figure)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(self,
                                        QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def compute_initial_figure(self):
        pass

    def set_figure(self, figure):
        '''
        Re-point this already-constructed canvas at a different figure.

        Needed for Designer-promoted canvases: the widget is instantiated
        with its own placeholder figure before the real one (e.g. a 3-D
        mode-shape figure owned by a plot object) is available.
        '''
        self.figure = figure
        figure.set_canvas(self)


class DelayedDoubleSpinBox(QDoubleSpinBox):
    '''
    reimplementation of QDoubleSpinBox to delay the emit of the
    valueChanged signal by 1.5 seconds after the last change of the value
    this allows for a function to be directly connected to the signal
    without the need to check for further changes of the value
    else when the user clicks through the values it would emit a
    lot of signals and the connected funtion would run this many times
    note that you have to connect to valueChangedDelayed signal if
    you want to make use of this functionality
    valueChanged signal works as in QDoubleSpinBox
    '''
    # define custom signals
    valueChangedDelayed = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        '''
        inherit from QDoubleSpinBox
        instantiate a timer and set its default timeout value (1500 ms)
        connect the valueChanged signal of QDoubleSpinBox to the
        start () slot of QTimer
        connect the timeout () signal of QTimer to delayed emit
        '''
        super(DelayedDoubleSpinBox, self).__init__(*args, **kwargs)
        self.timer = QTimer()
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.delayed_emit)
        self.valueChanged[float].connect(self.timer.start)

    # @pyqtSlot()
    def delayed_emit(self):
        '''
        stop the timer and send the current value of the QDoubleSpinBox
        '''
        self.timer.stop()
        self.valueChangedDelayed.emit(self.value())

    def set_timeout(self, timeout):
        '''
        set the timeout of the timer to a custom value
        '''
        if not isinstance(timeout, (int, float)):
            raise TypeError(f"timeout must be int or float, got {type(timeout).__name__!r}")
        self.timer.setInterval(timeout)
