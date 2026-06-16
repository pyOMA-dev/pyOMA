'''
pyOMA - A toolbox for Operational Modal Analysis
Copyright (C) 2015 - 2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Created on 05.03.2021

@author: womo1998
'''
import sys

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from PyQt5.QtWidgets import QDoubleSpinBox, QSizePolicy
from PyQt5.QtCore import pyqtSignal, QTimer


def my_excepthook(type_, value, tback):
    '''
    make qt application not crash on errors
    log the exception here
    then call the default handler
    '''
    sys.__excepthook__(type_, value, tback)


class MyMplCanvas(FigureCanvasQTAgg):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=2.5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        self.compute_initial_figure()

        #
        FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(self,
                                        QSizePolicy.Expanding,
                                        QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def compute_initial_figure(self):
        pass


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
        assert isinstance(timeout, (int, float))
        self.timer.setInterval(timeout)
