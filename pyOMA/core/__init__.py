# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
import logging
import sys

logging.basicConfig(stream=sys.stdout)

from .PreProcessingTools import PreProcessSignals, GeometryProcessor, SignalPlot
from .ModalBase import ModalBase
from .SSICovRef import BRSSICovRef, PogerSSICovRef
from .MultiSetupSSI import PreGERSSI, VarPreGERSSI
from .SSIData import SSIData, SSIDataMC
from .VarSSIRef import VarSSIRef
from .PLSCF import PLSCF
from .VarPLSCF import VarPLSCF
from .PRCE import PRCE
from .ERA import ERA
from .StabilDiagram import StabilCalc, StabilCluster, StabilPlot
from .ModeShapeBase import ModeShapeBase
from .PlotMSH import ModeShapePlot
from .PostProcessingTools import MergePoSER
from .Helpers import calculateMAC, calculateMPC, calculateMPD

# Optional pyvista backends. The module itself imports no plotting library,
# so this normally succeeds even without the pyOMA[pyvista] extra; the guard
# keeps the package importable should that ever stop being true.
try:
    from .PlotMSHpv import ModeShapePlotPVQt
    _PYVISTA_BACKENDS = ['ModeShapePlotPVQt']
except ImportError:  # pragma: no cover - exercised only without the extra
    _PYVISTA_BACKENDS = []

__all__ = [
    'PreProcessSignals', 'GeometryProcessor',
    'ModalBase',
    'BRSSICovRef', 'PogerSSICovRef',
    'PreGERSSI', 'VarPreGERSSI',
    'SSIData', 'SSIDataMC',
    'VarSSIRef',
    'PLSCF', 'VarPLSCF', 'PRCE', 'ERA',
    'StabilCalc', 'StabilCluster', 'StabilPlot',
    'ModeShapeBase', 'ModeShapePlot',
    'MergePoSER',
    'calculateMAC', 'calculateMPC', 'calculateMPD',
] + _PYVISTA_BACKENDS
