# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
import logging
import os
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
    from .PlotMSHpv import ModeShapePlotPVQt, ModeShapePlotPVJupyter
    _PYVISTA_BACKENDS = ['ModeShapePlotPVQt', 'ModeShapePlotPVJupyter']
except ImportError:  # pragma: no cover - exercised only without the extra
    _PYVISTA_BACKENDS = []

logger = logging.getLogger(__name__)

#: Optional override for the desktop mode-shape backend, honoured by the
#: example scripts and the Qt GUIs. Set to ``'pyvista'``, ``'matplotlib'`` or a
#: ModeShapeBase subclass to force a choice; leave ``None`` to defer to the
#: ``PYOMA_MSH_BACKEND`` environment variable.
MSH_BACKEND = None


def _pyvista_backend_for(context):
    '''Return the pyvista backend class for *context* (``'qt'``/``'notebook'``).'''
    if not _PYVISTA_BACKENDS:
        raise ImportError(
            "Mode-shape backend 'pyvista' was requested but the "
            "pyOMA[pyvista] extra is not installed.")
    return ModeShapePlotPVJupyter if context == 'notebook' else ModeShapePlotPVQt


def _msh_backend_from_name(name, context='qt'):
    '''Map a backend name to a class, or return *None* if unrecognised.

    *context* selects the pyvista variant: ``'qt'`` -> the desktop
    :class:`ModeShapePlotPVQt`, ``'notebook'`` -> the Jupyter
    :class:`ModeShapePlotPVJupyter`. The matplotlib backend serves both.
    '''
    key = str(name).lower()
    if key in ('pyvista', 'pv', 'vtk'):
        return _pyvista_backend_for(context)
    if key in ('matplotlib', 'mpl'):
        return ModeShapePlot
    return None


def resolve_mode_shape_backend(context='qt'):
    '''Return the mode-shape backend class the GUIs / notebooks should construct.

    Resolution order:

    1. the module-level :data:`MSH_BACKEND` override (a class, ``'pyvista'`` or
       ``'matplotlib'``), when set;
    2. the ``PYOMA_MSH_BACKEND`` environment variable
       (``'pyvista'`` / ``'matplotlib'`` / ``'auto'``);
    3. ``'auto'`` or unset -> the pyvista backend when the ``pyOMA[pyvista]``
       extra is importable, else :class:`ModeShapePlot`.

    Parameters
    ----------
    context : {'qt', 'notebook'}, optional
        Which pyvista variant to return when pyvista is chosen: the desktop
        :class:`ModeShapePlotPVQt` (``'qt'``, the default, used by the example
        scripts and the PyQt GUIs) or the Jupyter
        :class:`ModeShapePlotPVJupyter` (``'notebook'``, used by the example
        notebooks). The matplotlib backend is context-independent. An explicit
        class set in :data:`MSH_BACKEND` is returned as-is, ignoring *context*.

    Returns
    -------
    type
        :class:`ModeShapePlotPVQt`, :class:`ModeShapePlotPVJupyter` or
        :class:`ModeShapePlot`.
    '''
    if MSH_BACKEND is not None:
        if isinstance(MSH_BACKEND, type):
            return MSH_BACKEND
        cls = _msh_backend_from_name(MSH_BACKEND, context)
        if cls is not None:
            return cls
        logger.warning("Unknown MSH_BACKEND %r; falling back to auto.", MSH_BACKEND)

    env = os.environ.get('PYOMA_MSH_BACKEND', 'auto')
    if env.lower() not in ('auto', ''):
        cls = _msh_backend_from_name(env, context)
        if cls is not None:
            return cls
        logger.warning("Unknown PYOMA_MSH_BACKEND %r; falling back to auto.", env)

    if _PYVISTA_BACKENDS:
        return _pyvista_backend_for(context)
    return ModeShapePlot


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
    'MSH_BACKEND', 'resolve_mode_shape_backend',
    'MergePoSER',
    'calculateMAC', 'calculateMPC', 'calculateMPD',
] + _PYVISTA_BACKENDS
