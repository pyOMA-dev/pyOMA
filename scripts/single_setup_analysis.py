import sys
# repo_path = '/usr/wrk/people9/sima9999/git/pyOMA'
# repo_path = '/ismhome/staff/womo1998/git/pyOMA'
# sys.path.append(repo_path)
import os
from pathlib import Path

import numpy as np
from pyOMA.core.PreProcessingTools import PreProcessSignals, GeometryProcessor
from pyOMA.core.PLSCF import PLSCF
from pyOMA.core.PRCE import PRCE
from pyOMA.core.SSICovRef import BRSSICovRef
from pyOMA.core.SSIData import SSIData, SSIDataMC
from pyOMA.core.VarSSIRef import VarSSIRef
from pyOMA.core.StabilDiagram import StabilCalc, StabilPlot, StabilCluster
from pyOMA.core.PlotMSH import ModeShapePlot

from pyOMA.GUI.PlotMSHGUI import start_msh_gui
from pyOMA.GUI.StabilGUI import start_stabil_gui


# Define a function that loads the provided measurement file(s)
PreProcessSignals.load_measurement_file = np.load

working_dir = Path(f'/home/sima9999/git/pyOMA/tests/files/')
result_folder = Path(f'{working_dir}/measurement_1/')
meas_name = os.path.basename(result_folder)
setup_info=result_folder / 'setup_info.txt'
meas_file=result_folder / (meas_name + '.npy')
chan_dofs_file=result_folder / "channel_dofs.txt"

# Select OMA Method, one of: PLSCF PRCE BRSSICovRef PogerSSICovRef SSIData SSIDataMC VarSSIRef
method=BRSSICovRef
conf_file=working_dir / 'ssi_config.txt'

# define script switches
skip_existing=False
save_results=False
interactive=True


geometry_data = GeometryProcessor.load_geometry(
    nodes_file=working_dir / 'grid.txt',
    lines_file=working_dir / 'lines.txt',
    parent_childs_file=working_dir / 'parent_child_assignments.txt')

if not os.path.exists(result_folder / 'prep_signals.npz') or not skip_existing:
    prep_signals = PreProcessSignals.init_from_config(
        conf_file=setup_info,
        meas_file=meas_file,
        chan_dofs_file=chan_dofs_file)

else:
    prep_signals = PreProcessSignals.load_state(result_folder / 'prep_signals.npz')
    
prep_signals.decimate_signals(3)
prep_signals.decimate_signals(3)


if not os.path.exists(
        result_folder /
        'modal_data.npz') or not skip_existing:

    modal_data = method.init_from_config(conf_file, prep_signals)
    
    if save_results:
        prep_signals.save_state(result_folder / 'prep_signals.npz')
        modal_data.save_state(result_folder / 'modal_data.npz')
else:
    modal_data = method.load_state(
        result_folder / 'modal_data.npz', prep_signals)

if os.path.exists(result_folder / 'stabil_data.npz') and skip_existing:
    stabil_calc = StabilCluster.load_state(
        result_folder / 'stabil_data.npz', modal_data)
else:
    stabil_calc = StabilCluster(modal_data)
stabil_calc.export_results('/usr/scratch4/sima9999/test.txt')

if interactive:
    stabil_plot = StabilPlot(stabil_calc)
    start_stabil_gui(stabil_plot, modal_data, geometry_data)

if save_results:
    stabil_calc.save_state(result_folder / 'stabil_data.npz')

if interactive:

    mode_shape_plot = ModeShapePlot(
        prep_signals=prep_signals,
        stabil_calc=stabil_calc,
        geometry_data=geometry_data,
        modal_data=modal_data)
    start_msh_gui(mode_shape_plot)
