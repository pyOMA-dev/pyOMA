'''
Created on 04.03.2021

@author: womo1998
'''
import sys
from pathlib import Path
import matplotlib.pyplot as plt  # avoid import errors
import os

import numpy as np

from pyOMA.core.PreProcessingTools import PreProcessSignals, GeometryProcessor

from pyOMA.core.ModalBase import ModalBase
from pyOMA.core.PLSCF import PLSCF
from pyOMA.core.PRCE import PRCE
from pyOMA.core.SSICovRef import BRSSICovRef, PogerSSICovRef
from pyOMA.core.SSIData import SSIData, SSIDataMC
from pyOMA.core.VarSSIRef import VarSSIRef
#from pyOMA.core.ERA import *

from pyOMA.core.StabilDiagram import StabilCalc, StabilCluster, StabilPlot
from pyOMA.core.PostProcessingTools import MergePoSER
from pyOMA.core.PlotMSH import ModeShapePlot

from pyOMA.GUI.PlotMSHGUI import start_msh_gui
from pyOMA.GUI.StabilGUI import start_stabil_gui



def analysis_chain(tmpdir):
    # Generate some random measurement data
    num_channels = 10
    num_timesteps = 25 * 20 * 60  # 25 Hz for 20 minutes
    measurement = np.random.rand(num_timesteps, num_channels)

    # Initialize Geometry and PreProcessData
    geometry = GeometryProcessor.load_geometry(
        nodes_file='../input_files/grid',
        lines_file='../input_files/lines',
        parent_childs_file='../input_files/parent_childs')
    PreProcessSignals.load_measurement_file = lambda measurement: measurement
    prep_signals = PreProcessSignals.init_from_config(
        conf_file='../input_files/meas_1/setup_info.txt',
        meas_file=measurement,
        chan_dofs_file='../input_files/meas_1/channel_dofs')

    # test all functions of PreProcessData
    prep_signals.correct_offset()
    prep_signals.filter_signals(lowpass=12)
    prep_signals.decimate_signals(2)
    prep_signals.psd_welch(200)
    prep_signals.corr_welch(400)
    prep_signals.psd_blackman_tukey(200)
    prep_signals.welch(400)
    prep_signals.sv_psd()
    prep_signals.correlation(m_lags=400)
    prep_signals.sv_psd()

    prep_signals.save_state(tmpdir + 'test.npz')
    prep_signals = PreProcessSignals.load_state(tmpdir + 'test.npz')
    
    # for each OMA method
    for method, config in list(zip([PLSCF, PRCE, BRSSICovRef, SSIData, SSIDataMC, VarSSIRef],
                                   ['../input_files/meas_1/plscf_config.txt',
                                    '../input_files/meas_1/prce_config.txt',
                                    '../input_files/meas_1/ssi_config.txt',
                                    '../input_files/meas_1/ssi_config.txt',
                                    '../input_files/meas_1/ssi_config.txt',
                                    '../input_files/meas_1/varssi_config.txt']))[:]:
        modal_obj = method.init_from_config(config, prep_signals)
        # run the full analysis from the sample config file

        # run the full analysis manually
        # test save and load functions
        modal_obj.save_state(tmpdir + 'test.npz')
        modal_obj = method.load_state(tmpdir + 'test.npz', prep_signals)
        # pass the loaded object to StabilDiagram and PlotMSH

    # test all the functionality of StabilCalc
    # test the functionality of ModeShapePlot
    # test MergePoser


def PlotMSHGUI_test():
    working_dir = Path(sys.modules['__main__'].__file__.rstrip('basic_tests.py')) / 'files/'
    result_folder = working_dir / 'merged_poger/'
    geometry_data = GeometryProcessor.load_geometry(
        nodes_file=working_dir / 'grid.txt',
        lines_file=working_dir / 'lines.txt',
        parent_childs_file=working_dir / 'parent_childs.txt',)

    modal_data = PogerSSICovRef.load_state(result_folder / 'modal_data.npz')
    stabil_data = StabilCalc.load_state(result_folder / 'stabil_data.npz', modal_data)
    
    modeshapeplot = ModeShapePlot(
        geometry_data,
        modal_data=modal_data,
        stabil_calc=stabil_data)
    
    start_msh_gui(modeshapeplot)


def multi_setup_analysis():

    PreProcessSignals.load_measurement_file = np.load
    working_dir = Path(sys.modules['__main__'].__file__.rstrip('basic_tests.py')) / 'files/'

    geometry_data = GeometryProcessor.load_geometry(
        nodes_file=working_dir / 'grid.txt',
        lines_file=working_dir / 'lines.txt',
        parent_childs_file=working_dir / 'parent_childs.txt')

    meas_files = working_dir.glob('measurement*/')

    skip_existing = False
    save_results = False
    interactive = True

    m_lags = 400

    result_folder_merged = working_dir / 'merged_poger/'

    if not os.path.exists(result_folder_merged / 'modal_data.npz') \
            or not skip_existing:

        modal_data = PogerSSICovRef()

        for result_folder in list(meas_files):
            meas_name = os.path.basename(result_folder)

            if not os.path.exists(result_folder / 'prep_signals.npz') \
                    or not skip_existing:

                prep_signals = PreProcessSignals.init_from_config(
                    conf_file=result_folder / 'setup_info.txt',
                    meas_file=result_folder / (meas_name + '.npy'),
                    chan_dofs_file=result_folder / "channel_dofs.txt",)
                prep_signals.corr_blackman_tukey(m_lags)

                if save_results:
                    prep_signals.save_state(result_folder / 'prep_signals.npz')
            else:
                prep_signals = PreProcessSignals.load_state(
                    result_folder / 'prep_signals.npz')

            modal_data.add_setup(prep_signals)

        modal_data.pair_channels()

        modal_data.build_merged_subspace_matrix(199)
        modal_data.compute_modal_params(max_model_order=100, max_modes=24)
        
        if save_results:
            modal_data.save_state(result_folder_merged / 'modal_data.npz')
        prep_signals = None
    else:
        modal_data = PogerSSICovRef.load_state(
            result_folder_merged / 'modal_data.npz',)
        prep_signals = None

    if os.path.exists(result_folder_merged / 'stabil_data.npz') and skip_existing:
        stabil_calc = StabilCluster.load_state(
            result_folder_merged / 'stabil_data.npz', modal_data)
    else:
        stabil_calc = StabilCluster(modal_data)
    if interactive:
        stabil_plot = StabilPlot(stabil_calc)
        start_stabil_gui(stabil_plot, modal_data, geometry_data, prep_signals)

        if save_results:
            stabil_calc.save_state(result_folder_merged / 'stabil_data.npz')
            
    stabil_calc.export_results(working_dir / 'mode_export.txt')
            
    if interactive:

        mode_shape_plot = ModeShapePlot(
            stabil_calc=stabil_calc,
            geometry_data=geometry_data,
            modal_data=modal_data)
        start_msh_gui(mode_shape_plot)


def single_setup_analysis(
        result_folder,
        setup_info,
        meas_file,
        conf_file,
        method,
        geometry_data=None,
        chan_dofs_file=None,
        skip_existing=True,
        save_results=True,
        interactive=True):

    assert issubclass(method, ModalBase)

    for f in [result_folder, setup_info, meas_file, conf_file, chan_dofs_file]:
        if not os.path.exists(f):
            raise RuntimeError(f"The path {f} does not exist. Check your definitions.")

    if not os.path.exists(
            result_folder /
            'prep_signals.npz') or not skip_existing:
        prep_signals = PreProcessSignals.init_from_config(
            conf_file=setup_info,
            meas_file=meas_file,
            chan_dofs_file=chan_dofs_file)

        if save_results:
            prep_signals.save_state(result_folder / 'prep_signals.npz')
    else:
        prep_signals = PreProcessSignals.load_state(result_folder / 'prep_signals.npz')
    
    prep_signals.decimate_signals(10)
    
    if not os.path.exists(
            result_folder /
            'modal_data.npz') or not skip_existing:

        modal_data = method.init_from_config(conf_file, prep_signals)

        if save_results:
            modal_data.save_state(result_folder / 'modal_data.npz')
    else:
        modal_data = method.load_state(
            result_folder / 'modal_data.npz', prep_signals)

    if os.path.exists(result_folder / 'stabil_data.npz') and skip_existing:
        stabil_calc = StabilCalc.load_state(
            result_folder / 'stabil_data.npz', modal_data, prep_signals)
    else:
        stabil_calc = StabilCalc(modal_data, prep_signals)

    if interactive:
        stabil_plot = StabilPlot(stabil_calc)
        start_stabil_gui(stabil_plot, modal_data, geometry_data, prep_signals)

    if save_results:
        stabil_calc.save_state(result_folder / 'stabil_data.npz')

    return prep_signals, modal_data, stabil_calc


def merge_poser_test(skip_existing = False,
                     save_results = False,
                     interactive = True):

    PreProcessSignals.load_measurement_file = np.load

    working_dir = Path(sys.modules['__main__'].__file__.rstrip('basic_tests.py')) / 'files/'

    geometry_data = GeometryProcessor.load_geometry(
        nodes_file=working_dir / 'grid.txt',
        lines_file=working_dir / 'lines.txt')

    meas_files = working_dir.glob('measurement*/')
    
    merger = MergePoSER()


    for result_folder in meas_files:
        
        meas_name = os.path.basename(result_folder)
        
        prep_signals, modal_data, stabil_calc = single_setup_analysis(
            result_folder=result_folder,
            setup_info=result_folder / 'setup_info.txt',
            meas_file=result_folder / (meas_name + '.npy'),
            conf_file=working_dir / 'plscf_config.txt',
            method=PLSCF,
            geometry_data=geometry_data,
            chan_dofs_file=result_folder / "channel_dofs.txt",
            skip_existing=skip_existing,
            save_results=save_results,
            interactive=True)
        
        merger.add_setup(prep_signals, modal_data, stabil_calc)

    merger.merge()

    merger.save_state(working_dir / 'merged_poser.npz')

    if interactive:

        mode_shape_plot = ModeShapePlot(geometry_data=geometry_data,
                                        merged_data=merger)
        start_msh_gui(mode_shape_plot)



if __name__ == '__main__':
    # analysis_chain(tmpdir='/dev/shm/womo1998/')
    # PlotMSHGUI_test()
    # merge_poser_test(False,False,True)
    
    
    multi_setup_analysis()
