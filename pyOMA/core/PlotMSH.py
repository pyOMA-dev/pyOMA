'''
pyOMA - A toolbox for Operational Modal Analysis
Copyright (C) 2015 - 2021  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.

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

Module PlotMSH contains classes and functions for plotting mode shapes
obtained from any of the classes derived from ModalBase of the pyOMA project

.. TODO::
 * Implement scale (for correct drawing of axis arrows)
 * Use current axes settings when starting the animation
 * Remove PyQT dependency -> move the signal definitions somewhere else. Where?
 * Restore functionality needed to create the geometry in another GUI
 * Use the logging module to replace print commands at an appropriate
   logging level
 * Implement the plotting in  pyvista for better and faster 3D graphics
   `https://docs.pyvista.org/examples/99-advanced/warp-by-vector-eigenmodes.html`

'''

# system i/o
import matplotlib.animation
import matplotlib.patches
import mpl_toolkits.mplot3d.axes3d
from .PostProcessingTools import MergePoSER
from .VarSSIRef import VarSSIRef
from .SSICovRef import PogerSSICovRef
from .ModalBase import ModalBase
from .PreProcessingTools import PreProcessSignals, GeometryProcessor
from .StabilDiagram import StabilCalc
from .Helpers import calc_xyz, nearly_equal
import itertools
from pathlib import Path

import numpy as np
import matplotlib.markers
import matplotlib.colors
import matplotlib.figure
import matplotlib.backend_bases
# from PyQt5.QtCore import pyqtSignal
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# Matplotlib
# check if python is running in headless mode i.e. as a server script
# if 'DISPLAY' in os.environ:
#     matplotlib.use("Qt5Agg", force=True)

# Numpy

# project

NoneType = type(None)


class ModeShapePlot(object):
    '''
    This class is used for displaying modal values and modeshapes obtained
    by one of the classes derived from ModalBase as part the of the pyOMA project
    (Bauhaus-Universität Weimar, Institut für Strukturmechanik).


    Drawing abilities (outdated):
        * creation of 3d plots using matplotlib's mplot3 from the
          matplotlib toolkit
        * adjusting axis limits for each of the three axis
          i.e. zoom view, shift view (along single and multiple axis')
        * change the viewport e.g. x, y, z and isometric view
        * rotating and zooming through mouse interaction is currently
          supported by matplotlib, whereas panning is not
        * animate the currently displayed deformed structure
        * save the still frame

    currently **not** supported (outdated):
        * 3D surface plots, as they are not properly supported by the
          underlying matplotlib api
        * saving of the animation as a movie file
        * drawing multiple modeshapes into one plot
        * plot modeshape in a single call from a script i.e. use static methods

    .. TODO ::
         * clean up animation methods
           * move trace objects into a separate class method
           * use class methods to draw animated modeshapes
           * only update class objects in animation
           * implement enable/disable {nodes,lines, connecting lines, trace lines, etc.)
         * remove "real modeshape" functionality as it might mislead inexperienced users
         * Fix parent-childs assignment: assemble child displacement from the 
             weighted sum of raw mode shape (channel) data of the parents to allow
             for multiple channel averaging into a single child displacement,
             afterwards transform to polar coordinates            


    '''
    # define this class's signals and the types of data they emit
    # grid_requested = pyqtSignal(str, bool)
    # beams_requested = pyqtSignal(str, bool)
    # childs_requested = pyqtSignal(str, bool)
    # chan_dofs_requested = pyqtSignal(str, bool)

    def __init__(self,
                 geometry_data,
                 stabil_calc=None,
                 modal_data=None,
                 prep_signals=None,
                 merged_data=None,
                 selected_mode=[0, 0],
                 amplitude=1,
                 real=False,
                 scale=0.2,  # 0.1*10^x [m] where x=scale
                 dpi=100,
                 nodecolor='blue',
                 nodemarker='o',
                 nodesize=20,
                 beamcolor='blue',
                 beamstyle='-',
                 linewidth=1,
                 callback_fun=None,
                 fig=None,
                 save_ani_path=None,
                 ):
        '''
        Initializes the class object and automatically checks, which of
        the below use cases have to be considered


        +----------------+--------------+-------------+--------------+
        |Variable in     |  Merging Routine                          |
        |PlotMSH         +--------------+-------------+--------------+
        |                | single-setup |poger/preger |poser merging |
        +----------------+--------------+-------------+--------------+
        |modal_freq.     | modal_data   |modal_data   |merged_data   |
        +----------------+--------------+-------------+--------------+
        |modal_damping   | modal_data   |modal_data   |merged_data   |
        +----------------+--------------+-------------+--------------+
        |modeshapes      | modal_data   |modal_data   |merged_data   |
        +----------------+--------------+-------------+--------------+
        |num_channels    | prep_signals |modal_data   |merged_data   |
        +----------------+--------------+-------------+--------------+
        |chan_dofs       | prep_signals |modal_data   |merged_data   |
        +----------------+--------------+-------------+--------------+
        |select_modes    | stabil_data  |stabil_data  |merged_data   |
        +----------------+--------------+-------------+--------------+
        |nodes           | geometry_data|geometry_data|geometry_data |
        +----------------+--------------+-------------+--------------+
        |lines           | geometry_data|geometry_data|geometry_data |
        +----------------+--------------+-------------+--------------+
        |parent-childs   | geometry_data|geometry_data|geometry_data |
        +----------------+--------------+-------------+--------------+

        Parameters
        ----------
            geometry_data : PreProcessingTools.GeometryProcessor, required
                    Object containing all the necessary geometry information.

            stabil_calc : StabilDiagram.StabilCalc, optional
                    Object containing the information, which modes were
                    selected from modal_data.

            modal_data : ModalBase.ModalBase, optional
                    Object of one the classes derived from ModalBase.ModalBase,
                    containing the estimated modal parameters at multiple
                    model orders.

            prep_signals : PreProcessingTools.PreProcessSignals, optional
                    Object containing the signals data and information
                    about it.

            merged_data : PostProcessingTools.MergePoSER, SSICovRef.PogerSSICovRef, optional
                    Object containing the merged data

            selected_mode : list, optional
                    List of [model_order, mode_index] to define the mode
                    that is displayed upon startup

            amplitude : float, optional
                    Scaling factor to scale the magnitude of mode shape displacements

            real : bool, optional
                    Whether to plot only the real part or the magnitude
                    of the complex modal coordinates

            scale : float, optional
                    Scaling factor for other elements such as arrows, etc. 
                    as a fraction of the current view limits

            dpi : float, optional
                    Resolution of the drawing canvas

            nodecolor : matplotlib color, optional
                    Color which is used to draw the nodes

            nodemarker : matplotlib marker, optional
                    Marker which is used to draw the nodes

            nodesize : float, optional
                    Marker size for the nodes

            beamcolor : matplotlib color, optional
                    Color which is used to draw the lines

            beamstyle : matplotlib linestyle, optional
                    Linestyle which is used to draw the lines

            linewidth : float, optional
                    Line width which is used to draw the lines

            callback_fun : function, optional
                    A function that is executed upon changing to a new
                    mode, allows to print mode information or change some
                    other behaviour of the class. It takes the class itself
                    and the mode index as its parameters.

            fig : matplotlib.figure.Figure, optional
                    A matplotlib figure created externally to draw
                    the mode shapes, if an external GUI is used.
        '''
        assert isinstance(geometry_data, GeometryProcessor)
        self.geometry_data = geometry_data

        if stabil_calc is not None:
            assert isinstance(stabil_calc, StabilCalc)
        self.stabil_calc = stabil_calc

        if modal_data is not None:
            assert isinstance(modal_data, ModalBase)
        self.modal_data = modal_data

        if prep_signals is not None:
            assert isinstance(prep_signals, PreProcessSignals)
        self.prep_signals = prep_signals

        if merged_data is not None:
            assert isinstance(merged_data, MergePoSER)
        self.merged_data = merged_data

        '''
        identify which merging routine has been used
        ensure all required objects were provided
        warn if unneeded objects were provided
        extract needed parameters
        '''

        if merged_data is not None:
            merging = 'PoSER'
            req_obj = {}
            nreq_obj = {'modal_data':modal_data,
                        'prep_signals':prep_signals,
                        'stabil_calc':stabil_calc}
        elif isinstance(modal_data, PogerSSICovRef):
            merging = 'PoGER'
            req_obj = {'modal_data':modal_data,
                       'stabil_calc':stabil_calc}
            nreq_obj = {'prep_signals':prep_signals,
                        'merged_data':merged_data}
        elif modal_data is not None:
            merging = 'single'  # also when used from within stabil_diagram
            req_obj = {'modal_data':modal_data,
                       'stabil_calc':stabil_calc}
            nreq_obj = {'merged_data':merged_data}
        else:  # modal_data is None and merged_data is None
            # time-history visualization, testing
            merging = None
            req_obj = {}
            nreq_obj = {'prep_signals':prep_signals,
                        'stabil_calc':stabil_calc,
                        }

        for name, obj in req_obj.items():
            if obj is None:
                raise TypeError(f'Identified merging routine: {merging} requires argument {name}, which has not been provided.')
        for name, obj in nreq_obj.items():
            if obj is not None:
                logger.info(f'Identified merging routine: {merging} will not use argument {name}.')

        if merging == 'PoSER':
            self.chan_dofs = merged_data.merged_chan_dofs
            self.num_channels = merged_data.merged_num_channels

            self.modal_frequencies = merged_data.mean_frequencies
            self.modal_damping = merged_data.mean_damping
            self.mode_shapes = merged_data.merged_mode_shapes

            self.std_frequencies = merged_data.std_frequencies
            self.std_damping = merged_data.std_damping

            self.select_modes = list(zip(range(len(self.modal_frequencies)), [
                                     0] * len(self.modal_frequencies)))

            self.setup_name = merged_data.setup_name
            self.start_time = merged_data.start_time

        elif merging == 'PoGER':
            self.chan_dofs = modal_data.merged_chan_dofs
            self.num_channels = modal_data.merged_num_channels

            self.modal_frequencies = modal_data.modal_frequencies
            self.modal_damping = modal_data.modal_damping
            self.mode_shapes = modal_data.mode_shapes

            self.select_modes = stabil_calc.select_modes

            self.setup_name = modal_data.setup_name
            self.start_time = modal_data.start_time

        elif merging == 'single':
            prep_signals = modal_data.prep_signals
            self.chan_dofs = prep_signals.chan_dofs
            self.num_channels = prep_signals.num_analised_channels

            self.modal_frequencies = modal_data.modal_frequencies
            self.modal_damping = modal_data.modal_damping
            self.mode_shapes = modal_data.mode_shapes

            if isinstance(modal_data, VarSSIRef):
                self.std_frequencies = modal_data.std_frequencies
                self.std_damping = modal_data.std_damping
            else:
                self.std_frequencies = None
                self.std_damping = None
            self.select_modes = stabil_calc.select_modes
            self.setup_name = modal_data.setup_name
            self.start_time = modal_data.start_time

        else:
            if prep_signals is not None:
                self.chan_dofs = prep_signals.chan_dofs
                self.num_channels = prep_signals.num_analised_channels
            else:
                self.chan_dofs = []
                self.num_channels = 0

            self.modal_frequencies = np.array([[]])
            self.modal_damping = np.array([[]])
            self.mode_shapes = np.array([[[]]])
            self.select_modes = []
            self.setup_name = ''
            self.start_time = None

        self.disp_nodes = {i: [0, 0, 0]
                           for i in self.geometry_data.nodes.keys()}
        self.phi_nodes = {i: [0, 0, 0]
                          for i in self.geometry_data.nodes.keys()}

        # linestyles available in matplotlib
        styles = ['-', '--', '-.', ':', 'None', ' ', '', None]

        # markerstylesavailable in matplotlib
        markers = list(matplotlib.markers.MarkerStyle.markers.keys())

        assert isinstance(real, bool)
        self.real = real

        assert isinstance(scale, (int, float))
        self.scale = scale

        assert matplotlib.colors.is_color_like(beamcolor) or isinstance(
            beamcolor, (list, tuple, np.ndarray))
        self.beamcolor = beamcolor

        assert beamstyle in styles or isinstance(
            beamstyle, (list, tuple, np.ndarray))
        self.beamstyle = beamstyle

        assert matplotlib.colors.is_color_like(nodecolor)
        self.nodecolor = nodecolor

        assert nodemarker in markers or \
            (isinstance(nodemarker, (tuple)) and len(nodemarker) == 3)
        self.nodemarker = nodemarker

        assert isinstance(nodesize, (float, int))
        self.nodesize = nodesize

        assert isinstance(dpi, int)
        self.dpi = dpi

        assert isinstance(amplitude, (int, float))
        self.amplitude = amplitude

        assert isinstance(linewidth, (int, float)) or isinstance(
            linewidth, (list, tuple, np.ndarray))
        self.linewidth = linewidth

        if callback_fun is not None:
            assert callable(callback_fun)
        self.callback_fun = callback_fun

        # bool objects
        self.show_nodes = True
        self.show_lines = True
        self.show_nd_lines = True
        self.show_cn_lines = True
        self.show_parent_childs = True
        self.show_chan_dofs = True
        self.show_axis = True
        self.animated = False
        self.data_animated = False
        # self.draw_trace = True
        if save_ani_path is not None:
            assert isinstance(save_ani_path, Path)
        self.save_ani_path = save_ani_path

        # plot objects
        self.patches_objects = {}
        self.lines_objects = []
        self.nd_lines_objects = []
        self.cn_lines_objects = {}
        self.arrows_objects = []
        self.channels_objects = []
        self.trace_objects = []
        self.axis_obj = {}
        self.seq_num = 0

        if fig is None:
            fig = matplotlib.figure.Figure(
                 dpi=dpi, facecolor='#ffffff00')

            # remove all whitespace around the axes
            fig.subplots_adjust(0, 0, 1, 1, 0, 0)
            # fig.set_tight_layout(True)
            # self.canvas =
            matplotlib.backend_bases.FigureCanvasBase(fig)
        else:
            assert isinstance(fig, matplotlib.figure.Figure)
            # self.canvas = fig.canvas

        self.fig = fig

        # Add another subplot below of the 3D subplot, to be able to set
        # the clip path on all lines, etc. to a patch, that extends over
        # the whole figure -> PlotMSHGUI.resizeEvent_
        # ax2d = self.fig.add_subplot(111, fc='#ffffff00')
        # ax2d.patch.set_edgecolor('#ffffff00')

        # the 3D axes must be added manually, becaus add_subplot would
        # remove the other axes at the same position
        # self.subplot = mpl_toolkits.mplot3d.axes3d.Axes3D(fig,(0,0,1,1), anchor='C', fc='#ffffaabb')
        self.subplot = fig.subplots(subplot_kw=dict(projection='3d', anchor='C', fc='#ffffff00', box_aspect=(1, 1, 1)))
        # self.fig.add_axes(self.subplot)
        # self.subplot.patch.set_edgecolor('#ffffff00')
        # mpl_toolkits.mplot3d.axes3d.Axes3D.draw = draw_axes
        # self.subplot.set_box_aspect((1,1,1))
        self.subplot.set_aspect('equal', 'datalim')
        # nasty hack to disable clipping
        self.subplot.patch = fig.patch
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)

        self.subplot.grid(False)
        self.subplot.set_axis_off()
        if not self.select_modes:
            self.mode_index = None
        else:
            self.mode_index = self.select_modes[0]

        # instantiate the x,y,z axis arrows
        # self.draw_axis()

    # @pyqtSlot()
    def reset_view(self):
        '''
         * restore viewport
         * restore axis' limits
         * reset displacements values for all nodes
        '''
        self.stop_ani()
        # mpl_toolkits.mplot3d.axes3d.proj3d.persp_transformation = persp_transformation
        self.subplot.view_init(30, -60)
        self.subplot.autoscale_view()
        xmin, xmax, ymin, ymax, zmin, zmax = None, None, None, None, None, None
        for node in self.geometry_data.nodes.values():
            if xmin is None:
                xmin = node[0]
            if xmax is None:
                xmax = node[0]
            if ymin is None:
                ymin = node[1]
            if ymax is None:
                ymax = node[1]
            if zmin is None:
                zmin = node[2]
            if zmax is None:
                zmax = node[2]
            xmin = min(node[0], xmin)
            xmax = max(node[0], xmax)
            ymin = min(node[1], ymin)
            ymax = max(node[1], ymax)
            zmin = min(node[2], zmin)
            zmax = max(node[2], zmax)

        xrang = xmax - xmin
        xmed = xmax - xrang / 2
        yrang = ymax - ymin
        ymed = ymax - yrang / 2
        zrang = zmax - zmin
        zmed = zmax - zrang / 2

        rang = max(xrang, yrang, zrang)

        xmin, xmax = xmed - rang / 2, xmed + rang / 2
        ymin, ymax = ymed - rang / 2, ymed + rang / 2
        zmin, zmax = zmed - rang / 2, zmed + rang / 2
        # if xmin!=xmax:
        self.subplot.set_xlim3d(xmin, xmax)
        # if ymin!=ymax:
        self.subplot.set_ylim3d(ymin, ymax)
        # if zmin!=zmax:
        self.subplot.set_zlim3d(zmin, zmax)
        # print(xmin, xmax, ymin, ymax, zmin, zmax)
        self.draw_nodes()
        self.draw_lines()
        self.draw_chan_dofs()
        self.draw_parent_childs()
        self.draw_axis()
        if self.mode_index is not None:
            self.draw_msh()
        self.set_equal_aspect()
        # self.disp_nodes = { i : [0,0,0] for i in self.geometry_data.nodes.keys() }

        self.fig.canvas.draw()

    # @pyqtSlot()

    def change_viewport(self, viewport=None):
        '''
         Change the viewport e.g. azimuth and elevation and refresh the canvas

         Parameters
         ----------
             viewport: {'X', 'Y', 'Z', 'ISO'\\, optional
                 The viewport to set.
        '''
        roll = None
        if viewport == 'X':
            azim, elev = 0, 0
            # mpl_toolkits.mplot3d.axes3d.proj3d.persp_transformation = orthogonal_proj
            self.subplot.set_proj_type('ortho')
        elif viewport == 'Y':
            azim, elev = -90, 0
            # mpl_toolkits.mplot3d.axes3d.proj3d.persp_transformation = orthogonal_proj
            self.subplot.set_proj_type('ortho')
        elif viewport == 'Z':
            azim, elev = 0, 90
            # mpl_toolkits.mplot3d.axes3d.proj3d.persp_transformation = orthogonal_proj
            self.subplot.set_proj_type('ortho')
        elif viewport == 'ISO':
            azim, elev = -60, 30
            # mpl_toolkits.mplot3d.axes3d.proj3d.persp_transformation = persp_transformation
            self.subplot.set_proj_type('persp')
        elif isinstance(viewport, (list, tuple)):
            elev, azim, roll = viewport
        else:
            logger.warning(f'viewport not recognized: {viewport}')
            azim, elev = -60, 30
            # mpl_toolkits.mplot3d.axes3d.proj3d.persp_transformation = persp_transformation
            self.subplot.set_proj_type('persp')
        self.subplot.view_init(elev, azim, roll)
        self.fig.canvas.draw()

        if self.animated or self.data_animated:
            for line in self.lines_objects:
                line.set_visible(False)
            for line in self.nd_lines_objects:
                line.set_visible(False)
            for line in self.cn_lines_objects.values():
                line.set_visible(False)
            self.line_ani._setup_blit()

    # @pyqtSlot(str)
    def change_mode(self, frequency=None, index=None, mode_index=None,):
        '''
        If the user selects a new mode: plots the mode shape
        and returns modal values e.g. to a GUI caller.

        Parameters
        ----------
            frequency: float,optional
                A search for the closest frequency in the list of already
                selected indices (self.selected_indices) is performed
            index: integer, optional
                Alternatively, the index of the wanted mode can be directly given
            mode_index: integer, optional
                The number of the mode in the list of currently selected modes

        Returns
        -------
            order_index: integer
                Model order of the selected mode
            mode_index: integer
                Index of the selected mode at model order
            frequency: float
                natural frequency of the selected mode
            damping: float
                damping ratio of the selected mode
            MPC: float, optional
                Modal phase colinearity of the selected mode,
                if available from an instance of StabilDiagram.StabilCalc1
            MP: float, optional
                Mean phase of the selected mode,
                if available from an instance of StabilDiagram.StabilCalc1
            MPD: float, optional
                Mean phase deviation of the selected mode,
                if available from an instance of StabilDiagram.StabilCalc1

        '''
        # mode numbering starts at 1 python lists start at 0
        selected_indices = self.select_modes
        if frequency is not None:
            frequencies = np.array([self.modal_frequencies[index[0], index[1]]
                                    for index in selected_indices])
            f_delta = abs(frequencies - frequency)
            index = np.argmin(f_delta)

        if index is not None:
            mode_index = selected_indices[index]
        if mode_index is None:
            raise RuntimeError('No arguments provided!')
        # print(logger.info(ndex)
        frequency = self.modal_frequencies[mode_index[0], mode_index[1]]
        damping = self.modal_damping[mode_index[0], mode_index[1]]
        if self.stabil_calc:
            MPC = self.stabil_calc.MPC_matrix[mode_index[0], mode_index[1]]
            MP = self.stabil_calc.MP_matrix[mode_index[0], mode_index[1]]
            MPD = self.stabil_calc.MPD_matrix[mode_index[0], mode_index[1]]
        else:
            MPC = None
            MP = None
            MPD = None
        self.mode_index = mode_index

        if self.save_ani_path:
            cwd = self.save_ani_path / f'{self.select_modes.index(self.mode_index)}/'
            if not os.path.exists(cwd):
                os.makedirs(cwd)

        self.draw_msh()

        # print('self.callback_fun', self.callback_fun)
        if self.callback_fun is not None:
            # print('call')
            try:
                self.callback_fun(self, mode_index)
            except Exception as e:
                logger.warning(repr(e))

        # order, mode_num,....
        return mode_index[1], mode_index[0], frequency, damping, MPC, MP, MPD

    def get_frequencies(self):
        '''
        Returns
        -------
            frequencies: list
                Identified frequencies of all currently selected modes.
        '''
        selected_indices = self.select_modes

        frequencies = sorted([self.modal_frequencies[index[0], index[1]]
                              for index in selected_indices])
        return frequencies

    # @pyqtSlot()
    # @pyqtSlot(float)
    def change_amplitude(self, amplitude=None):
        '''
        Changes the amplitude of the mode shape, and redraws the
        modeshapes based on this amplitude.

        Parameters
        ----------
            amplitude: float, optional
        '''
        if amplitude is None:
            return
        amplitude = float(amplitude)
        if amplitude == self.amplitude:
            return

        self.amplitude = amplitude

        if self.mode_shapes.shape[2]:
            self.draw_msh()

    # @pyqtSlot(bool)
    def change_part(self, b):
        '''
        Change, which part of the complex number modeshapes should be
        drawn and redraw the modeshapes

        Parameters
        ----------
            b: bool
                If b, draws the magnitude of the modal coordinated, else
                phase information is considered. Default: b = False

        '''
        if b == self.real:
            return

        self.real = b
        self.draw_msh()

    def save_plot(self, path=None):
        '''
        Save the curently displayed frame as a graphics file

        Parameters
        ----------
            path: str (valid filepath), optional
                The full path, including the extension, where to save
                the graphic.
        '''

        if path:
            self.fig.canvas.print_figure(path, dpi=self.dpi)

    # @pyqtSlot(float, float, float, int)

    def add_node(self, x, y, z, i):
        '''
        Adds a node to the internal node table and initializes zero-value
        displacements for this node to the internal displacements table.
        Draws a single point at the coordinates and annotates it with
        its number. Stores the two plot objects in a table and removes
        any objects that might be in the table at the desired place
        to avoid duplicate nodes.

        Parameters
        ----------
            x,y,z: float
                3D-coordinates of the node
            i: integer
                Index of the node, must be previously determined
        '''
        # leave present value if there is any else put 0
        self.disp_nodes[i] = self.disp_nodes.get(i, [0, 0, 0])

        x, y, z = x + self.disp_nodes[i][0], y + self.disp_nodes[i][1], z + \
            self.disp_nodes[i][2]  # draw displaced nodes

        patch = self.subplot.scatter(
            x,
            y,
            z,
            color=self.nodecolor,
            marker=self.nodemarker,
            s=self.nodesize,
            visible=self.show_nodes)

        text = self.subplot.text(x, y, z, i, visible=self.show_nodes)

        if self.patches_objects.get(i) is not None:
            if isinstance(self.patches_objects[i], (tuple, list)):
                for obj in self.patches_objects[i]:
                    try:
                        obj.remove()
                    except BaseException:
                        pass

        self.patches_objects[i] = (patch, text)

        self.fig.canvas.draw_idle()

    # @pyqtSlot(tuple, int)
    def add_line(self, line, i):
        '''
        Add a line by adding the start node and end node to the internal
        line table and draws that line between the two nodes. Stores the
        line object in a table and removes any objects that might be in
        the table at the desired place, i.e. avoid duplicate lines

        Parameters
        ----------
            line: 2-tuple of integer
                The indices of the start- and end-node of the line
            i: integer
                Index of the line, must be previously determined

        '''
        if isinstance(self.beamcolor, (list, tuple, np.ndarray)):
            beamcolor = self.beamcolor[i]
        else:
            beamcolor = self.beamcolor
        if isinstance(self.beamstyle, (list, tuple, np.ndarray)):
            beamstyle = self.beamstyle[i]
        else:
            beamstyle = self.beamstyle
        if isinstance(self.linewidth, (list, tuple, np.ndarray)):
            linewidth = self.linewidth[i]
        else:
            linewidth = self.linewidth

        line_object = self.subplot.plot(
            [self.geometry_data.nodes[node][0]
             +self.disp_nodes[node][0] for node in line],
            [self.geometry_data.nodes[node][1]
             +self.disp_nodes[node][1] for node in line],
            [self.geometry_data.nodes[node][2]
             +self.disp_nodes[node][2] for node in line],
            color=beamcolor,
            linestyle=beamstyle,
            visible=self.show_lines,
            linewidth=linewidth)[0]

        while len(self.lines_objects) < i + 1:
            self.lines_objects.append(None)
        if self.lines_objects[i] is not None:
            try:
                self.lines_objects[i].remove()
            except ValueError:
                pass
                # del self.lines_objects[i]
        self.lines_objects[i] = line_object

        self.fig.canvas.draw_idle()

    # @pyqtSlot(tuple, int)
    def add_nd_line(self, line, i):
        '''
        Add a non-displaced line, which acts as a mesh-reference for the
        displaced lines. Works analogously to self.add_line

        Parameters
        ----------
            line: 2-tuple of integer
                The indices of the start- and end-node of the line
            i: integer
                Index of the line, must be previously determined

        '''
        if isinstance(self.beamcolor, (list, tuple, np.ndarray)):
            beamcolor = self.beamcolor[i]
        else:
            beamcolor = self.beamcolor
        if isinstance(self.linewidth, (list, tuple, np.ndarray)):
            linewidth = self.linewidth[i]
        else:
            linewidth = self.linewidth

        beamstyle = 'dotted'

        line_object = self.subplot.plot(
            [self.geometry_data.nodes[node][0] for node in line],
            [self.geometry_data.nodes[node][1] for node in line],
            [self.geometry_data.nodes[node][2] for node in line],
            color=beamcolor,
            linestyle=beamstyle,
            linewidth=1,
            visible=self.show_lines)[0]

        while len(self.nd_lines_objects) < i + 1:
            self.nd_lines_objects.append(None)
        if self.nd_lines_objects[i] is not None:
            try:
                self.nd_lines_objects[i].remove()
            except ValueError:
                pass
                # del self.nd_lines_objects[i]
        self.nd_lines_objects[i] = line_object

        self.fig.canvas.draw_idle()

    # @pyqtSlot(tuple, int)
    def add_cn_line(self, i):
        '''
        Draws a line between the displaced and the undisplaced node.

        Parameters
        ----------
            i: integer
                Index of the node
        '''

        beamcolor = 'lightgray'

        beamstyle = 'dotted'
        node = self.geometry_data.nodes[i]
        disp_node = self.disp_nodes.get(node, [0, 0, 0])

        line_object = self.subplot.plot(
            [node[0], node[0] + disp_node[0]],
            [node[1], node[1] + disp_node[1]],
            [node[2], node[2] + disp_node[2]],
            color=beamcolor,
            linestyle=beamstyle,
            linewidth=1,
            visible=self.show_cn_lines)[0]

        if self.cn_lines_objects.get(i, None) is not None:
            try:
                self.cn_lines_objects[i].remove()
            except ValueError:
                pass
        self.cn_lines_objects[i] = line_object

        self.fig.canvas.draw_idle()

    # @pyqtSlot(int, float, float, float, int, float, float, float, int)

    def add_parent_child(self, i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl, i):
        '''
        Takes parent-child definitions and adds these definitions to the
        internal parent-child table. Draws an arrow indicating the DOF
        at each node of parent and child. Arrows at equal positions and
        direction will be offset to avoid overlapping. Stores the two
        arrow objects in a table and removes any objects that might be
        in the table at the desired index i.e. avoid duplicate arrows

        Parameters
        ----------
            i_m: integer
                Index of the parent node
            x_m,y_m,z_m: float
                Scale factor for each parent DOF.
            i_sl: integer
                Index of the child node
            x_sl,y_sl,z_sl: float
                Scale factor for each child DOF.
        '''

        def offset_arrows(verts3d_new, all_arrows_list):
            '''
            avoid overlapping arrows as they are hard to distinguish
            therefore loop through all arrow object and compare their
            coordinates and directions (but ignore length) with the
            arrow to be newly created if there is an overlapping then
            offset the coordinates of the new arrow by 5 % of the
            length (hardcoded) in each direction (which should actually
            only be in the perpendicular plane)
            '''
            ((x_s, x_e), (y_s, y_e), (z_s, z_e)) = verts3d_new
            start_point = (x_s, y_s, z_s)
            length = x_e ** 2 + y_e ** 2 + z_e ** 2
            dir_norm = (x_e / length, y_e / length, z_e / length)
            while True:
                for arrow in itertools.chain.from_iterable(all_arrows_list):
                    (x, y, z, dx, dy, dz) = arrow._verts3d
                    (x_a, x_b) = x, x + dx
                    (y_a, y_b) = y, y + dy
                    (z_a, z_b) = z, z + dz
                    # (x_a, x_b), (y_a, y_b), (z_a, z_b) = arrow._verts3d
                    # transform from position vector to direction vector
                    x_c, y_c, z_c = (x_b - x_a), (y_b - y_a), (z_b - z_a)
                    this_start_point = (x_a, y_a, z_b)
                    this_length = x_c ** 2 + y_c ** 2 + z_c ** 2
                    if this_length == 0:
                        continue
                    this_dir_norm = (
                        x_c / this_length,
                        y_c / this_length,
                        z_c / this_length)
                    if start_point != this_start_point:  # starting point equal
                        continue
                    if this_dir_norm != dir_norm:  # direction equal
                        continue
                    # offset hardcoded
                    x_s, y_s, z_s = [
                        coord + 0.05 * this_length for coord in start_point]
                    # lazy offset, it should actually be in the plane
                    # perpendicular to the vector
                    start_point = (x_s, y_s, z_s)
                    length = x_e ** 2 + y_e ** 2 + z_e ** 2
                    dir_norm = (x_e / length, y_e / length, z_e / length)
                    break
                else:
                    break
            return ((x_s, x_e), (y_s, y_e), (z_s, z_e))

        color = "bgrcmyk"[int(np.fmod(i, 7))]  # equal colors for both arrows

        x_s, y_s, z_s = self.geometry_data.nodes[i_m]
        ((x_s, x_m), (y_s, y_m), (z_s, z_m)) = offset_arrows(
            ((x_s, x_m), (y_s, y_m), (z_s, z_m)), self.arrows_objects)

        # point the arrow towards the resulting direction
        arrow_m = LabeledArrow3D(x_s, y_s, z_s, x_m, y_m, z_m,
                                 mutation_scale=5, lw=1, arrowstyle="-|>",
                                 color=color, visible=self.show_parent_childs)
        arrow_m = self.subplot.add_artist(arrow_m)

        x_s, y_s, z_s = self.geometry_data.nodes[i_sl]
        ((x_s, x_sl), (y_s, y_sl), (z_s, z_sl)) = offset_arrows(
            ((x_s, x_sl), (y_s, y_sl), (z_s, z_sl)), self.arrows_objects)

        # point the arrow towards the resulting direction
        arrow_sl = LabeledArrow3D(x_s, y_s, z_s, x_sl, y_sl, z_sl,
                                  mutation_scale=5, lw=1, arrowstyle="-|>",
                                  color=color, visible=self.show_parent_childs)

        arrow_sl = self.subplot.add_artist(arrow_sl)

        while len(self.arrows_objects) < i + 1:
            self.arrows_objects.append(None)
        if self.arrows_objects[i] is not None:
            for obj in self.arrows_objects[i]:
                obj.remove()
        self.arrows_objects[i] = (arrow_m, arrow_sl)

        self.fig.canvas.draw_idle()

    # @pyqtSlot(int, int, tuple, int)

    def add_chan_dof(self, chan, node, az, elev, chan_name, i):
        '''
        Draws an arrow indicating a channel-DOF assignment. Annotates the
        arrow with the the channel name. Stores the two plot objects in a
        table and removes any objects that might be in the table at the
        desired index i.e. avoid duplicate arrows/texts.

        Parameters
        ----------
            chan: integer
                Index of the channel.
            node: integer
                Index of the node in the internal node table
            az, elev: float
                Azimuth and elevation of the DOF assignment
            chan_name: str
                Name of the channel to annotate
            i: integer
                Table index for the plot objects.

        .. TODO::
            * arrow lengths do not scale with the total dimension of the plot
        '''

        x_s, y_s, z_s = self.geometry_data.nodes[node]

        x_m, y_m, z_m = calc_xyz(
            az / 180 * np.pi, elev / 180 * np.pi, r=self.scale)

        # point the arrow towards the resulting direction
        arrow = LabeledArrow3D(x_s, y_s, z_s, x_m, y_m, z_m,
                               mutation_scale=5, lw=1, arrowstyle="-|>",
                               visible=self.show_chan_dofs)
        arrow = self.subplot.add_artist(arrow)

        arrow.add_label(chan_name, visible=self.show_chan_dofs)
        arrow.set_clip_path(None)

        while len(self.channels_objects) < i + 1:
            self.channels_objects.append(None)
        if self.channels_objects[i] is not None:
            self.channels_objects[i].remove()

        self.channels_objects[i] = arrow

        self.fig.canvas.draw_idle()

    # @pyqtSlot(float, float, float, int)
    def take_node(self, x, y, z, node):
        '''
        Remove a node at given coordinates and all objects connected to
        this node first (there should not be any). Remove the patch
        objects from the plot and remove the coordinates from the node
        and displacement tables.

        Parameters
        ----------
            x,y,z: float
                Coordinates of the node
            node: integer
                Index of the node

        .. TODO::
            * Function presumably breaks in the second for loop, because
              geometry_data and the internal tables become out of sync.

        '''

        d_x, d_y, d_z = self.disp_nodes.get(node, [0, 0, 0])
        d_x, d_y, d_z = abs(d_x), abs(d_y), abs(d_z)

        for j in [node] + list(range(max(len(self.patches_objects), node))):
            if self.patches_objects.get(j) is None:
                continue
            # ._offsets3d = ([x],[y],np.ndarray([z]))
            x_, y_, z_ = [float(val[0])
                          for val in self.patches_objects[j][0]._offsets3d]
            if x - d_x <= x_ <= x + d_x and \
                    y - d_y <= y_ <= y + d_y and \
                    z - d_z <= z_ <= z + d_z:
                for obj in self.patches_objects[j]:
                    obj.remove()
                del self.patches_objects[j]
                break
        else:  # executed when for loop runs through
            if self.patches_objects:
                logging.warning('patches_object not found')

        for j in [node] + \
                list(range(max(len(self.geometry_data.nodes), node))):

            if self.geometry_data.nodes.get(j) == [x, y, z]:
                del self.disp_nodes[j]
                break
        else:  # executed when for loop runs through
            if self.patches_objects:
                logging.warning('node not found')

        self.fig.canvas.draw_idle()

    # @pyqtSlot(tuple)
    def take_line(self, line):
        '''
        Remove a line between to nodes. If the plot objects are already
        in their displaced state, the comparison between the actual
        coordinates and these objects have to account for  displacement
        by comparing to an interval of coordinates. Remove the non-displaced
        lines, too.

        Parameters
        ----------
            line: 2-tuple of integers
                Tuple containg the indices of the start- and end-nodes

        '''
        assert isinstance(line, (tuple, list))
        assert len(line) == 2

        node_s, node_e = self.geometry_data.nodes[line[0]
                                                  ], self.geometry_data.nodes[line[1]]
        x_s, y_s, z_s = node_s
        x_e, y_e, z_e = node_e

        d_node_s = self.disp_nodes.get(line[0], [0, 0, 0])
        d_node_e = self.disp_nodes.get(line[1], [0, 0, 0])

        d_x_s, d_y_s, d_z_s = abs(
            d_node_s[0]), abs(
            d_node_s[1]), abs(
            d_node_s[2])
        d_x_e, d_y_e, d_z_e = abs(
            d_node_e[0]), abs(
            d_node_e[1]), abs(
            d_node_e[2])

        for j in range(len(self.lines_objects)):
            (x_s_, x_e_), (y_s_, y_e_), (z_s_, z_e_) = self.lines_objects[
                j]._verts3d
            if x_s - d_x_s <= x_s_ <= x_s + d_x_s and \
                    x_e - d_x_e <= x_e_ <= x_e + d_x_e and \
                    y_s - d_y_s <= y_s_ <= y_s + d_y_s and \
                    y_e - d_y_e <= y_e_ <= y_e + d_y_e and \
                    z_s - d_z_s <= z_s_ <= z_s + d_z_s and \
                    z_e - d_z_e <= z_e_ <= z_e + d_z_e:  # account for displaced lines

                self.lines_objects[j].remove()
                del self.lines_objects[j]
                break
            elif x_s - d_x_s <= x_e_ <= x_s + d_x_s and \
                    x_e - d_x_e <= x_s_ <= x_e + d_x_e and \
                    y_s - d_y_s <= y_e_ <= y_s + d_y_s and \
                    y_e - d_y_e <= y_s_ <= y_e + d_y_e and \
                    z_s - d_z_s <= z_e_ <= z_s + d_z_s and \
                    z_e - d_z_e <= z_s_ <= z_e + d_z_e:  # account for inverted lines

                self.lines_objects[j].remove()
                del self.lines_objects[j]
                break
        else:
            if self.lines_objects:
                logging.warning('line_object not found')

        for j in range(len(self.nd_lines_objects)):
            (x_s_, x_e_), (y_s_, y_e_), (z_s_, z_e_) = self.nd_lines_objects[
                j]._verts3d
            if x_s - d_x_s <= x_s_ <= x_s + d_x_s and \
                    x_e - d_x_e <= x_e_ <= x_e + d_x_e and \
                    y_s - d_y_s <= y_s_ <= y_s + d_y_s and \
                    y_e - d_y_e <= y_e_ <= y_e + d_y_e and \
                    z_s - d_z_s <= z_s_ <= z_s + d_z_s and \
                    z_e - d_z_e <= z_e_ <= z_e + d_z_e:  # account for displaced lines

                self.nd_lines_objects[j].remove()
                del self.nd_lines_objects[j]
                break
            elif x_s - d_x_s <= x_e_ <= x_s + d_x_s and \
                    x_e - d_x_e <= x_s_ <= x_e + d_x_e and \
                    y_s - d_y_s <= y_e_ <= y_s + d_y_s and \
                    y_e - d_y_e <= y_s_ <= y_e + d_y_e and \
                    z_s - d_z_s <= z_e_ <= z_s + d_z_s and \
                    z_e - d_z_e <= z_s_ <= z_e + d_z_e:  # account for inverted lines

                self.nd_lines_objects[j].remove()
                del self.nd_lines_objects[j]
                break
        else:
            if self.nd_lines_objects:
                logging.warning('line_object not found')
        self.fig.canvas.draw_idle()

    # @pyqtSlot(int, float, float, float, int, float, float, float)
    def take_parent_child(self, i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl):
        '''
        Remove the two arrows associated with the parent-child definition.

        Parameters
        ----------
            i_m: integer
                Index of the parent node
            x_m,y_m,z_m: float
                Scale factor for each parent DOF.
            i_sl: integer
                Index of the child node
            x_sl,y_sl,z_sl: float
                Scale factor for each child DOF.
        '''
        arrow_m = (i_m, x_m, y_m, z_m)
        arrow_sl = (i_sl, x_sl, y_sl, z_sl)

        node_m = arrow_m[0]
        x_s_m, y_s_m, z_s_m = self.geometry_data.nodes[node_m]
        x_e_m, y_e_m, z_e_m = arrow_m[1:4]
        length_m = x_e_m ** 2 + y_e_m ** 2 + z_e_m ** 2

        node_sl = arrow_sl[0]
        x_s_sl, y_s_sl, z_s_sl = self.geometry_data.nodes[node_sl]

        x_e_sl, y_e_sl, z_e_sl = arrow_sl[1:4]
        length_sl = x_e_sl ** 2 + y_e_sl ** 2 + z_e_sl ** 2

        for j in range(len(self.arrows_objects)):
            arrow_found = [False, False]
            for arrow in self.arrows_objects[j]:
                (x_s, x_e), (y_s, y_e), (z_s, z_e) = arrow._verts3d
                # transform from position vector to direction vector
                x_e, y_e, z_e = (x_e - x_s), (y_e - y_s), (z_e - z_s)

                # check positions with offsets and directions
                if (x_s - 0.05 * length_m <= x_s_m <= x_s + 0.05 * length_m and
                    y_s - 0.05 * length_m <= y_s_m <= y_s + 0.05 * length_m and
                    z_s - 0.05 * length_m <= z_s_m <= z_s + 0.05 * length_m and
                    x_e == x_e_m and
                    y_e == y_e_m and
                        z_e == z_e_m):
                    arrow_found[0] = True

                if (x_s -
                    0.05 *
                    length_sl <= x_s_sl <= x_s +
                    0.05 *
                    length_sl and y_s -
                    0.05 *
                    length_sl <= y_s_sl <= y_s +
                    0.05 *
                    length_sl and z_s -
                    0.05 *
                    length_sl <= z_s_sl <= z_s +
                    0.05 *
                        length_sl and x_e == x_e_sl and y_e == y_e_sl and z_e == z_e_sl):
                    arrow_found[1] = True

            # ie found the right parent child pair
            if arrow_found[0] and arrow_found[1]:
                # remove both parent child arrows
                for arrow in self.arrows_objects[j]:
                    arrow.remove()
                del self.arrows_objects[j]
                # restart the first for loop i.e. start j at 0 again
                break
            else:
                continue
        else:
            if self.arrows_objects:
                logging.warning('arrows_object not found')

        self.fig.canvas.draw_idle()

    # @pyqtSlot(int, int, tuple, int)
    def take_chan_dof(self, chan, node, dof):
        '''
        Remove the arrow and text objects associated with the channel -
        DOF assignment.

        Parameters
        ----------
            chan: integer
                Index of the channel.
            node: integer
                Index of the node in the internal node table
            dof: 3-tuple {az,elev,chan_name}
                az, elev: float
                    Azimuth and elevation of the DOF assignment
                chan_name: str
                    Name of the channel to annotate

        '''
        assert isinstance(node, int)
        assert isinstance(dof, (tuple, list))
        assert len(dof) == 3

        x_s, y_s, z_s = self.geometry_data.nodes[node]
        x_e, y_e, z_e = dof[0] + x_s, dof[1] + y_s, dof[2] + z_s

        for j in range(len(self.channels_objects)):
            (x_s_, x_e_), (y_s_, y_e_), (z_s_, z_e_) = \
                self.channels_objects[j][0]._verts3d
            if nearly_equal(x_s_, x_s, 2) and nearly_equal(x_e_, x_e, 2) and \
               nearly_equal(y_s_, y_s, 2) and nearly_equal(y_e_, y_e, 2) and \
               nearly_equal(z_s_, z_s, 2) and nearly_equal(z_e_, z_e, 2):
                for obj in self.channels_objects[j]:
                    obj.remove()
                del self.channels_objects[j]
                break
        else:
            if self.channels_objects:
                logging.warning('chandof_object not found')

        self.fig.canvas.draw_idle()

    def draw_axis(self):
        '''
        Draw the axis arrows. Length is based on the current data limits.
        Removes the current arrows if the exist.
        '''

        for axis in ['X', 'Y', 'Z']:
            if axis in self.axis_obj:
                try:
                    self.axis_obj[axis].remove()
                    del self.axis_obj[axis]
                except ValueError:
                    continue

        axis = self.subplot.add_artist(
            LabeledArrow3D(0, 0, 0, self.scale, 0, 0,
                           mutation_scale=20, lw=1, arrowstyle="-|>",
                           color="r", visible=self.show_axis))
        axis.add_label('X', color='r', visible=self.show_axis)

#         text = self.subplot.text(
#             self.scale * 1.1,
#             0,
#             0,
#             'X',
#             zdir=None,
#             color='r',
#             visible=self.show_axis)
        self.axis_obj['X'] = axis

        axis = self.subplot.add_artist(
            LabeledArrow3D(0, 0, 0, 0, self.scale, 0,
                           mutation_scale=20, lw=1, arrowstyle="-|>",
                           color="g", visible=self.show_axis))
        axis.add_label('Y', color='g', visible=self.show_axis)

#         text = self.subplot.text(
#             0,
#             self.scale * 1.1,
#             0,
#             'Y',
#             zdir=None,
#             color='g',
#             visible=self.show_axis)
        self.axis_obj['Y'] = axis

        axis = self.subplot.add_artist(
            LabeledArrow3D(0, 0, 0, 0, 0, self.scale,
                           mutation_scale=20, lw=1, arrowstyle="-|>",
                           color="b", visible=self.show_axis))
        axis.add_label('Z', color='b', visible=self.show_axis)

#         text = self.subplot.text(
#             0,
#             0,
#             self.scale * 1.1,
#             'Z',
#             zdir=None,
#             color='b',
#             visible=self.show_axis)
        self.axis_obj['Z'] = axis

        self.fig.canvas.draw_idle()

    def refresh_axis(self, visible=None):
        '''
        Refresh the axis arrows and make them visible/invisible, e.g.
        after programmatically changing visibility flags.

        Parameters
        ----------
            visible: bool, ooptional
                Visibility flag for the axis arrows

        '''
        visible = bool(visible)

        if visible is not None:
            self.show_axis = visible

        for axis in self.axis_obj.values():
            axis.set_visible(self.show_axis)
        self.fig.canvas.draw()

    # @pyqtSlot()
    def draw_nodes(self):
        ''''
        Draws nodes from the node list of PreProcessingTools.GeometryData
        The currently stored displacement values are used for moving the
        nodes.
        '''
        for key, node in self.geometry_data.nodes.items():
            self.add_node(*node, i=key)

    def refresh_nodes(self, visible=None):
        '''
        Refresh the nodes and make them visible/invisible, e.g.
        after programmatically changing visibility flags.

        Parameters
        ----------
            visible: bool, ooptional
                Visibility flag for the nodes

        '''

        if visible is not None:
            visible = bool(visible)
            self.show_nodes = visible

        for key in self.geometry_data.nodes.keys():
            node = self.geometry_data.nodes[key]
            disp_node = self.disp_nodes.get(key, [0, 0, 0])
            phase_node = self.phi_nodes.get(key, [0, 0, 0])
            patch = self.patches_objects.get(key, None)
            if isinstance(patch, (tuple, list)):
                for obj in patch:
                    obj.set_visible(self.show_nodes)
                x = node[0] + disp_node[0] * \
                    np.cos(self.seq_num / 25 * 2 * np.pi + phase_node[0])
                y = node[1] + disp_node[1] * \
                    np.cos(self.seq_num / 25 * 2 * np.pi + phase_node[1])
                z = node[2] + disp_node[2] * \
                    np.cos(self.seq_num / 25 * 2 * np.pi + phase_node[2])
                # print('in refresh nodes', x,y,z)
                # if 'PIV' in key:
                #    print(key, disp_node, phase_node)

                patch[0].set_offsets([x, y])
                patch[0].set_3d_properties(z, 'z')

                patch[1].set_position([x, y])
                patch[1].set_3d_properties(z, None)

        self.fig.canvas.draw_idle()

    def draw_lines(self):
        '''
        Draws all line from the line list of PreProcessingTools.GeometryProcessor
        The currently stored displacement values are used for moving the
        nodes.
        '''
        for i, line in enumerate(self.geometry_data.lines):
            self.add_line(line, i)
            self.add_nd_line(line, i)
            self.refresh_lines()
            self.refresh_nd_lines()

        # self.lines_objects[-1].remove()
        # del self.lines_objects[-1]

        # node = line[0]
        # self.lines_objects.append(
        #     self.subplot.plot(
        #     [self.geometry_data.nodes[node][0]
        #      + self.disp_nodes[node][0]],
        #     [self.geometry_data.nodes[node][1]
        #      + self.disp_nodes[node][1]],
        #     [self.geometry_data.nodes[node][2]
        #      + self.disp_nodes[node][2] ],
        #     color=self.beamcolor,
        #     marker='o', markersize=6,
        #     visible=self.show_lines,)[0])

        for i in self.geometry_data.nodes.keys():
            self.add_cn_line(i)

    def refresh_lines(self, visible=None):
        '''
        Refresh the lines and make them visible/invisible, e.g.
        after programmatically changing visibility flags.

        Parameters
        ----------
            visible: bool, ooptional
                Visibility flag for the lines

        '''

        if visible is not None:
            visible = bool(visible)
            self.show_lines = visible

        for line, line_node in zip(
                self.lines_objects, self.geometry_data.lines):
            x = [self.geometry_data.nodes[node][0] + self.disp_nodes[node][0]
                 * np.cos(self.seq_num / 25 * 2 * np.pi + self.phi_nodes[node][0])
                 for node in line_node]
            y = [self.geometry_data.nodes[node][1] + self.disp_nodes[node][1]
                 * np.cos(self.seq_num / 25 * 2 * np.pi + self.phi_nodes[node][1])
                 for node in line_node]
            z = [self.geometry_data.nodes[node][2] + self.disp_nodes[node][2]
                 * np.cos(self.seq_num / 25 * 2 * np.pi + self.phi_nodes[node][2])
                 for node in line_node]
            line.set_visible(self.show_lines)
            line.set_data_3d([x, y, z])
            # line.set_3d_properties(z)

        for key in self.geometry_data.nodes.keys():
            node = self.geometry_data.nodes[key]
            disp_node = self.disp_nodes.get(key, [0, 0, 0])
            phi_node = self.phi_nodes.get(key, [0, 0, 0])
            line = self.cn_lines_objects.get(key, None)
            if line is None:
                continue

            x = [node[0], node[0] + disp_node[0]
                 * np.cos(self.seq_num / 25 * 2 * np.pi + phi_node[0])]
            y = [node[1], node[1] + disp_node[1]
                 * np.cos(self.seq_num / 25 * 2 * np.pi + phi_node[1])]
            z = [node[2], node[2] + disp_node[2]
                 * np.cos(self.seq_num / 25 * 2 * np.pi + phi_node[2])]
            line.set_visible(self.show_cn_lines)
            line.set_data_3d([x, y, z])
            # line.set_3d_properties(z)

        self.fig.canvas.draw_idle()

    def refresh_nd_lines(self, visible=None):
        '''
        Refresh the non-displaced lines and make them visible/invisible, e.g.
        after programmatically changing visibility flags.

        Parameters
        ----------
            visible: bool, ooptional
                Visibility flag for the non-displaced lines

        '''

        if visible is not None:
            visible = bool(visible)
            self.show_nd_lines = visible

        for line, line_node in zip(
                self.nd_lines_objects, self.geometry_data.lines):
            x = [self.geometry_data.nodes[node][0]
                 for node in line_node]
            y = [self.geometry_data.nodes[node][1]
                 for node in line_node]
            z = [self.geometry_data.nodes[node][2]
                 for node in line_node]
            line.set_visible(self.show_nd_lines)
            line.set_data_3d([x, y, z])
            # line.set_3d_properties(z)

        self.fig.canvas.draw_idle()

    def refresh_cn_lines(self, visible=None):
        '''
        Refresh the connecting lines and make them visible/invisible, e.g.
        after programmatically changing visibility flags.

        Parameters
        ----------
            visible: bool, ooptional
                Visibility flag for the non-displaced lines

        '''

        if visible is not None:
            visible = bool(visible)
            self.show_cn_lines = visible

        for key, node in self.geometry_data.nodes.items():
            disp_node = self.disp_nodes.get(key, [0, 0, 0])
            phi_node = self.phi_nodes.get(key, [0, 0, 0])
            line = self.cn_lines_objects.get(key, None)
            if line is not None:
                x = [node[0], node[0] + disp_node[0]
                     * np.cos(self.seq_num / 25 * 2 * np.pi + phi_node[0])]
                y = [node[1], node[1] + disp_node[1]
                     * np.cos(self.seq_num / 25 * 2 * np.pi + phi_node[1])]
                z = [node[2], node[2] + disp_node[2]
                     * np.cos(self.seq_num / 25 * 2 * np.pi + phi_node[2])]
                line.set_visible(self.show_cn_lines)
                line.set_data_3d([x, y, z])
                # line.set_3d_properties(z)

        self.fig.canvas.draw_idle()

    def draw_parent_childs(self):
        '''
        Draw arrows for all parent-child definitions stored in the
        internal parent-child definition table.
        '''
        for i, (i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl) in enumerate(
                self.geometry_data.parent_childs):
            self.add_parent_child(
                i_m, x_m * self.scale, y_m * self.scale, z_m * self.scale,
                i_sl, x_sl * self.scale, y_sl * self.scale, z_sl * self.scale,
                i)

    def refresh_parent_childs(self, visible=None):
        '''
        Refresh the parent-child arrows and make them visible/invisible, e.g.
        after programmatically changing visibility flags.

        Will not be shown in displaced mode (modeshape)

        Parameters
        ----------
            visible: bool, ooptional
                Visibility flag for the parent-child arrows

        '''
        if visible is not None:
            visible = bool(visible)
            self.show_parent_childs = visible

        for patch in self.arrows_objects:
            for obj in patch:
                obj.set_visible(self.show_parent_childs)
        self.fig.canvas.draw_idle()

    def draw_chan_dofs(self):
        '''
        Draw arrows and numbers for all channel-DOF assignments stored
        in the channel - DOF assignment table of PreProcessingTools.GeometrProcessor
        '''
        for i, chan_dof in enumerate(self.chan_dofs):

            chan, node, az, elev, chan_name = chan_dof[0:4] + chan_dof[-1:]
            if node is None:
                continue
            if node not in self.geometry_data.nodes.keys():
                continue
            self.add_chan_dof(chan, node, az, elev, chan_name, i)

    def refresh_chan_dofs(self, visible=None):
        '''
        Refresh the arrows indicating the channel-dof assignments
        and make them visible/invisible, e.g. after programmatically
        changing visibility flags.

        Will not be shown in displaced mode (modeshape)

        Parameters
        ----------
            visible: bool, ooptional
                Visibility flag for the channel-dof assignment arrows

        '''
        if visible is not None:
            visible = bool(visible)
            self.show_chan_dofs = visible

        for patch in self.channels_objects:
            patch.set_visible(self.show_chan_dofs)
        self.fig.canvas.draw_idle()

    def draw_msh(self):
        '''
        Draw mode shapes by assigning displacement values to the
        nodes based on the channel - DOF assignments and the parent -
        child definitions. Draws the displaced nodes and beams.

        .. Todo::
            * The computation of resulting magnitude and phase angles for
              displacements based on parent-child definitions is currently
              more or less broken. It should be possible, even in 3D to
              compute exact solutions.
        '''

        def to_phase_mag(disp):
            if self.real:
                phase = np.angle(disp, True)
                mag = np.abs(disp)
                if phase < 0:
                    phase += 180
                    mag = -mag
                if phase > 90 and phase < 270:
                    mag = -mag
                phase = 0
            else:
                phase = np.angle(disp)
                mag = np.abs(disp)
            return phase, mag

        mode_shape = self.mode_shapes[:,
                                      self.mode_index[1], self.mode_index[0]]
        # print(mode_shape)
        mode_shape = ModalBase.rescale_mode_shape(mode_shape)
        ampli = self.amplitude

        self.disp_nodes = {i: [0, 0, 0]
                           for i in self.geometry_data.nodes.keys()}

        self.phi_nodes = {i: [0, 0, 0]
                          for i in self.geometry_data.nodes.keys()}

        chan_found = [False for chan in range(len(mode_shape))]

        for node in self.geometry_data.nodes.keys():
            this_chan_dofs = []
            for chan_dof in self.chan_dofs:
                chan, node_, az, elev, chan_name = chan_dof[0:4] + \
                    chan_dof[-1:]
                if node_ == node:
                    disp = mode_shape[chan]

                    # radius 1 is needed for the coordinate transformation to
                    # work
                    x, y, z = calc_xyz(
                        az * np.pi / 180, elev * np.pi / 180, r=1)

                    this_chan_dofs.append([chan, x, y, z, disp])

                    chan_found[chan] = True

            if len(this_chan_dofs) == 0:
                continue  # no sensors in this node

            elif len(this_chan_dofs) == 1:  # only one sensor in this node
                chan, x, y, z, disp = this_chan_dofs[0]

                phase, mag = to_phase_mag(disp)

                self.phi_nodes[node][0] = phase
                self.disp_nodes[node][0] = x * mag * ampli

                self.phi_nodes[node][1] = phase
                self.disp_nodes[node][1] = y * mag * ampli

                self.phi_nodes[node][2] = phase
                self.disp_nodes[node][2] = z * mag * ampli

            else:  # two or more sensors in this node

                # check if sensors are in direction of the coordinate
                # system or if they need to be transformed
                sum_x = 0
                sum_y = 0
                sum_z = 0
                for chan, x, y, z, disp in this_chan_dofs:
                    # print(chan,x,y,z)
                    if not np.isclose(x, 0):
                        sum_x += 1
                    if not np.isclose(y, 0):
                        sum_y += 1
                    if not np.isclose(z, 0):
                        sum_z += 1
                # print(sum_x, sum_y, sum_z)
                if sum_x <= 1 and sum_y <= 1 and sum_z <= 1:  # sensors are in coordinate direction

                    for chan, x, y, z, disp in this_chan_dofs:

                        phase, mag = to_phase_mag(disp)

                        if not np.isclose(x, 0):
                            self.phi_nodes[node][0] = phase
                            self.disp_nodes[node][0] = x * mag * ampli
                        elif not np.isclose(y, 0):
                            self.phi_nodes[node][1] = phase
                            self.disp_nodes[node][1] = y * mag * ampli
                        elif not np.isclose(z, 0):
                            self.phi_nodes[node][2] = phase
                            self.disp_nodes[node][2] = z * mag * ampli
                else:
                    num_sensors = max(len(this_chan_dofs), 3)
                    # at least three sensors are needed for the coordinate transformation
                    # if only two sensors are present, they will be complemented by
                    # a zero displacement assumption in perpendicular direction
                    normal_matrix = np.zeros((num_sensors, 3))
                    disp_vec = np.zeros(num_sensors, dtype=complex)
                    for i, (chan, x, y, z, disp) in enumerate(this_chan_dofs):
                        normal_matrix[i,:] = [x, y, z]
                        disp_vec[i] = disp

                    if i == 1:  # only two sensors were present
                        logging.info(
                            'Not enough sensors for a full 3D transformation at node {}, '
                            'will complement vectors with a zero displacement assumption in orthogonal direction.'.format(node))
                        # vector c is perpendicular to the first two vectors
                        c = np.cross(normal_matrix[0,:], normal_matrix[1,:])
                        # if angle between first two vectors is different from
                        # 90° vector c has to be normalized
                        c /= np.linalg.norm(c)
                        # print(node, c)
                        normal_matrix[2,:] = c

                    '''
                    ⎡ n_1,x  n_1,y  n_1,z ⎤ ⎡ q_res_x ⎤   ⎡ d_1 ⎤
                    ⎢ n_2,x  n_2,y  n_2,z ⎥ ⎢ q_res_y ⎥ = ⎢ d_2 ⎥
                    ⎣ n_3,x  n_3,y  n_3,z ⎦ ⎣ q_res_z ⎦   ⎣ d_3 ⎦
                    '''
                    # solve the well- or over-determined system of equations
                    q_res = np.linalg.lstsq(normal_matrix, disp_vec, rcond=None)[0]

                    for i in range(3):
                        disp = q_res[i]
                        # print(disp)
                        phase, mag = to_phase_mag(disp)

                        self.phi_nodes[node][i] = phase
                        self.disp_nodes[node][i] = mag * ampli

        for chan, found in enumerate(chan_found):
            if not found:
                logging.warning('Could not find channel - DOF assignment for '
                                'channel {}!'.format(chan))

        for i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl in self.geometry_data.parent_childs:

            if (x_m > 0 + y_m > 0 + z_m > 0) > 1:
                logging.warning(
                    'parent DOF includes more than one cartesian direction. Phase angles will be distorted.')

            parent_disp = self.disp_nodes[i_m][0] * x_m + \
                self.disp_nodes[i_m][1] * y_m + \
                self.disp_nodes[i_m][2] * z_m

            parent_phase = self.phi_nodes[i_m][0] * x_m + \
                self.phi_nodes[i_m][1] * y_m + \
                self.phi_nodes[i_m][2] * z_m

            if not np.allclose(x_sl, 0):
                # print(x, phase)
                if self.disp_nodes[i_sl][0] > 0:
                    logging.warning(
                        'A modal coordinate of {} has already been assigned to this DOF x of node {}. Overwriting!'.format(self.disp_nodes[i_sl][0], i_sl))
                self.phi_nodes[i_sl][0] = parent_phase
                self.disp_nodes[i_sl][0] += parent_disp * x_sl
            if not np.allclose(y_sl, 0):
                # print(y,phase)
                if self.disp_nodes[i_sl][1] > 0:
                    logging.warning(
                        'A modal coordinate of {} has already been assigned to this DOF y of node {}. Overwriting!'.format(self.disp_nodes[i_sl][1], i_sl))
                self.phi_nodes[i_sl][1] = parent_phase
                self.disp_nodes[i_sl][1] += parent_disp * y_sl
            if not np.allclose(z_sl, 0):
                # print(z,phase)
                if self.disp_nodes[i_sl][2] > 0:
                    logging.warning(
                        'A modal coordinate of {} has already been assigned to this DOF z of node {}. Overwriting!'.format(self.disp_nodes[i_sl][2], i_sl))
                self.phi_nodes[i_sl][2] = parent_phase
                self.disp_nodes[i_sl][2] += parent_disp * z_sl
            # print(i_m, parent_disp, self.disp_nodes[i_sl])
#             if self.draw_trace:
#                 if self.trace_objects:
#                     for i in range(len(self.trace_objects)-1,-1,-1):
#                         try:
#                             self.trace_objects[i].remove()
#                         except Exception as e:
#                             pass
#                             #print("Error",e)
#
#                         del self.trace_objects[i]
#
#                 moving_nodes = set()
#                 for chan_dof in self.chan_dofs:#
#                     chan_, node, az, elev, chan_name  = chan_dof[0:4]+ [chan_dof[-1]]
#                     if node is None:
#                         continue
#                     if not node in self.geometry_data.nodes.keys():
#                         continue
#                     moving_nodes.add(node)
#
#                 clist = itertools.cycle(list(matplotlib.cm.jet(np.linspace(0, 1, len(moving_nodes)))))#@UndefinedVariable
#                 for node in moving_nodes:
#                     self.trace_objects.append(self.subplot.plot(xs=self.geometry_data.nodes[node][0] + self.disp_nodes[node][0]
#                          * np.cos(np.linspace(0,359,360) / 360 * 2 * np.pi  + self.phi_nodes[node][0]),
#                          ys=self.geometry_data.nodes[node][1] + self.disp_nodes[node][1]
#                          * np.cos(np.linspace(0,359,360) / 360 * 2 * np.pi  + self.phi_nodes[node][1]),
#                          zs=self.geometry_data.nodes[node][2] + self.disp_nodes[node][2]
#                          * np.cos(np.linspace(0,359,360) / 360 * 2 * np.pi  + self.phi_nodes[node][2]),
#                          #marker = ',', s=1, edgecolor='none',
#                          color = next(clist)))

        self.refresh_nodes()
        self.refresh_lines()
        self.refresh_chan_dofs(False)
        self.refresh_parent_childs(False)
        if self.animated:
            self.stop_ani()
            self.animate()
        self.set_equal_aspect()

        self.fig.canvas.draw()

    def set_equal_aspect(self):

        minx, maxx, miny, maxy, minz, maxz = self.subplot.get_w_lims()
        dx, dy, dz = (maxx - minx), (maxy - miny), (maxz - minz)

        if dx != dy or dx != dz:
            midx = 0.5 * (minx + maxx)
            midy = 0.5 * (miny + maxy)
            midz = 0.5 * (minz + maxz)

            hrange = max(dy, dy, dz) * 0.5
            self.subplot.set_xlim3d(midx - hrange, midx + hrange)
            self.subplot.set_ylim3d(midy - hrange, midy + hrange)
            self.subplot.set_zlim3d(midz - hrange, midz + hrange)

    # @pyqtSlot()
    def stop_ani(self):
        '''
        Convenience method to stop the animation and restore the still plot
        '''
        if self.animated or self.data_animated:
            self.seq_num = next(self.line_ani.frame_seq)
            self.line_ani._stop()
            if self.trace_objects:
                for i in range(len(self.trace_objects) - 1, -1, -1):
                    try:
                        self.trace_objects[i].remove()
                    except BaseException as e:
                        print(e)
                        pass

                    del self.trace_objects[i]
            # self.draw_trace = False
            self.animated = False
            self.data_animated = False
            for c in self.connect_handles:
                self.fig.canvas.mpl_disconnect(c)
            self.draw_nodes()
            self.refresh_nodes()
            self.draw_lines()
            self.refresh_lines()
            self.refresh_nd_lines()
            self.refresh_parent_childs()
            self.refresh_chan_dofs()
            # self.draw_msh()

    # @pyqtSlot()

    def animate(self):
        '''
        Create necessary objects to animate the currently displayed
        deformed structure.

        If self.save_ani_path is given, the animation will be saved to that 
        folder. The **numbering** of the **files**
        follows the order in which the modes were selected in the
        stabilization diagram.

        '''

        # self.save_ani_path = False
        #
        # if self.save_ani_path:
        #     self.cwd = '/vegas/users/staff/womo1998/Projects/2019_Schwabach/tex/figures/ani_high/'  # os.getcwd()
        #     # for i in range(len(self.select_modes)):
        #     #    os.makedirs(os.path.join(self.cwd,str(i)), exist_ok=True)
        #
        # # self.draw_trace = True

        def init_lines():
            '''
            Initialize line objects for later update.
            '''
            minx, maxx, miny, maxy, minz, maxz = self.subplot.get_w_lims()

            # self.subplot.cla()
            # self.subplot.set_aspect('equal', 'datalim')
            # self.subplot.patch = self.fig.patch
            # self.subplot.grid(False)
            # self.subplot.set_axis_off()
            # return self.lines_objects
            # self.draw_lines()
            # self.draw_axis()
            for i, line in enumerate(self.lines_objects):
                line.set_visible(False)
                # line.set_clip_path(self.fig.patch)
                if isinstance(self.beamcolor, (list, tuple, np.ndarray)):
                    beamcolor = self.beamcolor[i]
                else:
                    beamcolor = self.beamcolor
                if isinstance(self.beamstyle, (list, tuple, np.ndarray)):
                    beamstyle = self.beamstyle[i]
                else:
                    beamstyle = self.beamstyle
                line.set_color(beamcolor)
                line.set_linestyle(beamstyle)

            for line in self.nd_lines_objects:
                # pass
                line.set_visible(False)
                # line.set_clip_path(self.fig.patch)

            for line in self.cn_lines_objects.values():
                line.set_visible(False)
                # line.set_clip_path(self.fig.patch)

            self.fig.canvas.draw()
            self.subplot.set_xlim3d(minx, maxx)
            self.subplot.set_ylim3d(miny, maxy)
            self.subplot.set_zlim3d(minz, maxz)

#             this_dirs={}
#
#             for node in ['1','2','3','4','5','6','7']:
#                 this_chans, this_az = [],[]
#                 for chan, node_, az,elev,header in self.chan_dofs:
#                     if node == node_:
#                         this_chans.append(chan)
#                         this_az.append(az)
#                 if len(this_chans) != 2:
#                     continue
#
#                 this_dirs[node]={}
#
#                 x,y=[],[]
#                 for t in np.linspace(-np.pi,np.pi,359):
#                     x.append(0)
#                     y.append(0)
#                     for j,az in enumerate(this_az):
#                         x_,y_= self.calc_xy(np.radians(az))
#                         x[-1]+= np.abs(msh[j])*x_* np.cos(t  + np.angle(msh[j]))
#                         y[-1]+= np.abs(msh[j])*y_* np.cos(t  + np.angle(msh[j]))
#                 #plot.figure(figsize=(8,8))
#                 if i == 1 and k==0:
#                     ind = ['1','4','5','6','3','2'].index(node)
#                     import matplotlib.cm
#                     color=list(matplotlib.cm.hsv(np.linspace(0, 1, 7)))[ind]
#                     plot.plot(x,y, label=['108','126','145','160','188','TMD'][ind], color=color)

            if self.show_cn_lines:
                if self.trace_objects:
                    for i in range(len(self.trace_objects) - 1, -1, -1):
                        try:
                            self.trace_objects[i].remove()
                        except BaseException:
                            pass

                        del self.trace_objects[i]
                # assemble the list of moving nodes for which traces
                # should be drawn, this currently does not account for
                # parent-child definitions
                moving_nodes = set()
                for chan_dof in self.chan_dofs:
                    _, node, _, _, = chan_dof[0:4]
                    if node is None:
                        continue
                    if node not in self.geometry_data.nodes.keys():
                        continue
                    moving_nodes.add(node)

                clist = itertools.cycle(
                    ['darkgray' for i in range(len(moving_nodes))])
                for node in moving_nodes:
                    self.trace_objects.append(
                        self.subplot.plot(
                            xs=self.geometry_data.nodes[node][0] + self.disp_nodes[node][0] *
                            np.cos(np.arange(0, 2 * np.pi, np.pi / 180) + self.phi_nodes[node][0]),
                            ys=self.geometry_data.nodes[node][1] + self.disp_nodes[node][1] *
                            np.cos(np.arange(0, 2 * np.pi, np.pi / 180) + self.phi_nodes[node][1]),
                            zs=self.geometry_data.nodes[node][2] + self.disp_nodes[node][2] *
                            np.cos(np.arange(0, 2 * np.pi, np.pi / 180) + self.phi_nodes[node][2]),
                            color=next(clist), linewidth=1, linestyle=(0, (1, 1)))[0])
                    # for artist in self.trace_objects[-1]:
                    #     artist.set_clip_on(False)

            # self.subplot.patch = self.fig.patch
            return self.lines_objects + \
                self.nd_lines_objects + \
                self.trace_objects + \
                list(self.cn_lines_objects.values())  # + \
                # list(self.axis_obj.values())
            # return self.lines_objects#, self.nd_lines_objects

        def update_lines(num):
            '''
            Subfunction to calculate displacements based on magnitude and phase angle
            '''
            # print(num)

#             if not self.traced: clist = itertools.cycle(matplotlib.rcParams['axes.color_cycle'])

            for i, (line, line_node) in enumerate(
                    zip(self.lines_objects, self.geometry_data.lines)):
                x = [self.geometry_data.nodes[node][0] + self.disp_nodes[node][0]
                     * np.cos(num / 25 * 2 * np.pi + self.phi_nodes[node][0])
                     for node in line_node]
                y = [self.geometry_data.nodes[node][1] + self.disp_nodes[node][1]
                     * np.cos(num / 25 * 2 * np.pi + self.phi_nodes[node][1])
                     for node in line_node]
                z = [self.geometry_data.nodes[node][2] + self.disp_nodes[node][2]
                     * np.cos(num / 25 * 2 * np.pi + self.phi_nodes[node][2])
                     for node in line_node]

                # NOTE: there is no .set_data() for 3 dim data...
                line.set_visible(self.show_lines)
                line.set_data_3d([x, y, z])
            rets = [self.lines_objects]
                # line.set_3d_properties(z)
            if self.nd_lines_objects[0].get_visible() != self.show_nd_lines:
                for line in self.nd_lines_objects:
                    line.set_visible(self.show_nd_lines)

                rets.append(self.nd_lines_objects)

            for trace_objects in self.trace_objects:
                trace_objects = [trace_objects]  # hack to circumvent many code changes
                for artist in trace_objects:
                    artist.set_visible(self.show_cn_lines)
                rets.append(trace_objects)

            if self.axis_obj['X'].get_visible() != self.show_axis:
                for axis in self.axis_obj.values():
                    axis.set_visible(self.show_axis)

                rets.append(self.axis_obj.values())

            if self.save_ani_path and num <= 25:
                self.fig.savefig(
                    self.save_ani_path / f'{self.select_modes.index(self.mode_index)}' / f'ani_{num}.pdf')
                # if i>25: self.stop_ani()

            return [num for sublist in rets for num in sublist]  # self.lines_objects #+ \
                # self.nd_lines_objects + \
                # list(self.cn_lines_objects.values())

        # self.cla()
        # self.patches_objects = {}
        # self.lines_objects = []
        # self.nd_lines_objects = []
        # self.cn_lines_objects = {}
        # self.arrows_objects = []
        # self.channels_objects = []
        # self.axis_obj = {}

        if self.animated:
            return self.stop_ani()
        else:
            if self.data_animated:
                self.stop_ani()
            self.animated = True

        c1 = self.fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        c2 = self.fig.canvas.mpl_connect('button_press_event', self._button_press)
        c3 = self.fig.canvas.mpl_connect(
            'button_release_event',
            self._button_release)
        self.connect_handles = [c1, c2, c3]
        self.button_pressed = None

        self.line_ani = matplotlib.animation.FuncAnimation(
            fig=self.fig,
            func=update_lines,
            init_func=init_lines,
            interval=50,
            save_count=50,
            blit=True)

        self.fig.canvas.draw()

    # @pyqtSlot()

    def filter_and_animate_data(self, callback=None):
        '''
        Animate the acquired vibration data to check the real vibration
        displacement of the structure against the identified modes.
        '''

        def init_lines():
            # print('init')
            # self.clear_plot()
            minx, maxx, miny, maxy, minz, maxz = self.subplot.get_w_lims()

            self.subplot.cla()
            # return self.lines_objects
            self.draw_lines()
            for line in self.lines_objects:
                line.set_visible(False)
            for line in self.nd_lines_objects:
                line.set_visible(False)
            for line in self.cn_lines_objects.values():
                line.set_visible(False)

            self.subplot.set_xlim3d(minx, maxx)
            self.subplot.set_ylim3d(miny, maxy)
            self.subplot.set_zlim3d(minz, maxz)

            return self.lines_objects + \
                self.nd_lines_objects + \
                list(self.cn_lines_objects.values())
            # return self.lines_objects#, self.nd_lines_objects

        def update_lines(num):
            '''
            Subfunction to calculate displacements.
            '''
            self.callback(f'{num/self.prep_signals.sampling_rate:.4f}')
            disp_nodes = {i: [0, 0, 0]
                          for i in self.geometry_data.nodes.keys()}
            for chan_dof in self.chan_dofs:
                chan_, node, az, elev, = chan_dof[0:4]

                if node is None:
                    continue
                if node not in self.geometry_data.nodes.keys():
                    continue
                x, y, z = calc_xyz(az * np.pi / 180, elev * np.pi / 180)
                disp_nodes[node][0] += self.prep_signals.signals_filtered[num,
                                                                       chan_] * x * self.amplitude
                disp_nodes[node][1] += self.prep_signals.signals_filtered[num,
                                                                       chan_] * y * self.amplitude
                disp_nodes[node][2] += self.prep_signals.signals_filtered[num,
                                                                       chan_] * z * self.amplitude

            # print(num)
            for line, line_node in zip(
                    self.lines_objects, self.geometry_data.lines):
                x = [self.geometry_data.nodes[node][0] + disp_nodes[node][0]
                     for node in line_node]
                y = [self.geometry_data.nodes[node][1] + disp_nodes[node][1]
                     for node in line_node]
                z = [self.geometry_data.nodes[node][2] + disp_nodes[node][2]
                     for node in line_node]
                # NOTE: there is no .set_data() for 3 dim data...
                line.set_visible(self.show_lines)
                line.set_data_3d([x, y, z])
                line.set_color('b')
                # line.set_3d_properties(z)

            for line in self.nd_lines_objects:
                line.set_visible(self.show_nd_lines)

            for key in self.geometry_data.nodes.keys():
                node = self.geometry_data.nodes[key]
                disp_node = disp_nodes.get(key, [0, 0, 0])
                x = [node[0], node[0] + disp_node[0]]
                y = [node[1], node[1] + disp_node[1]]
                z = [node[2], node[2] + disp_node[2]]
                line = self.cn_lines_objects.get(key, None)
                if line is not None:
                    line.set_data_3d([x, y, z])
                    line.set_visible(self.show_cn_lines)
                    # line.set_3d_properties(z)

            return self.lines_objects + \
                self.nd_lines_objects + \
                list(self.cn_lines_objects.values())

        # self.cla()
        # self.patches_objects = {}
        self.lines_objects = []
        self.nd_lines_objects = []
        self.cn_lines_objects = {}
        self.arrows_objects = []
        self.channels_objects = []
        self.axis_obj = {}

        if self.data_animated:
            return self.stop_ani()
        else:
            if self.animated:
                self.stop_ani()
            self.data_animated = True

        c1 = self.fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        c2 = self.fig.canvas.mpl_connect('button_press_event', self._button_press)
        c3 = self.fig.canvas.mpl_connect(
            'button_release_event',
            self._button_release)
        self.connect_handles = [c1, c2, c3]
        self.button_pressed = None

        # self.prep_signals.filter_data(lowpass, highpass)
        if callback is not None:
            self.callback = callback
        self.line_ani = matplotlib.animation.FuncAnimation(
            fig=self.fig,
            func=update_lines,
            frames=range(
                self.prep_signals.signals_filtered.shape[0]),
            init_func=init_lines,
            interval=1 /
            self.prep_signals.sampling_rate,
            save_count=0,
            blit=True)

        self.fig.canvas.draw()

    def _button_press(self, event):
        if event.inaxes == self.subplot:
            self.button_pressed = event.button

    def _button_release(self, event):
        self.button_pressed = None

    def _on_move(self, event):
        if not self.button_pressed:
            return

        for line in self.lines_objects:
            line.set_visible(False)
        for line in self.nd_lines_objects:
            line.set_visible(False)
        for line in self.cn_lines_objects.values():
            line.set_visible(False)
        # self.fig.canvas.draw()
        self.line_ani._setup_blit()
        # self.line_ani._start()


class LabeledArrow3D(matplotlib.patches.FancyArrowPatch):
    '''
    credit goes to (don't know the original author):
    http://pastebin.com/dWvFxb1Q
    draw an arrow in 3D space
    '''

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        '''
        inherit from matplotlib.patches.FancyArrowPatch
        and set self._verts3d class variable
        dx,dy,dz is understood as fractions of the axis'limits
        '''

        self.text = None
        self._verts3d = (x, y, z, dx, dy, dz)
        super().__init__((x, x + dx), (y, y + dy), *args, **kwargs)

    def set_visible(self, b):

        if self.text is not None:
            self.text.set_visible(b)
        super().set_visible(b)

    def add_label(self, text, color=None, visible=True):

        if self.axes is None:
            logging.warning('The arrow must be added to an axes, before a label can be added.')

        (x, y, z, dx, dy, dz) = self._verts3d

        self.text = self.axes.text(
            x + dx,
            y + dy,
            z + dz,
            text,
            color=color,
            visible=visible)

    def draw(self, renderer):
        '''
        get the projection from the 3D point to 2D point to draw the arrow
        '''

        # scale and direction of the arrow as fractions of axis limits
        x, y, z, dx, dy, dz = self._verts3d

        minx, maxx, miny, maxy, minz, maxz = self.axes.get_w_lims()
        lx, ly, lz = (maxx - minx), (maxy - miny), (maxz - minz)

        # rescale arrow to fraction axis limits
        xs3d = [x, x + lx * dx]
        ys3d = [y, y + ly * dy]
        zs3d = [z, z + lz * dz]
        xs, ys, zs = mpl_toolkits.mplot3d.axes3d.proj3d.proj_transform(
            xs3d, ys3d, zs3d, self.axes.M)
        if self.text:
            self.text.set_position_3d((xs3d[1], ys3d[1], zs3d[1]))
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1, dx, dy, dz = self._verts3d
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = mpl_toolkits.mplot3d.axes3d.proj3d.proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


if __name__ == "__main__":
    pass
