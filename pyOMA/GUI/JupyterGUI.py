# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2025  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
"""ipywidgets-based interactive GUI for Jupyter notebooks."""
import logging
import ipywidgets
import ipympl
from IPython.core.display_functions import display
import numpy as np
import scipy.spatial
import scipy.stats
from pyOMA.core.Helpers import get_method_dict
import os
from pathlib import Path
# import ipywidgets
# import ipympl.backend_nbagg # the ''%matplotlib widget' backend

# Monkeypatch webagg to support blitting until https://github.com/matplotlib/matplotlib/pull/27160 is merged upstream
# copied and modified from https://github.com/raphaelquast/EOmaps/blob/66d32e2f5219059ab32a02457c535652d3e3f881/eomaps/_maps_base.py#L147

# #Creating a logger
# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)
#
# # log exceptions to logger instead of stderr
# def showtraceback(self, *args, **kwargs):
#
#     logger.critical("Unhandled exception", exc_info=sys.exc_info())
#
# from IPython import get_ipython
# ipython = get_ipython()
# ipython.set_custom_exc((Exception,), showtraceback)


class SnappingCursor:
    """
    A cross-hair cursor that snaps to the data point of a line, which is
    closest to the cursor.

    .. TODO::
        waiting for https://github.com/matplotlib/matplotlib/pull/27160 to be approved
        then blitting should be tested and enabled
    """

    def __init__(self, ax, f_data, order_data):
        self.ax = ax

        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')  # , animated=True)
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')  # , animated=True)

        self.data_shape = f_data.shape

        self.n_points = np.prod(self.data_shape)
        self.data = np.ma.empty((self.n_points, 2))

        self.data[:, 0] = f_data.reshape((self.n_points,))
        self.data[:, 1] = order_data.reshape((self.n_points,))

        # copy mask from frequency-data (x) to order-data (y)
        if isinstance(f_data, np.ma.MaskedArray):
            mask = f_data.mask.reshape((self.n_points,))
        else:
            mask = np.ma.nomask
        self.data.mask[:, 0] = mask
        self.data.mask[:, 1] = mask

        self._last_index = None

        self.callbacks = {'show_current_info':lambda *args, **kwargs: None,
                          'mode_selected':lambda *args, **kwargs: None,
                          'mode_deselected':lambda *args, **kwargs: None, }

        self.update_pix_data()

    def add_callback(self, name, func):
        if name not in ['show_current_info', 'mode_selected', 'mode_deselected']:
            raise ValueError(f"'name' must be one of {['show_current_info', 'mode_selected', 'mode_deselected']}, got {name!r}.")
        self.callbacks[name] = func

    def set_mask(self, mask, name=None):  # name just for backwards compatibility
        n_mask = mask.reshape((self.n_points,))
        self.data.mask[:, 0] = n_mask
        self.data.mask[:, 1] = n_mask
        self.update_pix_data()

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            self._last_index = None
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata

            x_, y_ = self.ax.transData.transform(
                np.vstack([x, y]).T).T

            index = self.findIndexNearestXY(x_, y_)

            if index == self._last_index:
                return  # still on the same data point. Nothing to do.
            self._last_index = index

            x = self.data[index, 0:1]
            y = self.data[index, 1:2]
            # update the line positions
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)

            # self.ax.figure.canvas.restore_region(self.bg)
            # self.ax.draw_artist(self.horizontal_line)
            # self.ax.draw_artist(self.vertical_line)
            # self.ax.figure.canvas.blit(self.ax.figure.bbox)
            # self.ax.figure.canvas.flush_events()

            self.ax.figure.canvas.draw()

            if np.ma.is_masked(x):
                self.data_index = None
            else:
                self.data_index = np.unravel_index(index, self.data_shape)
                self.callbacks['show_current_info'](self.data_index)

    def on_button(self, event=None):
        if self.data_index is not None:
            self.callbacks['mode_selected'](self.data_index)

    def findIndexNearestXY(self, x_point, y_point):
        '''
        Finds the nearest neighbour
        '''
        # distance = np.square(self.pix_data[:, 1] - y_point) + np.square(self.pix_data[:, 0] - x_point)
        # index = np.argmin(distance)

        _, index = self.tree.query(np.hstack([x_point, y_point]), 1)
        return index

    def update_pix_data(self, event=None):
        self.pix_data = self.ax.transData.transform(self.data)
        self.pix_data._mask = self.data._mask

        # the slow thing is the redraw of all poles
        # the following only speeds up tree lookup
        #
        # xmin, xmax = self.ax.get_xlim()
        # ymin, ymax = self.ax.get_ylim()
        #
        # xmask = np.logical_and(self.data[:,0]>xmin, self.data[:,0]<xmax)
        # ymask = np.logical_and(self.data[:,1]>ymin, self.data[:,1]<ymax)
        #
        # datamask = ~np.logical_and(xmask, ymask)
        # data_mask = np.hstack([datamask[:,np.newaxis],datamask[:,np.newaxis]])
        #
        # self.pix_data._mask = np.logical_or(self.data._mask, data_mask)

        self.tree = scipy.spatial.KDTree(self.pix_data)
        # fig = self.ax.get_figure()
        # self.bg = fig.canvas.copy_from_bbox(fig.bbox)
        # self.ax.draw_artist(self.horizontal_line)
        # self.ax.draw_artist(self.vertical_line)
        # fig.canvas.blit(fig.bbox)


class OutputWidgetHandler(logging.Handler):
    """ Custom logging handler sending logs to an output widget """

    def __init__(self, *args, **kwargs):
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        layout = {
            'width': '80%',
            'height': '160px',
            'border': '1px solid black',
            'overflow': 'scroll',
            'position':'bottom'
        }
        self.out = ipywidgets.Output(layout=layout)

    def emit(self, record):
        """ Overload of logging.Handler method """
        formatted_record = self.format(record)
        new_output = {
            'name': 'stdout',
            'output_type': 'stream',
            'text': formatted_record + '\n'
        }

        self.out.outputs = self.out.outputs[-8:] + (new_output,)

    def show_logs(self):
        """ Show the logs """
        display(self.out)

    def clear_logs(self):
        """ Clear the current logs """
        self.out.clear_output()


def _html_row(label, value, fmt):
    """Return an HTML table row string, or empty string if *value* is NaN."""
    if np.isnan(value):
        return ''
    return f'<tr>\n<td> {label}:</td>\n <td> {value:{fmt}} </td>\n</tr>\n'


def _build_modal_values_html(vals, capabilities):
    """Build an HTML table string from modal values tuple.

    Parameters
    ----------
    vals : tuple
        ``(n, f, stdf, d, stdd, mpc, mp, mpd, dmp, _dmpd, mtn, MC, ex_1, ex_2)``
        as returned by ``stabil_calc.get_modal_values``.
    capabilities : dict
        The ``stabil_calc.capabilities`` dict; used to decide whether to
        scale confidence intervals.

    Returns
    -------
    str
        HTML table string.
    """
    n, f, stdf, d, stdd, mpc, mp, mpd, dmp, _dmpd, mtn, MC, ex_1, ex_2 = vals
    rows = [
        _html_row('Frequency [Hz]', f, '1.3f'),
        _html_row('CI Frequency [Hz]', stdf, '1.3e'),
        _html_row('Model order', n, '1.0f'),
        _html_row('Damping [%]', d, '1.3f'),
        _html_row('CI Damping [%]', stdd, '1.3e'),
        _html_row('MPC [-]', mpc, '1.5f'),
        _html_row('MP  [°]', mp, '1.3f'),
        _html_row('MPD [-]', mpd, '1.5f'),
        _html_row('dMP  [°]', dmp, '1.3f'),
        _html_row('MTN [%]', mtn, '1.5f'),
        _html_row('MC [%]', MC, '1.5f'),
        _html_row('Ext [-]', ex_1, '1.5f'),
        _html_row('Ext [-]', ex_2, '1.5f'),
    ]
    return '<table>\n' + ''.join(rows) + '</table>'


def _build_stabil_softbox(stabil_calc, update_fn):
    """Build the soft-criteria VBox widget for the stabilisation UI."""
    widgets = []
    lb = ipywidgets.Label("Soft criteria:")
    widgets.append(lb)
    if stabil_calc.capabilities['f']:
        sl_df = ipywidgets.FloatLogSlider(value=stabil_calc.df_max, base=10, min=-4, max=0, step=0.1, description="Frequency [%]")
        sl_df.observe(lambda change: update_fn(df_max=float(change['new'])),
                      names='value', type='change')
        widgets.append(sl_df)
    if stabil_calc.capabilities['d']:
        sl_dd = ipywidgets.FloatLogSlider(value=stabil_calc.dd_max, base=10, min=-4, max=0, step=0.1, description="Damping [%]")
        sl_dd.observe(lambda change: update_fn(dd_max=float(change['new'])),
                      names='value', type='change')
        widgets.append(sl_dd)
    if stabil_calc.capabilities['msh']:
        sl_dmac = ipywidgets.FloatLogSlider(value=stabil_calc.dmac_max, base=10, min=-4, max=0, step=0.1, description="MAC [%]")
        sl_dmac.observe(lambda change: update_fn(dmac_max=float(change['new'])),
                        names='value', type='change')
        widgets.append(sl_dmac)
    # ..TODO:: add Eigenvalue distance selector
    return ipywidgets.VBox(widgets, layout=ipywidgets.Layout(width='350px', border='solid 1px'))


def _build_stabil_hardbox(stabil_calc, update_fn):
    """Build the hard-criteria VBox widget for the stabilisation UI."""
    widgets = []
    lb = ipywidgets.Label("Hard criteria:")
    widgets.append(lb)
    if stabil_calc.capabilities['std']:
        # sl_stdf = ipywidgets.FloatLogSlider(value=stabil_calc.stdf_max, base=10, min=-2, max=4, step=0.1, description='CI F. [Hz]')
        sl_stdf = ipywidgets.FloatSlider(value=stabil_calc.stdf_max, min=0, max=43, step=0.1, description='CI F. [Hz]')
        sl_stdf.observe(lambda change: update_fn(stdf_max=float(change['new'])),
                        names='value', type='change')
        widgets.append(sl_stdf)
        # sl_stdd = ipywidgets.FloatLogSlider(value=stabil_calc.stdd_max, base=10, min=-2, max=4, step=0.1, description='CI D. [%]')
        sl_stdd = ipywidgets.FloatSlider(value=stabil_calc.stdd_max, min=0, max=100, step=0.1, description='CI D. [%]')
        sl_stdd.observe(lambda change: update_fn(stdd_max=float(change['new'])),
                        names='value', type='change')
        widgets.append(sl_stdd)
    if stabil_calc.capabilities['d']:
        sl_d_range = ipywidgets.FloatRangeSlider(value=stabil_calc.d_range, min=0, max=20, step=0.1, description='Damping range [%]')
        sl_d_range.observe(lambda change: update_fn(d_range=change['new']),
                          names='value', type='change')
        widgets.append(sl_d_range)
    if stabil_calc.capabilities['msh']:
        sl_mpc = ipywidgets.FloatSlider(value=stabil_calc.mpc_min, min=0, max=1, step=0.01, description='MPC_min')
        sl_mpc.observe(lambda change: update_fn(mpc_min=float(change['new'])),
                        names='value', type='change')
        widgets.append(sl_mpc)
        sl_mpd = ipywidgets.FloatSlider(value=stabil_calc.mpd_max, min=0, max=180, step=1, description='MPD_max [°]')
        sl_mpd.observe(lambda change: update_fn(mpd_max=float(change['new'])),
                        names='value', type='change')
        widgets.append(sl_mpd)
    if stabil_calc.capabilities['mtn']:
        sl_mtn = ipywidgets.FloatSlider(value=stabil_calc.mtn_min, min=0, max=100, step=1, description='MTN_max []')
        widgets.append(sl_mtn)
        # ..TODO:: implement
    if stabil_calc.capabilities['MC']:
        sl_mc = ipywidgets.FloatSlider(value=stabil_calc.MC_min, min=0, max=1, step=0.01, description='MC_min []')
        sl_mc.observe(lambda change: update_fn(MC_min=float(change['new'])),
                        names='value', type='change')
        widgets.append(sl_mc)
    return ipywidgets.VBox(widgets, layout=ipywidgets.Layout(width='350px', border='solid 1px'))


def _setup_stabil_canvas(fig):
    """Attach an ipympl canvas to *fig* and return it."""
    canvas = ipympl.backend_nbagg.Canvas(fig)
    _manager = ipympl.backend_nbagg.FigureManager(canvas, 0)
    canvas.header_visible = False
    canvas.toolbar_position = 'left'
    canvas.footer_visible = False
    canvas.resizable = False
    return canvas


def _build_stabil_view_boxes(stabil_plot, stabil_calc):
    """Build the view, select-mode, and current-mode panel widgets.

    Returns
    -------
    tuple
        ``(viewbox, selectbox, currentbox, cb_stb, cb_all, cb_psd, rbs,
        dd, select_mode_values, current_mode_values)``
    """
    lb = ipywidgets.Label('View')
    cb_stb = ipywidgets.Checkbox(
        value=stabil_plot.stable_plot['plot_stable'].get_visible(),
        description='Stable poles', indent=False, layout=ipywidgets.Layout(width='100px'))
    cb_all = ipywidgets.Checkbox(
        value=stabil_plot.stable_plot['plot_pre'].get_visible(),
        description='All poles', indent=False, layout=ipywidgets.Layout(width='100px'))
    cb_psd = ipywidgets.Checkbox(
        value=stabil_plot.psd_plot[0][0].get_visible() if stabil_plot.psd_plot else False,
        description='Show PSD', indent=False, layout=ipywidgets.Layout(width='100px'))
    rbs = ipywidgets.RadioButtons(
        options=['Stable', 'All', 'Off'], value='Off',
        description='Cursor', layout=ipywidgets.Layout(width='100px'))
    viewbox = ipywidgets.VBox(
        [lb, cb_stb, cb_all, cb_psd, rbs],
        layout=ipywidgets.Layout(width='200px', border='solid 1px'))

    frequencies = [f'{f:1.3f}' for f in stabil_calc.get_frequencies()]
    dd = ipywidgets.Dropdown(
        options=frequencies, value=frequencies[-1] if frequencies else None,
        description='Selected mode:', style={'description_width': '100px'},
        layout=ipywidgets.Layout(width='200px'))
    select_mode_values = ipywidgets.HTMLMath(value='')
    selectbox = ipywidgets.VBox(
        [dd, select_mode_values],
        layout=ipywidgets.Layout(width='230px', border='solid 1px'))

    lb2 = ipywidgets.Label('Current mode:')
    current_mode_values = ipywidgets.HTMLMath(value='')
    currentbox = ipywidgets.VBox(
        [lb2, current_mode_values],
        layout=ipywidgets.Layout(width='230px', border='solid 1px'))

    return viewbox, selectbox, currentbox, cb_stb, cb_all, cb_psd, rbs, dd, select_mode_values, current_mode_values


def _setup_stabil_ui(stabil_plot):
    """Build canvas, controls and cursor for one stabilisation diagram.

    Returns ``(content_vbox, snap_cursor)``.  The log output widget is NOT
    included in the returned vbox — the caller assembles the final layout.
    """
    stabil_calc = stabil_plot.stabil_calc
    stabil_plot.update_stabilization()

    fig = stabil_plot.fig
    ax = stabil_plot.ax
    canvas = _setup_stabil_canvas(fig)

    snap_cursor = SnappingCursor(ax, stabil_calc.masked_frequencies, stabil_calc.order_dummy)

    def refresh_cursor():
        """Re-apply the snapping cursor mask after stabilisation criteria change."""
        if rbs.value == 'Stable':
            snap_cursor.set_mask(stabil_calc.get_stabilization_mask('mask_stable'))
        elif rbs.value == 'All':
            snap_cursor.set_mask(stabil_calc.get_stabilization_mask('mask_pre'))

    def _update(**kwargs):
        stabil_plot.update_stabilization(**kwargs)
        refresh_cursor()

    softbox = _build_stabil_softbox(stabil_calc, _update)
    hardbox = _build_stabil_hardbox(stabil_calc, _update)

    viewbox, selectbox, currentbox, cb_stb, cb_all, cb_psd, rbs, dd, select_mode_values, current_mode_values = \
        _build_stabil_view_boxes(stabil_plot, stabil_calc)
    frequencies = list(dd.options)

    hbox = ipywidgets.HBox([softbox, hardbox, viewbox, selectbox, currentbox],
                           layout=ipywidgets.Layout(justify_content='space-around'))

    cid = [None]  # mutable container so the closure can update it without global

    def toggle_cursor_snap(change):
        if change['new'] == 'Stable':
            snap_cursor.set_mask(stabil_calc.get_stabilization_mask('mask_stable'))
        elif change['new'] == 'All':
            snap_cursor.set_mask(stabil_calc.get_stabilization_mask('mask_pre'))

        if change['new'] == 'Off':
            snap_cursor.horizontal_line.set_visible(False)
            snap_cursor.vertical_line.set_visible(False)
            if cid[0] is not None:
                canvas.mpl_disconnect(cid[0])
                cid[0] = None
        else:
            cid[0] = canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
            snap_cursor.horizontal_line.set_visible(True)
            snap_cursor.vertical_line.set_visible(True)

    def mode_selector_change(index):
        freqs = [f'{f:1.3f}' for f in stabil_calc.get_frequencies()]
        if index in stabil_calc.select_modes:
            current = f'{stabil_calc.masked_frequencies[index[0], index[1]]:1.3f}'
        else:
            current = freqs[0]
        dd.options = freqs
        dd.value = current

    def update_value_view(widget, frequency=None, mode_index=None):
        if frequency is not None:
            selected_indices = stabil_calc.select_modes
            freqs = np.array([stabil_calc.masked_frequencies[index[0], index[1]]
                            for index in selected_indices])
            f_delta = abs(freqs - frequency)
            idx = np.argmin(f_delta)
            mode_index = selected_indices[idx]

        vals = stabil_calc.get_modal_values(mode_index)
        n, f, stdf, d, stdd, mpc, mp, mpd, dmp, _dmpd, mtn, MC, ex_1, ex_2 = vals

        if stabil_calc.capabilities['std']:
            num_blocks = stabil_calc.modal_data.num_blocks
            stdf = scipy.stats.t.ppf(
                0.95, num_blocks) * stdf / np.sqrt(num_blocks)
            stdd = scipy.stats.t.ppf(
                0.95, num_blocks) * stdd / np.sqrt(num_blocks)
            vals = (n, f, stdf, d, stdd, mpc, mp, mpd, dmp, _dmpd, mtn, MC, ex_1, ex_2)

        widget.value = _build_modal_values_html(vals, stabil_calc.capabilities)

    cb_stb.observe(handler=lambda change: stabil_plot.toggle_stable(bool(change['new'])), names='value', type='change')
    cb_all.observe(handler=lambda change: stabil_plot.toggle_all(bool(change['new'])), names='value', type='change')
    cb_psd.observe(handler=lambda change: stabil_plot.plot_sv_psd(bool(change['new'])), names='value', type='change')

    stabil_calc.add_callback('add_mode', mode_selector_change)
    stabil_calc.add_callback('remove_mode', mode_selector_change)

    dd.observe(handler=lambda change: update_value_view(select_mode_values, frequency=float(change['new'])), names='value', type='change')

    snap_cursor.add_callback('show_current_info', lambda mode_index: update_value_view(current_mode_values, mode_index=mode_index))
    snap_cursor.add_callback('mode_selected', stabil_plot.toggle_mode)
    canvas.mpl_connect('button_press_event', snap_cursor.on_button)
    canvas.mpl_connect('resize_event', snap_cursor.update_pix_data)
    ax.callbacks.connect('xlim_changed', snap_cursor.update_pix_data)
    ax.callbacks.connect('ylim_changed', snap_cursor.update_pix_data)
    rbs.observe(handler=toggle_cursor_snap, names='value', type='change')

    rbs.value = 'Stable'

    if frequencies:
        update_value_view(select_mode_values, frequency=float(frequencies[0]))

    dpi = fig.get_dpi()
    height = fig.get_figheight()
    fig.set_size_inches((1360 / dpi, height))

    fig.canvas.draw()
    snap_cursor.update_pix_data(None)

    content_vbox = ipywidgets.VBox(
        [fig.canvas, hbox],
        layout=ipywidgets.Layout(align_items='center', padding="0px 0px 0px 100px", overflow="scroll"))
    return content_vbox, snap_cursor


def _stabil_single_outer(tab_content, handler):
    """Assemble the outer VBox for a single-setup stabilisation display."""
    return ipywidgets.VBox(
        [tab_content, handler.out],
        layout=ipywidgets.Layout(align_items='center', padding="0px 0px 0px 100px", overflow="scroll"))


def _stabil_multi_outer(tab_contents, setup_names, handler):
    """Assemble the outer VBox for a multi-setup stabilisation display."""
    tab = ipywidgets.Tab(children=tab_contents)
    tab.titles = setup_names
    _tab_css = ipywidgets.HTML(
        '<style>'
        '.widget-tab > .p-TabBar, .widget-tab > .lm-TabBar '
        '{ overflow-x: auto; flex-shrink: 0; }'
        '</style>')
    return ipywidgets.VBox(
        [_tab_css, tab, handler.out],
        layout=ipywidgets.Layout(align_items='center'))


def _attach_pyoma_logging(handler):
    """Attach *handler* to every ``pyOMA.*`` logger in the current hierarchy."""
    for logger in [logging.getLogger(n) for n in logging.root.manager.loggerDict]:
        if 'pyOMA.' in logger.name:
            logger.addHandler(handler)


def StabilGUIWeb(stabil_plots, setup_names=None):
    """Display an interactive stabilisation diagram in Jupyter.

    Parameters
    ----------
    stabil_plots : StabilPlot or list of StabilPlot
        A single plot object (original single-setup API) or a list for
        multi-setup analysis.  In the multi-setup case each setup is shown
        in its own tab inside an ``ipywidgets.Tab``.
    setup_names : list of str, optional
        Tab labels used when *stabil_plots* is a list.  Defaults to
        ``['Setup 1', 'Setup 2', ...]``.

    Returns
    -------
    widget : ipywidgets.VBox
        Assembled widget ready for ``display()``.
    cursors : SnappingCursor or list of SnappingCursor
        Cursor object(s) — single cursor for single-plot call, list for
        multi-setup call.
    """
    is_single = not isinstance(stabil_plots, list)
    if is_single:
        stabil_plots = [stabil_plots]
    if setup_names is None:
        setup_names = [f'Setup {i + 1}' for i in range(len(stabil_plots))]

    handler = OutputWidgetHandler()
    handler.out.layout.width = '1360px'
    handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    _attach_pyoma_logging(handler)

    cursors = []
    tab_contents = []
    for sp in stabil_plots:
        content, cursor = _setup_stabil_ui(sp)
        tab_contents.append(content)
        cursors.append(cursor)

    if is_single:
        return _stabil_single_outer(tab_contents[0], handler), cursors[0]

    return _stabil_multi_outer(tab_contents, setup_names, handler), cursors


def _msh_build_optbox(msp):
    """Build the Options checkbox VBox for PlotMSHWeb."""
    lb = ipywidgets.Label(value='Options:')
    cb1 = ipywidgets.Checkbox(value=msp.show_axis, description='Show Axis Arrows',
                              indent=False, layout=ipywidgets.Layout(width='150px', height='30px'))
    cb2 = ipywidgets.Checkbox(value=msp.show_nodes, description='Show Nodes',
                              indent=False, layout=ipywidgets.Layout(width='100px', height='30px'))
    cb3 = ipywidgets.Checkbox(value=msp.show_lines, description='Show Lines',
                              indent=False, layout=ipywidgets.Layout(width='100px', height='30px'))
    cb4 = ipywidgets.Checkbox(value=msp.show_cn_lines, description='Show Connecting Lines',
                              indent=False, layout=ipywidgets.Layout(width='170px', height='30px'))
    cb5 = ipywidgets.Checkbox(value=msp.show_nd_lines, description='Show Non-displaced Lines',
                              indent=False, layout=ipywidgets.Layout(width='180px', height='30px'))
    cb6 = ipywidgets.Checkbox(value=msp.show_parent_childs, description='Show Parent-Child Assignm.',
                              indent=False, layout=ipywidgets.Layout(width='200px', height='30px'))
    cb7 = ipywidgets.Checkbox(value=msp.show_chan_dofs, description='Show Channel-DOF Assignm.',
                              indent=False, layout=ipywidgets.Layout(width='190px', height='30px'))
    optbox = ipywidgets.VBox([lb, cb1, cb2, cb3, cb4, cb5, cb6, cb7],
                             layout=ipywidgets.Layout(border='solid 1px'))
    return optbox, (cb1, cb2, cb3, cb4, cb5, cb6, cb7)


def _msh_build_viewbox(msp):
    """Build the View controls VBox for PlotMSHWeb."""
    lb = ipywidgets.Label(value='View:')
    fse = ipywidgets.FloatSlider(value=msp.subplot.elev, min=-180, max=180, step=1,
                                 description='Elevation', continuous_update=True)
    fsa = ipywidgets.FloatSlider(value=msp.subplot.azim, min=-180, max=180, step=1,
                                 description='Azimuth', continuous_update=True)
    fsr = ipywidgets.FloatSlider(value=msp.subplot.roll, min=-180, max=180, step=1,
                                 description='Roll', continuous_update=True)

    view_buttons = []
    for desc, w in [('X', '30px'), ('Y', '30px'), ('Z', '30px'), ('ISO', '40px')]:
        btn = ipywidgets.Button(description=desc,
                                layout=ipywidgets.Layout(width=w, height='30px'))
        view_buttons.append(btn)
    hbox = ipywidgets.HBox(view_buttons)
    res_btn = ipywidgets.Button(description='Reset')
    viewbox = ipywidgets.VBox([lb, hbox, fse, fsa, fsr, res_btn],
                              layout=ipywidgets.Layout(border='solid 1px'))
    return viewbox, (fse, fsa, fsr), view_buttons, res_btn


def _msh_build_modebox(msp):
    """Build the Mode controls VBox for PlotMSHWeb."""
    lb = ipywidgets.Label(value='Mode:')
    frequencies = [f'{f:1.3f}' for f in msp.get_frequencies()]
    if msp.mode_index is not None:
        current = f'{msp.modal_frequencies[msp.mode_index[0], msp.mode_index[1]]:1.3f}'
        dd = ipywidgets.Dropdown(options=frequencies, value=current)
    else:
        dd = ipywidgets.Dropdown(options=frequencies)
    ft = ipywidgets.FloatText(value=msp.amplitude, description='Amplitude')
    cb = ipywidgets.Checkbox(value=msp.real, description='Real Modeshape',)
    btn_play = ipywidgets.Button(icon='play')
    btn_play.on_click(lambda change: msp.animate())
    btn_stop = ipywidgets.Button(icon='stop')
    btn_stop.on_click(lambda change: msp.stop_ani())
    hbox = ipywidgets.HBox([btn_play, btn_stop])
    reload_btn = ipywidgets.Button(description='Reload Mode Selection', layout={'width':'90%'})
    modebox = ipywidgets.VBox([lb, dd, ft, cb, hbox, reload_btn],
                              layout=ipywidgets.Layout(border='solid 1px'))
    return modebox, dd, ft, cb, reload_btn, frequencies


def _msh_mode_change_text_with_stabil(msp, mode, order, frequency, damping, MPC, MP, MPD):
    """Build mode-info HTML using stabil_calc when available."""
    n, f, stdf, d, stdd, mpc, mp, mpd, dmp, _dmpd, mtn, MC, ex_1, ex_2 = msp.stabil_calc.get_modal_values((order, mode))
    if msp.stabil_calc.capabilities['std']:
        num_blocks = msp.tabil_calc.modal_data.num_blocks
        stdf = scipy.stats.t.ppf(0.95, num_blocks) * stdf / np.sqrt(num_blocks)
        stdd = scipy.stats.t.ppf(0.95, num_blocks) * stdd / np.sqrt(num_blocks)
    vals = (n, f, stdf, d, stdd, mpc, mp, mpd, dmp, _dmpd, mtn, MC, ex_1, ex_2)
    return _build_modal_values_html(vals, msp.stabil_calc.capabilities)


def _msh_mode_change_text_simple(frequency, damping, order, mode, MPC, MP, MPD):
    """Build mode-info HTML without stabil_calc (basic mode info only)."""
    text = f'''
            <table>
              <tr>
                  <td> Frequency [Hz]:</td>
                  <td> {frequency:1.3f} </td>
              </tr>
              <tr>
                  <td> Damping [%]:</td>
                  <td> {damping:1.3f} </td>
              </tr>
              '''
    if order is not None:
        text += f'''
              <tr>
                  <td> Model order:</td>
                  <td> {order} </td>
              </tr>
              '''
    if mode is not None:
        text += f'''
              <tr>
                  <td> Mode number:</td>
                  <td> {mode} </td>
              </tr>
              '''
    if MPC is not None:
        text += f'''
              <tr>
                  <td> MPC [-]:</td>
                  <td> {MPC:1.3f} </td>
              </tr>
              '''
    if MP is not None:
        text += f'''
              <tr>
                  <td> MP  [°]:</td>
                  <td> {MP:1.3f} </td>
              </tr>
              '''
    if MPD is not None:
        text += f'''
              <tr>
                  <td> MPD [-]:</td>
                  <td> {MPD:1.3f} </td>
              </tr>
              '''
    text += '''
            </table>
            '''
    return text


def PlotMSHWeb(msp):
    """Display an interactive 3-D mode-shape viewer in Jupyter.

    Wraps a :class:`~pyOMA.core.PlotMSH.ModeShapePlot` object in an
    ipywidgets layout with controls for mode selection, animation, and
    display options.

    Parameters
    ----------
    msp : ModeShapePlot
        Populated mode-shape plot object.

    Returns
    -------
    ipywidgets.HBox
        Assembled widget ready for ``display()``.
    """

    # setup Figure for display with ipympl
    fig = msp.fig
    _ax = msp.subplot
    canvas = ipympl.backend_nbagg.Canvas(fig)
    # uncommenting  following line breaks display on some windows systems
    # msp.canvas = canvas
    _manager = ipympl.backend_nbagg.FigureManager(canvas, 0)
    canvas.header_visible = False
    canvas.toolbar_position = 'right'
    canvas.footer_visible = False
    canvas.resizable = False

    # reset view
    msp.reset_view()

    # setup logger for output in UI
    logger = logging.getLogger('core.PlotMSH')
    handler = OutputWidgetHandler()
    handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    optbox, (cb1, cb2, cb3, cb4, cb5, cb6, cb7) = _msh_build_optbox(msp)
    viewbox, (fse, fsa, fsr), view_buttons, res_btn = _msh_build_viewbox(msp)
    modebox, dd, ft, cb, reload_btn, _ = _msh_build_modebox(msp)
    if msp.mode_index is not None:
        current = f'{msp.modal_frequencies[msp.mode_index[0], msp.mode_index[1]]:1.3f}'

    # build "Info" box
    lb = ipywidgets.Label(value='Info:')
    html = ipywidgets.HTMLMath(value='')

    infobox = ipywidgets.VBox([lb, html], layout=ipywidgets.Layout(border='solid 1px'))

    # build final layout
    hbox = ipywidgets.HBox([optbox, viewbox, modebox, infobox],
                           layout=ipywidgets.Layout(justify_content='space-around'))
    vbox = ipywidgets.VBox([fig.canvas, ipywidgets.Label(value='Left click to rotate, middle click to pan, right click to zoom.'), hbox, handler.out],
                           layout=ipywidgets.Layout(align_items='center'))

    # define callbacks and other logic
    def observe_opt_btns(b):
        if b:
            cb1.observe(lambda d: msp.refresh_axis(d['new']), names='value', type='change')
            cb2.observe(lambda d: msp.refresh_nodes(d['new']), names='value', type='change')
            cb3.observe(lambda d: msp.refresh_lines(d['new']), names='value', type='change')
            cb4.observe(lambda d: msp.refresh_cn_lines(d['new']), names='value', type='change')
            cb5.observe(lambda d: msp.refresh_nd_lines(d['new']), names='value', type='change')
            cb6.observe(lambda d: msp.refresh_parent_childs(d['new']), names='value', type='change')
            cb7.observe(lambda d: msp.refresh_chan_dofs(d['new']), names='value', type='change')
        else:
            for widget in [cb1, cb2, cb3, cb4, cb5, cb6, cb7]:
                widget.unobserve_all()

    def observe_sliders(b):
        if b:
            fse.observe(handler=change_viewport, names='value', type='change')
            fsa.observe(handler=change_viewport, names='value', type='change')
            fsr.observe(handler=change_viewport, names='value', type='change')
        else:
            for fs in [fse, fsa, fsr]:
                fs.unobserve_all()

    def change_viewport(change):
        if isinstance(change, ipywidgets.Button):
            observe_sliders(False)
            sender = change.description
            if sender == 'X':
                fsa.value, fse.value = 0.0, 0.0
            elif sender == 'Y':
                fsa.value, fse.value = -90.0, 0.0
            elif sender == 'Z':
                fsa.value, fse.value = 0.0, 90.0
            elif sender == 'ISO':
                fsa.value, fse.value = -60.0, 30.0
            msp.change_viewport(sender)
            observe_sliders(True)
        else:
            msp.change_viewport((fse.value, fsa.value, fsr.value))

    def reset_view(self):
        msp.stop_ani()
        observe_opt_btns(False)
        observe_sliders(False)
        cb1.value = True
        cb2.value = True
        cb3.value = True
        cb4.value = True
        cb5.value = True
        cb6.value = False
        cb7.value = False
        fse.value = 30
        fsa.value = -60
        fsr.value = 0
        msp.reset_view()
        observe_opt_btns(True)
        observe_sliders(True)

    def mode_change(current_freq):
        mode, order, frequency, damping, MPC, MP, MPD = msp.change_mode(float(current_freq))
        if msp.stabil_calc is not None:
            text = _msh_mode_change_text_with_stabil(msp, mode, order, frequency, damping, MPC, MP, MPD)
        else:
            text = _msh_mode_change_text_simple(frequency, damping, order, mode, MPC, MP, MPD)
        dd.value = f'{msp.modal_frequencies[msp.mode_index[0], msp.mode_index[1]]:1.3f}'
        html.value = text

    def reload_modes(btn):
        # update Dropdown widget with new frequencies and set it to the current
        cur = dd.value
        freqs = [f'{f:1.3f}' for f in msp.get_frequencies()]
        dd.options = freqs
        if cur in freqs:
            dd.value = cur

    # connect widgets and callbacks
    observe_opt_btns(True)

    for button in view_buttons:
        button.on_click(change_viewport)
    res_btn.on_click(reset_view)

    dd.observe(handler=lambda change: mode_change(float(change['new'])) , names='value', type='change')
    ft.observe(handler=lambda change: msp.change_amplitude(float(change['new'])) , names='value', type='change')
    cb.observe(handler=lambda change: msp.change_part(bool(change['new'])) , names='value', type='change')

    reload_btn.on_click(reload_modes)
    if msp.mode_index is not None:
        mode_change(current)

    return vbox


def _config_read_file(widget, file):
    """Load *file* contents into *widget*, or show a placeholder if missing."""
    if os.path.exists(file):
        with open(file, 'r') as f:
            widget.value = f.read()
    else:
        widget.value = 'File does not exist'


def _config_save_file(widget, file):
    """Write *widget* contents to *file*."""
    with open(file, 'w') as f:
        f.write(widget.value)


def _config_general_box(config_dict):
    """Build the General-tab widget box for :func:`ConfigGUIWeb`.

    Mutates *config_dict* via observer callbacks and on initial method sync.
    """
    method_dict = get_method_dict()
    method = config_dict.get('method', '')
    if method in method_dict.values():
        method_name = [name for name, m in method_dict.items() if method == m][0]
    else:
        method_name = list(method_dict.keys())[0]

    project_dir_widg = ipywidgets.Text(
        value=str(config_dict.get('project_dir', '')), description='Project Directory',
        layout={'width': '800px'}, style={'description_width': '200px'})
    project_dir_widg.observe(
        handler=lambda change: config_dict.update({'project_dir': Path(change['new'])}),
        names='value', type='change')
    setup_dir_widg = ipywidgets.Text(
        value=str(config_dict.get('setup_dir', '')), description='Setup Directory',
        layout={'width': '800px'}, style={'description_width': '200px'})
    setup_dir_widg.observe(
        handler=lambda change: config_dict.update({'setup_dir': Path(change['new'])}),
        names='value', type='change')
    result_dir_widg = ipywidgets.Text(
        value=str(config_dict.get('result_dir', '')), description='Result Directory',
        layout={'width': '800px'}, style={'description_width': '200px'})
    result_dir_widg.observe(
        handler=lambda change: config_dict.update({'result_dir': Path(change['new'])}),
        names='value', type='change')
    meas_file_widg = ipywidgets.Text(
        value=str(config_dict.get('meas_file', '')), description='Measurement File',
        layout={'width': '800px'}, style={'description_width': '200px'})
    meas_file_widg.observe(
        handler=lambda change: config_dict.update({'meas_file': Path(change['new'])}),
        names='value', type='change')
    method_widg = ipywidgets.Dropdown(
        options=list(method_dict.keys()), value=method_name, layout={'width': '800px'})
    method_widg.observe(
        handler=lambda change: config_dict.update({'method': method_dict[change['new']]}),
        names='value', type='change')
    config_dict.update({'method': method_dict[method_name]})
    skip_existing_widg = ipywidgets.Checkbox(
        value=config_dict.get('skip_existing', ''), description='Skip existing results')
    skip_existing_widg.observe(
        handler=lambda change: config_dict.update({'skip_existing': bool(change['new'])}),
        names='value', type='change')
    save_results_widg = ipywidgets.Checkbox(
        value=config_dict.get('save_results', ''), description='Save new results')
    save_results_widg.observe(
        handler=lambda change: config_dict.update({'save_results': bool(change['new'])}),
        names='value', type='change')
    return ipywidgets.VBox([
        project_dir_widg, setup_dir_widg, result_dir_widg, meas_file_widg,
        method_widg, ipywidgets.HBox([skip_existing_widg, save_results_widg])])


def _config_file_editor_box(label, file_path_str, file_key, config_dict,
                             placeholder, read_and_display_fn, save_contents_fn):
    """Build a file-editor VBox (path text + textarea + load/save buttons).

    Parameters
    ----------
    label : str
        Description label for the path Text widget.
    file_path_str : str
        Initial file path string.
    file_key : str
        Key to update in *config_dict* when the path changes.
    config_dict : dict
        Config dictionary mutated by observer callbacks.
    placeholder : str
        Placeholder text for the Textarea widget.
    read_and_display_fn : callable
        Function ``(widget, file_path)`` that loads file contents into widget.
    save_contents_fn : callable
        Function ``(widget, file_path)`` that saves widget contents to file.

    Returns
    -------
    tuple
        ``(vbox, file_widg)`` — assembled VBox and the path Text widget.
    """
    file_widg = ipywidgets.Text(value=file_path_str, description=label,
                                layout={'width':'800px'}, style={'description_width': '200px'})
    file_widg.observe(handler=lambda change: config_dict.update({file_key: Path(change['new'])}),
                      names='value', type='change')
    text_area = ipywidgets.Textarea(placeholder=placeholder,
                                    layout={'width':'800px', 'height':'400px'})
    save_btn = ipywidgets.Button(description='Save')
    load_btn = ipywidgets.Button(description='Load')
    save_btn.on_click(lambda b: save_contents_fn(text_area, file_widg.value))
    load_btn.on_click(lambda b: read_and_display_fn(text_area, file_widg.value))
    load_btn.click()
    btn_box = ipywidgets.HBox([load_btn, save_btn])
    return ipywidgets.VBox([file_widg, text_area, btn_box]), file_widg


def ConfigGUIWeb(config_dict):
    """Display an interactive configuration file editor in Jupyter.

    Renders editable text areas for each configuration file whose path is
    specified in *config_dict*.

    Parameters
    ----------
    config_dict : dict
        Dictionary with optional keys ``'project_dir'``, ``'setup_dir'``,
        ``'result_dir'`` and others mapping to configuration file paths.

    Returns
    -------
    ipywidgets.Widget
        Assembled widget ready for ``display()``.
    """

    nodes_file = config_dict.get('nodes_file', '')
    lines_file = config_dict.get('lines_file', '')
    parent_child_file = config_dict.get('parent_child_file', '')
    setup_info_file = config_dict.get('setup_info_file', '')
    chan_dofs_file = config_dict.get('chan_dofs_file', '')
    oma_conf_file = config_dict.get('oma_conf_file', '')

    tab_contents = ['General', 'Geometry', 'Setup Info', 'Channel-DOF-Assignments', 'OMA Config']

    general_box = _config_general_box(config_dict)

    # Geometry
    geometry_contents = ['Nodes', 'Lines', 'Parent-Child-Assignments']

    nodes_box, _ = _config_file_editor_box(
        'Nodes File', str(nodes_file), 'nodes_file', config_dict,
        'Load nodes file', _config_read_file, _config_save_file)
    lines_box, _ = _config_file_editor_box(
        'Lines File', str(lines_file), 'lines_file', config_dict,
        'Load lines file', _config_read_file, _config_save_file)
    parent_child_box, _ = _config_file_editor_box(
        'Parent-Child Assignments File', str(parent_child_file), 'parent_child_file', config_dict,
        'Load parent child assignments file', _config_read_file, _config_save_file)

    geometry_tab = ipywidgets.Tab()
    geometry_tab.children = [nodes_box, lines_box, parent_child_box]
    geometry_tab.titles = geometry_contents

    # Setup Info
    setup_info_box, _ = _config_file_editor_box(
        'Setup Info File', str(setup_info_file), 'setup_info_file', config_dict,
        'Load setup info file', _config_read_file, _config_save_file)

    # Chan-DOFs
    channel_dof_box, _ = _config_file_editor_box(
        'Channel-DOF-Assignments File', str(chan_dofs_file), 'chan_dofs_file', config_dict,
        'Load channel-DOF-assignments file', _config_read_file, _config_save_file)

    # OMA Config
    oma_conf_box, _ = _config_file_editor_box(
        'Modal Analysis Config File', str(oma_conf_file), 'oma_conf_file', config_dict,
        'Load modal analysis config file', _config_read_file, _config_save_file)

    children = [general_box, geometry_tab, setup_info_box, channel_dof_box, oma_conf_box]

    tab = ipywidgets.Tab()
    tab.children = children
    tab.titles = tab_contents
    return tab
