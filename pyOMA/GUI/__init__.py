# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2015-2026  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.
import logging
import os
import sys
logging.basicConfig(stream=sys.stdout)

# The pyvista backend renders through pyvistaqt, which embeds VTK in a native X
# window.  Under a Wayland Qt session that aborts the process with a BadWindow
# X error, so route Qt through Xwayland (xcb) before the first QApplication is
# created.  Importing any pyOMA.GUI submodule runs this package __init__ first,
# so this is the single place the guard needs to live.  Harmless on pure-X
# sessions; a user-set QT_QPA_PLATFORM is kept.
if sys.platform == 'linux' and os.environ.get('WAYLAND_DISPLAY'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

# from . import HelpersGUI
# from . import JupyterGUI
# from . import PlotMSHGUI
# from . import StabilGUI
