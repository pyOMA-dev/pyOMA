System Identification / Modal Analysis
======================================

.. currentmodule:: pyOMA.core

.. autosummary::
   :recursive:
   :toctree: core
   :template: custom-module-template.rst

   SSICovRef

.. SSIData, VarSSIRef, ERA, PLSCF, PRCE and ModalBase cannot go through the
   autosummary/:recursive: mechanism above: pyOMA/core/__init__.py imports a
   same-named class from each of these modules (e.g. ``from .PLSCF import
   PLSCF``), which shadows the submodule attribute on the ``pyOMA.core``
   package. Sphinx's autosummary resolves ``pyOMA.core.PLSCF`` by first
   trying ``getattr(pyOMA.core, "PLSCF")``, which then finds the class
   instead of the module, so the generated stub documents the wrong object
   (and comes out empty). Documenting these modules directly with
   automodule sidesteps that lookup, since automodule imports the module by
   its fully qualified name instead.

.. automodule:: pyOMA.core.SSIData
   :members:
   :show-inheritance:

.. automodule:: pyOMA.core.VarSSIRef
   :members:
   :show-inheritance:

.. automodule:: pyOMA.core.ERA
   :members:
   :show-inheritance:

.. automodule:: pyOMA.core.PLSCF
   :members:
   :show-inheritance:

.. automodule:: pyOMA.core.PRCE
   :members:
   :show-inheritance:

.. automodule:: pyOMA.core.ModalBase
   :members:
   :show-inheritance:
