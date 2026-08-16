"""Core contracts: effects, tool interface, policy gate, enforcement, audit.

Deliberately empty of imports: ``config`` imports ``core.paths``, and core
modules import ``config.models``, so keeping this package init side-effect
free keeps the import graph acyclic. Import from submodules directly
(``core.gate``, ``core.enforce``, ...).
"""
