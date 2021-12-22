##########
ts_ess_labjack
##########

``ts_ess_labjack`` provides a data client for the ESS CSC, to communicate with LabJack T4 or T7 modules that read environmental sensors.

`Documentation <https://ts-ess-labjack.lsst.io>`_

The package is compatible with Vera Rubin LSST DM's ``scons`` build system, and the `eups <https://github.com/RobertLuptonTheGood/eups>`_ package management system.
Assuming you have the basic DM stack installed you can do the following, from within the package directory:

* ``setup -r .`` to setup the package and dependencies.
* ``scons`` to build the package and run unit tests.
* ``scons install declare`` to install the package and declare it to eups.
* ``package-docs build`` to build the documentation.
  This requires ``documenteer``; see `building single package docs <https://developer.lsst.io/stack/building-single-package-docs.html>`_ for installation instructions.


This code uses ``pre-commit`` to maintain ``black`` formatting and ``flake8`` and ``mypy`` compliance.
To enable this: run ``pre-commit install`` once.
