.. py:currentmodule:: lsst.ts.ess.labjack

.. _lsst.ts.ess.labjack:

###################
lsst.ts.ess.labjack
###################

.. image:: https://img.shields.io/badge/GitHub-gray.svg
    :target: https://github.com/lsst-ts/ts_ess_labjack
.. image:: https://img.shields.io/badge/Jira-gray.svg
    :target: https://jira.lsstcorp.org/issues/?jql=labels+%3D+ts_ess_labjack

Overview
========

This package provides `LabJackDataClient`, data client for the `ESS CSC <ts_ess_csc>`_ that reads environmental data from a `LabJack`_ T4 or T7 module and reports it as one one or more ESS telemetry topics.

`LabJackDataClient` can only write array-based ESS telemetry topics, such as `temperature`_ and `pressure`_.

This data client lives in its own package because it requires a LabJack-specific library.
In order to run an ESS CSC that uses `LabJackDataClient`, all you have to do is install this package.

.. _LabJack: https://labjack.com
.. _temperature: https://ts-xml.lsst.io/sal_interfaces/ESS.html#ess-telemetry-temperature
.. _pressure: https://ts-xml.lsst.io/sal_interfaces/ESS.html#ess-telemetry-pressure
.. _ts_ess_csc: https://ts-ess_csc.lsst.io

.. _lsst.ts.ess.labjack-user_guide:

User Guide
==========

To make `LabJackDataClient` available to the ESS CSC you must install this package and the LabJack software.

* Download the appropriate labjack-ljm installer from https://labjack.com/support/software/installers/ljm
* Install according to these directions: https://labjack.com/support/software/installers/ljm/ljm-installation-instructions
* Install the Python wrapper using ``pip install labjack-ljm``.

Configuration
-------------

`LabJackDataClient` has its own configuration schema provided by the `LabJackDataClient.get_config_schema` class method (`source <https://github.com/lsst-ts/ts_ess_labjack/blob/main/python/lsst/ts/ess/labjack/labjack_data_client.py>`_).

This configuration specifies which LabJack to talk to, which inputs to read, and how to map those inputs to ESS telemetry topics.

.. _lsst.ts.ess.labjack-developer_guide:

Developer Guide
===============

`LabJackDataClient` uses LabJack's own `labjack-ljm software <labjack_software>`_ to communicate with a LabJack T4 or T7.
If this proves to be a problem, it is possible to use modbus instead.

.. _labjack_software: https://labjack.com/software

.. _lsst.ts.ess.labjack-api_reference:

Python API reference
--------------------

.. automodapi:: lsst.ts.ess.labjack
   :no-main-docstr:

.. _lsst.ts.ess.labjack-contributing:

Contributing
------------

``lsst.ts.ess.labjack`` is developed at https://github.com/lsst-ts/ts_ess_labjack.
You can find Jira issues for this module using `labels=ts_ess_labjack <https://jira.lsstcorp.org/issues/?jql=project%3DDM%20AND%20labels%3Dts_ess_labjack>`_.

Version History
===============

.. toctree::
    version_history
    :maxdepth: 1
