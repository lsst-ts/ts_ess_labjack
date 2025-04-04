.. py:currentmodule:: lsst.ts.ess.labjack

.. _lsst.ts.ess.version_history:

###############
Version History
###############

v1.2.6
------

* Add libusb to and fix the conda recipe.

v1.2.5
------

* Fix the conda recipe.
* Fix acceleromater simulator startup bug.
* Add debug statement to generic LabJack data client class.

v1.2.4
------

* Change parent class from BaseDataClient to BaseReadDataClient for better reconnection control.

v1.2.3
------

* Verify fix for air turbulence speed magnitude calculation.
* Update Jira URL.

v1.2.2
------

* Update the version of ts-conda-build to 0.4 in the conda recipe.

v1.2.1
------

* Retry reading telemetry after a timeout.

v1.2.0
------

* Eliminate TopicReaders and replace with telemetry processors.
* Add support for the Gill 3D Anemometer.
* Remove logging configuration from tests.

v1.1.0
------

* Adopt to common code move to ts_ess_common.
* Code improvements.

v1.0.1
------

* Rename telemetry items for which the topic has the same name.

v1.0.0
------

* Use mock ESS topics from ts_salobj.
  This requires ts_salobj 7.5.
* Delete the local mock topics.

v0.6.8
------

* Simplify the README.

v0.6.7
------

* Remove scons support.
* Git hide egg info and simplify .gitignore.
* Further refinements for ts_pre_commit_config:

  * Stop running pytest linters in ``pyproject.toml``.
  * Remove unused bits from ``conda/meta.yaml``.
  * Remove ``setup.cfg``.

v0.6.6
------

* `LabJackAccelerometerDataClient`: simplify the configuration and code.
  This requires ts_xml 16.
* Use ts_pre_commit_config.
* ``Jenkinsfile``: use the shared library.

v0.6.5
------

* `LabJackAccelerometerDataClient`: fix ``accelerometer`` topic output.
* ``Jenkinsfile``: do not run as root.

v0.6.4
------

* pre-commit: update software versions.

v0.6.3
------

* `LabJackDataClient`: log raw data as it is read, at debug level.
* Configure pre-commit to run isort and mypy.

v0.6.2
------

* `LabJackAccelerometerDataClient`: fix scaling of published PSD data.

v0.6.1
------

* `BaseLabJackDataClient`: stop streaming when connecting and disconnecting.
  This deals with a LabJack left in a strange state.
* `LabJackAccelerometerDataClient`: fix an error in the streaming callback and increase the timeout for starting streaming.

v0.6.0
------

* Improve `LabJackAccelerometerDataClient`:

    * Support multiple accelerometers.
    * Support per-channel offset and scale.
    * Optionally write "raw" acceleration data (in addition to the acceleration PSDs).

* `LabJackDataClient`: change definition of offset parameter to match `LabJackAccelerometerDataClient`.
  Now published data = (raw data - offset) * scale.
  This change will not affect existing use of this data client (as configured in ts_config_ocs), because that has offset=0.

* Improve mock ESS topics:

    * Add `BaseMockEssTopic`, and refactor the other topics to use it.
    * Add `MockESSAccelerometerTopic`

* conda/meta.yaml: update to support multiple versions of Python.

v0.5.0
------

* Add `LabJackAccelerometerDataClient` and `BaseLabJackDataClient`.
* Enable mypy type checking.
* Update type annotations for Python 3.10.

v0.4.0
------

* Build with pyproject.toml.
* Modernize the continuous integration Jenkinsfile
* .pre-commit-config.yaml: update versions.

v0.3.0
------

* `LabJackDataClient`: add DNS name resolution.
* git ignore ``.hypothesis``.
* ``setup.cfg``: add ``asyncio_mode = auto``.

v0.2.1
------

* Fix conda build.

v0.2.0
------

* Update for ts_salobj 7.

v0.1.0
------

* The first release.
