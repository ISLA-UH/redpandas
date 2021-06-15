# RedVox RedPandas

The RedVox RedPandas repository contains routines to streamline preprocessing of [RedVox API 900](https://bitbucket.org/redvoxhi/redvox-protobuf-api/src/master/) 
and [API 1000](https://github.com/RedVoxInc/redvox-api-1000) (API M) data.
The RedPandas pipeline is designed for integrability with other legacy and heterogeneous data sources.

In this library you will find

----
## Table of Contents

<!-- toc -->


- [RedVox RedPandas Installation](#redvox-redpandas-installation)
    - [Requirements for installing RedVox RedPandas](#requirements-for-installing-redvox-redpandas)
    - [Installing and/or upgrading RedVox RedPandas with pip](#installing-and/or-upgrading-RedVox-RedPandas-with-pip)
    - [Verifying the installation](#verifying-the-installation)
- [Using RedVox RedPandas](#using-redvox-redpandas)
    - [Example: Skyfall](#example-skyfall)
        - [Downloading the RedVox Skyfall data](#downloading-the-redvox-skyfall-data)
        - [Running the Skyfall example](#running-the-skyfall-example)
- [Development](#development)
    - [Reporting issues](#reporting-issues)
    - [Version History](#version-history)
    - [License](#license)

<!-- tocstop -->

----

### RedVox RedPandas Installation

This section covers the basics on how to install and upgrade the RedVox RedPandas library.

#### Requirements for installing RedVox RedPandas

The RedPandas library uses the [RedVox Python SDK](https://github.com/RedVoxInc/redvox-python-sdk)

#### Installing and/or upgrading RedVox RedPandas with pip

The RedPandas library can be installed using [pip](https://pip.pypa.io/en/stable/). The pip distribution can be found 
in PyPi at https://pypi.org/project/redvox-pandas/.

You can install/upgrade RedPandas using the following command in your terminal:
```
pip install redvox-pandas --upgrade
```
#### Verifying the installation

You can run the following command to check if the RedPandas library has been installed correctly:
```
pip show redvox-redpandas
```
It should return the name of the module, version, author, author email, license, location of the installed module and requires.

Return to _[Table of Contents](#table-of-contents)_

### Using RedVox RedPandas

This section covers the basics on how to use the RedVox RedPandas library.

#### Example: Skyfall

A balloon hoisted a commercial, off-the-shelf, smartphone to a height of 36 km (around 119,000 feet) and purposely burst
to let the smartphone freefall (hence the name _Skyfall_). As the smartphone fell back to Earth, it recorded its 30 minute 
descent using the [RedVox Infrasound Recorder](https://www.redvoxsound.com/) app. You can find more information about this project at 
(a link to the paper will be added once it is published).

The Skyfall data is a great dataset to showcase the RedPandas library for processing smartphone data.


##### Downloading the RedVox Skyfall data

You will need to download the necessary data to run the Skyfall example. There are three methods to accomplish this:
1) Using the [RedVox Python SDK cloud-download](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/cli#cloud-download-command-details) 
(recommended). Note that you will need to install the [GUI RedVox Python SDK](https://github.com/RedVoxInc/redvox-python-sdk/blob/master/docs/python_sdk/installation.md#installing-optional-dependencies) 
dependecies to use the cloud-download.
2) Using the [RedVox Python SDK Command Line Interface (CLI)](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/cli#data-req-command-details).
3) Using Redvox.io (link)

##### Running the Skyfall example

There are three main steps to run the Skyfall example: setting up the configuration file, preprocessing RedVox data, and
plotting the data.

###### Configuration file

The configuration file is 

###### Preprocessing RedVox data

###### Plotting RedVox data


Return to _[Table of Contents](#table-of-contents)_

### Development

This section covers reporting issues and bugs, version history and license.

#### Reporting issues

Please feel free to submit issues on the [issue tracker](https://github.com/RedVoxInc/redpandas/issues). 

#### Version History

See [CHANGELOG.md](https://github.com/RedVoxInc/redpandas/tree/master/docs/CHANGELOG.md)

#### License

[The Apache License](https://github.com/RedVoxInc/redpandas/tree/master/docs/LICENSE.md)

Return to _[Table of Contents](#table-of-contents)_
