# RedVox RedPandas

The RedVox RedPandas repository contains routines to streamline preprocessing of [RedVox API 900](https://bitbucket.org/redvoxhi/redvox-protobuf-api/src/master/) 
and [API 1000](https://github.com/RedVoxInc/redvox-api-1000) (API M) data.
The RedPandas pipeline is designed for integrability with other legacy and heterogeneous data sources.

----
## Table of Contents

<!-- toc -->


- [RedVox RedPandas Installation](#redvox-redpandas-installation)
    - [Requirements for installing RedVox RedPandas](#requirements-for-installing-redvox-redpandas)
    - [Installing and/or upgrading RedVox RedPandas with pip](#installing-and/or-upgrading-RedVox-RedPandas-with-pip)
    - [Verifying the installation](#verifying-the-installation)
- [Using RedVox RedPandas](#using-redvox-redpandas)
    - [Basic definitions](#basic-definitions)
    - [Opening RedVox data with RedPandas](#opening-redvox-data-with-redpandas)
    - [Extracting sensor information with RedPandas](#extracting-sensor-information-with-redpandas)
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

#### Basic definitions

Common terms used throughout the RedVox RedPandas Documentation.

_RedVox related terms:_

- _RedVox_: Not the NYC based rock band.

- _RedVox Infrasound Recorder_: A smartphone app that can record audio and other stimuli such as pressure. 
To learn more about the app, click [here](https://www.redvoxsound.com).

- _RedVox Python SDK_: A Software Development Kit (SDK) developed to read, create, edit, and write RedVox files 
(files ending in .rdvxz for [RedVox API 900](https://bitbucket.org/redvoxhi/redvox-protobuf-api/src/master/) 
 files and .rdvxm for [RedVox API 1000](https://github.com/RedVoxInc/redvox-api-1000) files).
For more details, click [here](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk)


_RedPandas related terms:_

- _Station_: a device used to record data, e.g., a smartphone recording infrasound waves using 
[RedVox Infrasound Recorder](https://www.redvoxsound.com/) app.

- _Sensor_: a device that responds to a physical stimulus, e.g., pressure, accelerometer. The units for each available sensor can
be found in [RedVox SDK Sensor Documentation](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/data_window/station#sensor-data-dataframe-access).

- _Epoch_ or _epoch time_: unix time (also referred to as the epoch time), the number of seconds since 1 January 1970. 
The RedPandas' native unit of time is epoch in seconds.

#### Downloading RedVox data

You can collect data with the [RedVox Infrasound Recorder](https://www.redvoxsound.com/) smartphone app.

There are three methods to download the RedVox collected data and/or RedPandas example dataset 
(such as [Skyfall](#example-skyfall)):

1) Using the [RedVox Python SDK cloud-download](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/cli#cloud-download-command-details) 
(recommended). Note that you will need to install the [GUI RedVox Python SDK](https://github.com/RedVoxInc/redvox-python-sdk/blob/master/docs/python_sdk/installation.md#installing-optional-dependencies) 
dependecies to use the cloud-download.
2) Using the [RedVox Python SDK Command Line Interface (CLI)](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/cli#data-req-command-details) 
(recommended if your computer cannot install the GUI dependecies).
3) Using Redvox.io (link)

The downloaded RedVox data will have the formats .rdvxz for [RedVox API 900](https://bitbucket.org/redvoxhi/redvox-protobuf-api/src/master/) 
files and .rdvxm for [RedVox API 1000](https://github.com/RedVoxInc/redvox-api-1000) files (also known as API M).

#### Opening RedVox data with RedPandas

Once the RedVox data has been [downloaded](#downloading-redvox-data), the RedPandas function ``build`` can be used
to extract the data into a compressed pickle (.pkl.pkl.lz4) containing a 
[RedVox DataWindow](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/data_window#data-window).

_Example:_

```python
import redpandas.redpd_datawin as rpd_dw

rpd_dw.build(api_input_directory="path/to/RedVox/data",
             start_epoch_s= ,
             end_epoch_s= ,
             redvox_station_ids= ,
             event_name= ,
             output_directory= ,
             output_filename= ,
             start_buffer_minutes= ,
             end_buffer_minutes= ,
             debug=True)
```

Note that ``build`` will create an output directory ``path/to/file/rpd_files`` based on the path/to/file given in
the ``api_input_directory`` variable. A folder named ``dw``,  
(short for [RedVox DataWindow](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/data_window#data-window))
containing the compressed pickle, and a JSON file will be created inside the ``rpd_files`` folder. 

To work with the compressed pickle with the [RedVox DataWindow](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/data_window#data-window)
data, the following code in your Python environment can be applied:

```python
from redvox.common.data_window import DataWindow

rdvx_data: DataWindow = DataWindow.from_json_file(base_dir="path/to/file/rpd_files",
                                                  file_name="file_name.pkl")
```
For more information on how to use RedVox DataWindow directly, visit 
[Using the Data Window Results](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/data_window#using-the-data-window-results)
and [RedVox Data Window Station](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/data_window/station#station).

#### Extracting sensor information with RedPandas



The available sensors in a station can vary depending on the smartphone and the options available
in the [RedVox Infrasound Recorder](https://www.redvoxsound.com/) app. For a complete list 
of available sensors, visit [RedVox Sensor Data](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/data_window/station#sensor-data). 

``sensor_label = one of: ['audio', 'barometer', 'accelerometer', 'gyroscope', 'magnetometer', 'health', 'location', 'image']``

_Example:_
```python
# RedPandas and RedVox Pyhton SDK 
import redpandas.redpd_build_station as rpd_sta
import redpandas.redpd_scales as rpd_scales
from redvox.common.data_window import DataWindow

# Open saved RedVox DataWindow pickle
rdvx_data: DataWindow = DataWindow.from_json_file(base_dir="path/to/file/rpd_files",
                                                  file_name="file_name.pkl")

# Extract sensor information from stations into a dictionary
sensors_dictionary = rpd_sta.build_station(station= ,
                                           sensor_label="audio",
                                           highpass_type="obspy",
                                           frequency_filter_low= 1./rpd_scales.Slice.T100S,
                                           filter_order= 4) 
```
The ``build_station`` function will return the sensor name, sample rate in Hz, timestamps in epoch second, raw data,
and high passed data (only for sensors: barometer, accelerometer, gyroscope, and magnetometer). 





```python
import pandas as pd
import redpandas.redpd_build_station as rpd_build_sta  # RedPandas

# Make sensor dictionary into a Pandas DataFrame
df_sensors = pd.DataFrame([rpd_build_sta.station_to_dict_from_dw(station=station,
                                                                 sdk_version=rdvx_data.sdk_version,
                                                                 sensor_labels="audio")
                                                for station in rdvx_data.stations])

```



#### Example: Skyfall

A balloon hoisted a commercial, off-the-shelf, smartphone to a height of 36 km (around 119,000 feet) and purposely burst
to let the smartphone freefall (hence the name _Skyfall_). As the smartphone fell back to Earth, it recorded its 30 minute 
descent using the [RedVox Infrasound Recorder](https://www.redvoxsound.com/) app. You can find more information about this project at 
(a link to the paper will be added once it is published).

The Skyfall data is a great dataset to showcase the RedPandas library for processing smartphone data.


##### Downloading the RedVox Skyfall data

You will need to download the necessary data to run the Skyfall example. 

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
