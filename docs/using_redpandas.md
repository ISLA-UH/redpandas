# How to use RedVox RedPandas

This section covers the basics on how to use the RedVox RedPandas library.

----
## Table of Contents

<!-- toc -->



- [Basic definitions](#basic-definitions)
- [Opening RedVox data with RedPandas](#opening-redvox-data-with-redpandas)
- [Extracting sensor information with RedPandas](#extracting-sensor-information-with-redpandas)
- [Plotting with RedPandas](#plotting-with-redpandas)
- [Saving and opening RedPandas parquet files](#saving-and-opening-redpandas-parquet-files)
- [RedPandas example: Skyfall](#redpandas-example-skyfall)
    - [Downloading the RedVox Skyfall data](#downloading-the-redvox-skyfall-data)
    - [Running the Skyfall example](#running-the-skyfall-example)
- [Frequently asked questions (FAQ)](#frequently-asked-questions-faq)

<!-- tocstop -->

----

### Basic definitions

The following terms are common terminology used throughout the RedVox RedPandas Documentation.

_RedVox related terms:_

- _RedVox_: Not the NYC based rock band.

- _RedVox Infrasound Recorder_: A smartphone app that can record audio and other stimuli such as pressure. 
To learn more about the app, click [here](https://www.redvoxsound.com).

- _RedVox Python SDK_: A Software Development Kit (SDK) developed to read, create, edit, and write RedVox files 
(files ending in .rdvxz for [RedVox API 900](https://bitbucket.org/redvoxhi/redvox-protobuf-api/src/master/) 
 files and .rdvxm for [RedVox API 1000](https://github.com/RedVoxInc/redvox-api-1000) files).
For more details, click [here](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk).

_RedPandas related terms:_

- _Station_: a device used to record data, e.g., a smartphone recording infrasound waves using 
[RedVox Infrasound Recorder](https://www.redvoxsound.com/) app. Also a Python class designed in the RedVox Python SDK to
store station and sensor data. For more information on the Station Python class, 
click [here](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/data_window/station).

- _Sensor_: a device that responds to a physical stimulus, e.g., barometer, accelerometer. The units for each available sensor can
be found in [RedVox SDK Sensor Documentation](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/data_window/station#sensor-data-dataframe-access).

- _Epoch_ or _epoch time_: unix time (also referred to as the epoch time), the number of seconds since 1 January 1970. 
The RedPandas' native unit of time is UTC epoch time in seconds.

Return to _[Table of Contents](#table-of-contents)_

### Downloading RedVox data

You can collect data with the [RedVox Infrasound Recorder](https://www.redvoxsound.com/) smartphone app and download it.

There are three methods to download the RedVox collected data and/or RedPandas example datasets 
(such as [Skyfall](#example-skyfall)):

1) Using RedVox Cloud Platform (link) (recommended).
2) Using the [RedVox Python SDK cloud-download](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/cli#cloud-download-command-details). Note that you will need to install the [GUI RedVox Python SDK](https://github.com/RedVoxInc/redvox-python-sdk/blob/master/docs/python_sdk/installation.md#installing-optional-dependencies) 
dependencies to use the cloud-download.
3) Using the [RedVox Python SDK Command Line Interface (CLI)](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/cli#data-req-command-details) 
(recommended if your computer cannot install the GUI dependencies for RedVox Python SDK cloud-download).


The downloaded RedVox data will have the formats .rdvxz for [RedVox API 900](https://bitbucket.org/redvoxhi/redvox-protobuf-api/src/master/) 
files and .rdvxm for [RedVox API 1000](https://github.com/RedVoxInc/redvox-api-1000) files (also known as API M).

Return to _[Table of Contents](#table-of-contents)_

### Opening RedVox data with RedPandas

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

Return to _[Table of Contents](#table-of-contents)_

### Extracting sensor information with RedPandas

The available [sensors](#basic-definitions) in a station can vary depending on the smartphone and the options available
in the [RedVox Infrasound Recorder](https://www.redvoxsound.com/) app. For a complete list 
of available sensors, visit [RedVox Sensor Data](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/data_window/station#sensor-data-dataframe-access). 

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
and high passed data (only for barometer, accelerometer, gyroscope, and magnetometer sensors). 

The extracted sensor data can be structured into a 
[Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) for easier 
manipulation:  

```python
import pandas as pd
import redpandas.redpd_build_station as rpd_build_sta  # RedPandas

# Make sensor dictionary into a Pandas DataFrame
df_sensors = pd.DataFrame([rpd_build_sta.station_to_dict_from_dw(station=station,
                                                                 sdk_version=rdvx_data.sdk_version,
                                                                 sensor_labels="audio")
                                                for station in rdvx_data.stations])

```
You can check the available columns in ``df_sensors`` with 
```python
df_sensors.columns
```

For more information on columns found in RedPandas, column names and their contents, visit [RedVox RedPandas DataFrame Columns](columns_name.md).

Return to _[Table of Contents](#table-of-contents)_

### Plotting with RedPandas




### Saving and opening RedPandas parquet files

Due to their structure, parquet files do not handle nested arrays (i.e., 2d arrays). The barometer, accelerometer, gyroscope and magnetometer sensors data are 
nested arrays in the [RedPandas DataFrame](#extracting-sensor-information-with-redpandas). To save the RedPandas 
DataFrame to a parquet file for later use, you can implement the following approach in your Python environment:

```python
import numpy as np

# Example for barometer sensor
# Create new columns with shape tuple for future unflattening/reshaping
df_sensors[["barometer_wf_raw_ndim",
            "barometer_wf_highpass_ndim",
            "barometer_nans_ndim"]] = \
df_sensors[["barometer_wf_raw",
            "barometer_wf_highpass",
            "barometer_nans"]].applymap(np.shape)

# Change tuples to 1D np.array to save it to parquet (parquet does not save tuples)
df_sensors[["barometer_wf_raw_ndim",
            "barometer_wf_highpass_ndim",
            "barometer_nans_ndim"]] = \
df_sensors[["barometer_wf_raw_ndim",
            "barometer_wf_highpass_ndim",
            "barometer_nans_ndim"]].applymap(np.asarray)

# Flatten each row in waveform (wf) columns
df_sensors[["barometer_wf_raw",
            "barometer_wf_highpass",
            "barometer_nans"]] = \
df_sensors[["barometer_wf_raw",
            "barometer_wf_highpass",
            "barometer_nans"]].applymap(np.ravel)

df_sensors.to_parquet("path/to/output/directory/parquet_file_name.parquet")
```
Note that this approach only applies to the ``"{sensor}_wf_raw"``, ``"{sensor}_wf_highpass"``, ``"{sensor}_wf_nans"`` columns. 
The rest of the columns in the DataFrame [extracted with ``build_station`` function](#extracting-sensor-information-with-redpandas) 
will have a 1d array.


Opening a RedPandas parquet will take an extra step due to the array flattening described above. 
The function ``df_column_unflatten`` recovers the original nested arrays of the sensors.

_Example:_
```python
import pandas as pd
import redpandas.redpd_preprocess as rpd_prep

# Open RedPandas parquet file
df_sensors = pd.read_parquet("path/to/output/directory/parquet_file_name.parquet")

# Unflatten barometer raw data column
rpd_prep.df_column_unflatten(df=df_sensors,
                             col_wf_label="barometer_wf_raw",
                             col_ndim_label="barometer_wf_raw_ndim")

```
Return to _[Table of Contents](#table-of-contents)_

### RedPandas example: Skyfall

The Skyfall data is a great dataset to showcase the RedPandas library for processing smartphone data collected 
with the [RedVox Infrasound Recorder app](https://www.redvoxsound.com).

A balloon hoisted a commercial, off-the-shelf, smartphone to a height of 36 km (around 119,000 feet) and purposely burst
to let the smartphone freefall (hence the name _Skyfall_). As the smartphone fell back to Earth, it recorded its 30 minute 
descent using the [RedVox Infrasound Recorder](https://www.redvoxsound.com/) app. You can find more information about this project at 
(a link to the paper will be added once it is published).


#### Downloading the RedVox Skyfall data

You will need to download the necessary data to run the Skyfall example. 

#### Running the Skyfall example

There are three main steps to run the Skyfall example: setting up the configuration file, preprocessing RedVox data, and
plotting the data.

##### Configuration file

The configuration file is 

##### Preprocessing RedVox data

##### Plotting RedVox data

Return to _[Table of Contents](#table-of-contents)_

### Frequently asked questions (FAQ)

**_I have a RedPandas parquet and when I try to open and plot the barometer / accelerometer / gyroscope / magentometer 
sensors, it breaks._**

One common problem is that you need to unflatten the columns with the barometer, accelerometer, gyroscope, and/or 
magnetometer sensors. Check the section [Saving and opening RedPandas parquet files](#saving-and-opening-redpandas-parquet-files).
An easy way to diagnose if you need to unflatyen the column is by checking that ``df["accelerometer_wf_raw"][0]`` 
prints a 1d numpy array. If that is the case then you need to unflatten those data columns.

**_A function is broken, what do I do?_**

Please feel free to submit issues on the [issue tracker](https://github.com/RedVoxInc/redpandas/issues). 


Return to _[Table of Contents](#table-of-contents)_