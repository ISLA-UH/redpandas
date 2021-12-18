# How to use RedPandas

This section covers the basics on how to use the RedVox Pandas (RedPandas) library.


## Table of Contents

<!-- toc -->

- [Basic definitions](#basic-definitions)
- [Downloading RedVox data](#downloading-redvox-data)
- [Opening RedVox data with RedPandas](#opening-redvox-data-with-redpandas)
    - [For raw RedVox data (.rdvxz, .rdvxm)](#for-raw-redvox-data-rdvxz-rdvxm)
    - [More options](#more-options)
    - [Exporting RedPandas DataFrame](#exporting-redpandas-dataframe)
- [Opening RedPandas parquet files](#opening-redpandas-parquet-files)
- [Data manipulation with RedPandas](#data-manipulation-with-redpandas)
- [Frequently asked questions (FAQ)](#frequently-asked-questions-faq)

<!-- tocstop -->

### Basic definitions

The following terms are common terminology used throughout the RedPandas Documentation.

_RedVox related terms:_

- _RedVox_: Not the NYC based rock band. RedVox refers to products developed by [RedVox, Inc.](http://nelha.hawaii.gov/our-clients/redvox/)

- _RedVox Infrasound Recorder_: A smartphone app that can record audio and other stimuli such as pressure. 
Visit [RedVox Sound](https://www.redvoxsound.com) to learn more about the app.

- _RedVox Python SDK_: A Software Development Kit (SDK) developed to read, create, edit, and write RedVox files 
(files ending in .rdvxz for [RedVox API 900](https://bitbucket.org/redvoxhi/redvox-protobuf-api/src/master/) 
 files and .rdvxm for [RedVox API 1000](https://github.com/RedVoxInc/redvox-api-1000) files). Visit
  [GitHub RedVox Python SDK](https://github.com/RedVoxInc/redvox-python-sdk) to learn more about the SDK.

_RedPandas related terms:_

- _RedPandas_: short for [RedVox Pandas](https://pypi.org/project/redvox-pandas/).

- _RedPandas DataFrame_: a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
created with the RedPandas library and usually containing RedVox data.

- _Station_: a device used to record data, e.g., a smartphone recording infrasound waves using the
[RedVox Infrasound Recorder](https://www.redvoxsound.com/) app. Also a Python class designed in the RedVox Python SDK to
store station and sensor data. Visit 
[Station Documentation](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/data_window/station)
for more information on the Station Python class. A station has sensors (see below).

- _Sensor_: a device that responds to a physical stimulus, e.g., barometer, accelerometer. The units for each available sensor can
be found in [RedVox SDK Sensor Documentation](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/data_window/station#sensor-data-dataframe-access).
A station should always have audio sensor (and hence audio data).

- _Epoch_ or _epoch time_: unix time (also referred to as the epoch time), the number of seconds since 1 January 1970. 
The RedPandas' native unit of time is UTC epoch time in seconds. For example the epoch time for Thursday, July 1, 2021 at 9:00:00 am
UTC would be 1625130000.


Return to _[Table of Contents](#table-of-contents)_.

### Downloading RedVox data

The data collected with the [RedVox Infrasound Recorder](https://www.redvoxsound.com/) smartphone app can be downloaded 
with one of these four methods:

1) Moving the RedVox files from your smartphone RedVox folder to your computer.
2) Using the [RedVox Cloud Platform](https://beta.redvox.io/#/home).
3) Using the [RedVox Python SDK cloud-download](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/cli#cloud-download-command-details). Note that you will need to install the [GUI RedVox Python SDK](https://github.com/RedVoxInc/redvox-python-sdk/blob/master/docs/python_sdk/installation.md#installing-optional-dependencies) 
dependencies to use the cloud-download.
4) Using the [RedVox Python SDK Command Line Interface (CLI)](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/cli#data-req-command-details) 
(recommended if your computer cannot install the GUI dependencies for RedVox Python SDK cloud-download).


The downloaded RedVox data will have the formats .rdvxz for [RedVox API 900](https://bitbucket.org/redvoxhi/redvox-protobuf-api/src/master/) 
files and .rdvxm for [RedVox API 1000](https://github.com/RedVoxInc/redvox-api-1000) files (also known as API M).

Return to _[Table of Contents](#table-of-contents)_.

### Opening RedVox data with RedPandas

The following subsections explain how to convert RedVox data to a 
[Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) and save as a parquet 
for easy [data manipulation](advance_use_redpandas.md). 
If you want to manipulate RedVox data files directly in your Python environment, visit the [RedVox Python SDK](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk).


#### For raw RedVox data (.rdvxz, .rdvxm)

The easiest method to covert RedVox data to a RedPandas dataframe is by using the function
[redpd_dataframe](https://redvoxinc.github.io/redpandas/redpd_df.html#redpandas.redpd_df.redpd_dataframe). 
This approach is ideal for python newcomers, new RedVox users, and for a superficial first glance at new RedVox data.

_Opening RedVox files (.rdvxz, .rdvxm) example:_

```python
from redpandas.redpd_df import redpd_dataframe
from redvox.common.data_window import DataWindow

"""
Extract RedVox data into a Pandas DataFrame
"""
# Absolute path
INPUT_DIR = "path/to/redvox/data"

#Make a RedVox DataWindow
rdvx_data: DataWindow = DataWindow(input_dir=INPUT_DIR)

# Load RedVox data into a RedVox DataWindow (dw), make a pandas DataFrame and save it as parquet
redpd_dataframe(input_dw=rdvx_data)
```
Note that [redpd_dataframe](https://redvoxinc.github.io/redpandas/redpd_df.html#redpandas.redpd_df.redpd_dataframe) 
will only export audio sensor. For more options, such as type of sensors, visit the [More options](#more-options) section.

For more information on columns found in the [RedPandas DataFrame](#basic-definitions) saved in the parquet, column names 
and their contents, visit [RedVox RedPandas DataFrame Columns](https://github.com/RedVoxInc/redpandas/blob/master/docs/redpandas/columns_name.md#redpandas-dataframe-columns). 
For more information on manipulation of pandas DataFrames, visit [pandas.DataFrame documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

Return to _[Table of Contents](#table-of-contents)_.

#### More options

The function [redpd_dataframe](https://redvoxinc.github.io/redpandas/redpd_df.html#redpandas.redpd_df.redpd_dataframe)
 has a few optional variables to provide more flexibility when creating the RedPandas parquet.

_Opening Redvox files (.rdvxz, .rdvxm) with more options example:_

```python
from redpandas.redpd_df import redpd_dataframe

df_data = redpd_dataframe(input_dir="path/to/redvox/data",  # input_dw_or_path directory where the data is located. Only variable REQUIRED
                          sensor_labels=["audio", "barometer"],  # name of sensors, if None, audio will be loaded
                          highpass_type='obspy',  # type of highpass applied: 'obspy', 'butter', 'rc', default 'obspy'
                          frequency_filter_low=0.01,  # apply highpass filter
                          filter_order=4  #the order of the filter integer. Default is 4
                          )  
```

For the ``sensor_label`` variable, the available [sensors](#basic-definitions) in a station can vary depending on the smartphone and available options
in the [RedVox Infrasound Recorder](https://www.redvoxsound.com/) app. The current available sensors RedPandas works with 
are: ``['audio', 'barometer', 'accelerometer', 'gyroscope', 'magnetometer', 'health', 'location', 'image']`` but note that 
some sensors might not be present in your data. For a complete list of available sensors in the RedVox Python SDK, visit 
[RedVox Sensor Data](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/data_window/station#sensor-data-dataframe-access). 

Return to _[Table of Contents](#table-of-contents)_.

### Exporting RedPandas DataFrame

The function [export_df_to_parquet](https://redvoxinc.github.io/redpandas/redpd_df.html#redpandas.redpd_df.export_df_to_parquet)
exports to parquet for later use.

_Exporting RedPandas example:_
```python
from redpandas.redpd_df import export_df_to_parquet

export_df_to_parquet(df=df_data,
                     output_dir_pqt="path/to/save/parquet",
                     output_filename_pqt="A_Cool_example")
```
Return to _[Table of Contents](#table-of-contents)_.

### Opening RedPandas parquet files

Due to their structure, parquet files do not handle nested arrays (i.e., 2d arrays). The barometer, accelerometer, gyroscope and magnetometer sensors data are 
nested arrays in the RedPandas DataFrame. 

The function [df_unflatten](https://redvoxinc.github.io/redpandas/redpd_preprocess.html#redpandas.redpd_preprocess.df_unflatten)
recovers the original shape of all columns.

_Unflattening RedPandas parquet example:_
```python
import pandas as pd
from redpandas.redpd_preprocess import df_unflatten

# Open RedPandas parquet file
df_sensors = pd.read_parquet("path/to/output/directory/parquet_file_name.parquet")

# Unflatten all data stored in columns
df_unflatten(df=df_sensors)
```

The function [df_column_unflatten](https://redvoxinc.github.io/redpandas/redpd_preprocess.html#redpandas.redpd_preprocess.df_column_unflatten) 
recovers the original nested arrays of the sensors individually. This function is useful if only unflattening a particular column is necessary.

_Unflattening barometer raw data column example:_
```python
import pandas as pd
from redpandas.redpd_preprocess import df_column_unflatten

# Open RedPandas parquet file
df_sensors = pd.read_parquet("path/to/output/directory/parquet_file_name.parquet")

# Unflatten barometer raw data column
df_column_unflatten(df=df_sensors,
                    col_wf_label="barometer_wf_raw",
                    col_ndim_label="barometer_wf_raw_ndim")

```
For more information on the columns expected in ``col_ndim_label``, visit [RedPandas DataFrame Columns](columns_name.md#columns-related-to-parquet-savingopening).

Return to _[Table of Contents](#table-of-contents)_.


### Data manipulation with RedPandas

Visit the [Advanced data manipulation Documentation](advance_use_redpandas.md) to learn more about data manipulation with 
RedPandas such as ensonifying RedVox data, plotting waveforms, and more.

Return to _[Table of Contents](#table-of-contents)_.


### Frequently asked questions (FAQ)

- I have a RedPandas parquet and when I try to open and plot the barometer / accelerometer / gyroscope / magnetometer 
        sensors, it breaks.

    One common problem is that you need to unflatten the columns with the barometer, accelerometer, gyroscope, and/or 
    magnetometer sensors. Check the section [Opening RedPandas parquet files](#opening-redpandas-parquet-files).
    An easy way to diagnose if you need to unflatten the column is by checking that ``df["accelerometer_wf_raw"][0]`` (for example) 
    prints a 1d numpy array. If that is the case then you need to unflatten those data columns.

- A function is broken, what do I do?

    Please feel free to submit issues on the [issue tracker](https://github.com/RedVoxInc/redpandas/issues). 


Return to _[Table of Contents](#table-of-contents)_.

Return to _[main page](https://github.com/RedVoxInc/redpandas#redpandas)_.