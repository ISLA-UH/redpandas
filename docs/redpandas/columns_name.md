# RedPandas DataFrame Columns

This section covers the column labels in the RedPandas DataFrame.

## Table of Contents

<!-- toc -->


- [Columns always included in RedPandas](#columns-always-included-in-redpandas)
    - [Columns related to station specs](#columns-related-to-station-specs)
    - [Columns related to audio sensor](#columns-related-to-audio-sensor)
- [Variable columns in RedPandas](#variable-columns-in-redpandas)
    - [Columns related to barometer, accelerometer, gyroscope, and magnetometer sensors](#columns-related-to-barometer-accelerometer-gyroscope-and-magnetometer-sensors)
    - [Columns related to location sensor](#columns-related-to-location-sensor)
    - [Columns related to best location sensor](#columns-related-to-best-location-sensor)
    - [Columns related to health sensor](#columns-related-to-health-sensor)
    - [Columns related to image sensor](#columns-related-to-image-sensor)
    - [Columns related to time synchronization](#columns-related-to-time-synchronization)
    - [Columns related to clock offset and model](#columns-related-to-clock-offset-and-model)
- [Columns related to parquet saving/opening](#columns-related-to-parquet-savingopening)

<!-- tocstop -->


### Columns always included in RedPandas:
#### Columns related to station specs
- ``station_id``: name or ID number of station
- ``station_start_date_epoch_micros``: microseconds since epoch UTC when the station started recording
- ``station_make``: make of the station, e.g., the make would be Samsung for a Samsung S10
- ``station_model``: model of the station, e.g., the model would be SM-G973U1 for a Samsung S10
- ``station_app_version``: version of the [RedVox Infrasound Recorder app](https://www.redvoxsound.com/) at time of data collection
- ``redvox_sdk_version``: version of the 
[RedVox Python SDK](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk) used to create the DataFrame

For more information about station metadata, visit [RedVox Station Metadata documentation](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/data_window/station#station-metadata).

Return to _[Table of Contents](#table-of-contents)_.

#### Columns related to audio sensor

- ``audio_sensor_name``: name of audio sensor
- ``audio_sample_rate_nominal_hz``: nominal sample rate in Hz
- ``audio_sample_rate_corrected_hz``: corrected sample rate in Hz
- ``audio_epoch_s``: audio data timestamps in epoch UTC seconds
- ``audio_wf_raw``: raw audio data
- ``audio_wf``: demeaned audio data
- ``audio_nans``: if there are any, index position of nans in the audio data

Return to _[Table of Contents](#table-of-contents)_.

### Variable columns in RedPandas

Depending on the selected sensors in ``sensor_labels`` in the functions ``build_station`` and 
``station_to_station_to_dict_from_dw``, and the sensors available in the data, the following columns 
might be included in the DataFrame.


#### Columns related to barometer, accelerometer, gyroscope, and magnetometer sensors

- ``{sensor_label}_sensor_name``: name of barometer / accelerometer / gyroscope / magnetometer sensor
- ``{sensor_label}_sample_rate_hz``: sensor sample rate in Hz
- ``{sensor_label}_epoch_s``: sensor data timestamps in epoch UTC seconds
- ``{sensor_label}_wf_raw``: raw sensor data
- ``{sensor_label}_wf_highpass``: highpassed sensor data
- ``{sensor_label}_nans``: if there are any, index position of nans in the sensor data

Return to _[Table of Contents](#table-of-contents)_.

#### Columns related to location sensor

- ``location_sensor_name``: name of location sensor
- ``location_sample_rate_hz``: location sensor sample rate in Hz
- ``location_epoch_s``: location sensor data timestamps in epoch UTC seconds
- ``location_gps_epoch_s``: location sensor gps timestamps in epoch UTC seconds  
- ``location_latitude``: location sensor latitude data in degrees
- ``location_longitude``: location sensor longitude data in degrees
- ``location_altitude``: location sensor altitude data in meters
- ``location_bearing``: location sensor bearing data in degrees
- ``location_speed``: location sensor speed data in meters per second
- ``location_horizontal_accuracy``: in meters
- ``location_vertical_accuracy``: in meters
- ``location_bearing_accuracy``: in degrees
- ``location_speed_accuracy``: meters per second
- ``location_provider``: 

Return to _[Table of Contents](#table-of-contents)_.

#### Columns related to best location sensor

- ``best_location_sensor_name``: name of location sensor
- ``best_location_sample_rate_hz``: location sensor sample rate in Hz
- ``best_location_epoch_s``: location sensor data timestamps in epoch UTC seconds
- ``best_location_gps_epoch_s``: location sensor gps timestamps in epoch UTC seconds  
- ``best_location_latitude``: location sensor latitude data in degrees
- ``best_location_longitude``: location sensor longitude data in degrees
- ``best_location_altitude``: location sensor altitude data in meters
- ``best_location_bearing``: location sensor bearing data in degrees
- ``best_location_speed``: location sensor speed data in meters per second
- ``best_location_horizontal_accuracy``: in meters
- ``best_location_vertical_accuracy``: in meters
- ``best_location_bearing_accuracy``: in degrees
- ``best_location_speed_accuracy``: meters per second
- ``best_location_provider``: 

Return to _[Table of Contents](#table-of-contents)_.

#### Columns related to health sensor

- ``health_sensor_name``: name of health sensor
- ``health_sample_rate_hz``: health sensor sample rate in Hz
- ``health_epoch_s``: health sensor data timestamps in epoch UTC seconds
- ``battery_charge_remaining_per``: percentage of remaining battery charge
- ``battery_current_strength_mA``: battery current strenght in mA
- ``internal_temp_deg_C``: internal temperature in degrees Celsius
- ``network_type``: network type, one of: 'UNKNOWN_NETWORK', 'NO_NETWORK', 'WIFI', 'CELLULAR', 'WIRED', 'Nan'
- ``network_strength_dB``: network strenght in dB
- ``power_state``: power state, one of: 'UNKNOWN_POWER_STATE', 'UNPLUGGED', 'CHARGING', 'CHARGED', 'Nan'
- ``available_ram_byte``: available memory RAM in bytes
- ``available_disk_byte``: available disk space in bytes
- ``cell_service_state``: cell service, one of: 'UNKNOWN', 'EMERGENCY', 'NOMINAL', 'OUT_OF_SERVICE', 'POWER_OFF', 'Nan'

Return to _[Table of Contents](#table-of-contents)_.

#### Columns related to image sensor

- ``image_sensor_name``: name of image sensor
- ``image_sample_rate_hz``: image sensor sample rate in Hz
- ``image_epoch_s``: image sensor data timestamps in epoch UTC seconds
- ``image_bytes``: image data in bytes
- ``image_codec``: image codec

Return to _[Table of Contents](#table-of-contents)_.

#### Columns related to time synchronization

- ``synchronization_epoch_s``: timesync timestamps in epoch UTC seconds
- ``synchronization_latency_ms``: latency of the data in milliseconds
- ``synchronization_offset_ms``: time offset of the data in milliseconds
- ``synchronization_best_offset_ms``: mean offset of the data in milliseconds
- ``synchronization_offset_delta_ms``: difference between offset and best offset in milliseconds
- ``synchronization_number_exchanges``: number of exchangs in timesync

For more information about time synchronization, visit 
[RedVox Timesync and Offset Model Documentation](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/data_window/station#timesync-and-offset-model).

Return to _[Table of Contents](#table-of-contents)_.

#### Columns related to clock offset and model

- ``clock_start_time_epoch_s``: start timestamp of model in microseconds since epoch UTC
- ``clock_best_latency_ms``: mean latency of the data in milliseconds 
- ``clock_best_latency_std_ms``: standard deviation of best latency of the data in milliseconds
- ``clock_offset_s``: the offset at start time
- ``clock_number_bins``: the number of data bins used to create the model, default is 1 if model is empty
- ``clock_number_samples``: the number of samples per data bin; default is 3 (minimum to create a balanced line)
- ``clock_offset_slope``: the slope of the change in offset
- ``clock_offset_model_score``: R2 value of the model; 1.0 is best, 0.0 is worst

For more information about the clock offset and model, visit 
[RedVox Timesync and Offset Model Documentation](https://github.com/RedVoxInc/redvox-python-sdk/tree/master/docs/python_sdk/data_window/station#timesync-and-offset-model).

Return to _[Table of Contents](#table-of-contents)_.

### Columns related to parquet saving/opening

Due to their structure, parquet files do not handle nested arrays (i.e., 2d arrays). The barometer, accelerometer, gyroscope and magnetometer sensors data are 
nested arrays in the RedPandas DataFrame. The columns ``{sensor}_wf_raw`` and ``{sensor}_wf_highpass`` will be flattened arrays
(i.e., 1d arrays) that require the function ``df_column_unflatten`` to unflatten after loading the parquet. The columns ``{sensor}_wf_raw_ndim"``
and ``{sensor}_wf_raw_ndim"`` contain the original column shape.

For more information on how to unflatten the barometer, accelerometer, gyroscope and magnetometer raw and highpass columns, 
visit [Opening RedPandas parquet files](using_redpandas.md#opening-redpandas-parquet-files).


Return to main _[Table of Contents](https://github.com/RedVoxInc/redpandas/blob/master/docs/README.md)_.
