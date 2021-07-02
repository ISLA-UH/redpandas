# How to use RedPandas - Advanced data manipulation  

The following subsections showcase features from the RedPandas library once the [RedPandas DataFrame](using_redpandas.md#basic-definitions) 
has been constructed and saved as a [parquet](using_redpandas.md#opening-redvox-data-with-redpandas).  

## Table of Contents

<!-- toc -->

- [Ensonify RedVox data](#ensonify-data)
- [Plot]()
- [Filter]()
- [STFT]()
- [Coherence (?)]()

<!-- tocstop -->
  
  
### Ensonify RedVox data

You can listen to your RedVox dataset

```python
import pandas as pd
import redpandas.redpd_ensonify as rpd_sound

INPUT_DIR = "path/to/redvox/data"
df = pd.read_parquet(INPUT_DIR + "/rpd_files/Redvox_df.parquet")

sensor_column_label_list = ["audio_wf"]  # List of column labels with sensor waveform data
sensor_fs_label_list = ["audio_sample_rate_nominal_hz"]  # List of column labels with sensor sample rates
sensor_name_key_list = ["aud"]  # Optional: list of labels for saving 

rpd_sound.ensonify_sensors_pandas(df=df,
                                  sig_id_label='station_id',
                                  sensor_column_label_list=sensor_column_label_list,
                                  sig_sample_rate_label_list=sensor_fs_label_list,
                                  wav_sample_rate_hz=192000.,  # 8000., 16000., 48000., and 96000. also available
                                  output_wav_directory=INPUT_DIR,
                                  output_wav_filename='A_cool_example',
                                  sensor_name_list=sensor_name_key_list)

#The number of elements in sensor_column_label_list and sig_sample_rate_label_list should be the same.
```
The .wav files will be located in a folder named ``wav`` in the directory provided in ``output_wav_directory``. You can listen to the .wav files in the free and open-sourced app [Audacity](https://www.audacityteam.org/).


Note that for 3 component sensors, e.g., accelerometer, the optional ``sensor_name_list`` parameter should take into account 
the X, Y and Z components. For example, to ensonify the accelerometer ``sensor_column_label_list = ["accelerometer_wf_raw"]``, ``sensor_fs_label_list = accelerometer_sample_rate_hz``, 
and ``sensor_name_key_list = ["Acc_X", "Acc_Y", "Acc_Z"]``. Do not forget to [unflatten](using_redpandas.md#opening-redpandas-parquet-files) 
the sensor data column. 


### Plot waveforms

![](img/metrics.png)


### Filter

### STFT