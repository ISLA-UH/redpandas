## RedVox Pandas (RedPandas) Version History

## 1.3.3 (2021-12-21)
- Added df_unflatten which now unflattens all columns in RedPandas DataFrame (assuming the DataFrame was exported using export_df_to_parquet).
- Changes to export_df_to_parquet to automatically flatten TFR.

## 1.3.2 (2021-12-16)
- Now error will appear if the wrong station is provided in wiggles.
- Fix in plot_mic and plot_bar
- Can now export matrixes to df in export_df_to_parquet

## 1.3.1 (2021-12-10)
- Fix for RedVox SDK 3.1.1

## 1.3.0 (2021-12-09)
- Backend changes to adapt for new RedVox SDK 3.1.0
- Added RedPandas version in the RedPandas DataFrame
- Updated requirements to match RedVox SDK 3.1.0

## 1.2.13-15 (2021-10-28)
- Minor changes to mesh plot and decimating functions
- Changes to mesh to avoid memory leaks

## 1.2.11-2 (2021-09-17)
- Added show y ticks value to mesh plot 

## 1.2.9-10 (2021-09-09)
- Fixed bug in wiggles and mesh with ylabels
- Fixed bug with redpd_filter with decimating

## 1.2.7-8 (2021-09-08)
- Fixed bug exporting and opening dataframes with empty values for certain columns
- Fixed wiggles bug with plotting empty values in dataframe

## 1.2.6 (2021-09-01)
- Fixed Skyfall example, added missing file

## 1.2.5 (2021-08-31)
- Fixed bugs in mesh and wiggles plots
- Optimized Skyfall examples

## 1.1.10-1.2.4 (2021-08-03)
- Add support for Python 3.9
- Bump other RedVox deps that have been updated to Python 3.9

## 1.1.7-9 (2021-07-27)

- Fix Skyfall example
- Fix ylabels in plot_wiggles_pandas where magnetometer highpass would not plot

## 1.1.6 (2021-07-27)

- Renaming for clearness: redpd_dw_to_parquet module renamed to redpd_df and redpd_dw_to_parquet function renamed to redpd_dataframe. 
Function redpd_dataframe has been simplified.
- Plots have been moved into folder redpd_plot, new modules: wiggles, mesh and coherence 
- The function redpd_dataframe now returns the constructed dataframe, exporting to pickle and parquet options added.
- Plot_wiggles_pandas now returns a Matplotlib Figure Instance, option to not show the figure added. Plot_wiggles_pandas now also 
checks if input station or column name are valid, and if there is available data. Plot_wiggles_pandas does not break anymore if there 
is no data available.
- In redpd_datawin module, function build eliminated.
- New templates for API documentation to reST stay organized when exported to html.


## 1.1.5 (2021-07-07)

- New and improved plot_wiggle_pandas, plot_mesh_pandas
- Major clean up and docstring documentation added
- More documentation and examples added
- API documentation added
- Now GitHub RedPandas repository public

## 1.1.4 (2021-06-16)

- Fixed mislabelling, more documentation
- More options added to plot_mesh_pandas, skyfall ensonify created

## 1.1.3 (2021-06-16)

- Skyfall example updated, Time Domain Representation plots updated (TDR), added Time Frequency Representation (TFR).
- Updated filter modules for 3 component sensors such as acceleration.
- Added documentation

Return to [_main page_](https://github.com/RedVoxInc/redpandas#redpandas).



