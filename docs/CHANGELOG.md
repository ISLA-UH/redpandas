## RedVox Pandas (RedPandas) Version History

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



