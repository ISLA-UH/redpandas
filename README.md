# RedPandas

[![Current version on PyPI](https://img.shields.io/badge/pypi-v1.4.10-blue)](https://pypi.org/project/redvox-pandas/)
[![Python versions supported](https://img.shields.io/badge/python-3.8+%20-blue)]()

### Description

This repository contains routines to streamline preprocessing of [RedVox API 900](https://bitbucket.org/redvoxhi/redvox-protobuf-api/src/master/) 
and [API 1000](https://github.com/RedVoxInc/redvox-api-1000) (API M) data.
The RedVox Pandas (also known as RedPandas) pipeline is designed for integrability with other legacy and heterogeneous data sources.

<p>
<img src="https://github.com/ISLA-UH/redpandas/blob/master/docs/redpandas/img/cover_fig.png?raw=true" alt="Wiggles" width="700">
</p>

### Installation 

To install with pip:
```shell
pip install redvox-pandas
```

More installation instructions can be found in the [RedPandas Installation Documentation](https://github.com/ISLA-UH/redpandas/blob/master/docs/redpandas/installation.md).

### Examples 
A full example can be found in [examples directory](https://github.com/ISLA-UH/redpandas/tree/master/examples/skyfall#examples-skyfall).

### Getting started

Check the [User Documentation](https://github.com/ISLA-UH/redpandas/blob/master/docs/redpandas/using_redpandas.md#how-to-use-redpandas) to get started.

You can find examples on [how to plot audio waveforms](https://redvoxinc.github.io/redvox-examples/04_plot_wiggles.html)
and [audio spectrogram](https://redvoxinc.github.io/redvox-examples/05_plot_spectrogram.html) in [RedVox Examples](https://redvoxinc.github.io/redvox-examples/index.html).

### API Documentation

Check the [API Documentation](https://redvoxinc.github.io/redpandas/) for more details on RedPandas functions.

### Development

- Please feel free to submit issues and bugs on the GitHub [issue tracker](https://github.com/ISLA-UH/redpandas/issues).
- [Version History](https://github.com/ISLA-UH/redpandas/blob/master/docs/CHANGELOG.md)
- [The Apache License](https://github.com/ISLA-UH/redpandas/blob/master/LICENSE)

### Note

RedPandas depends on [RedVox SDK version](https://github.com/RedVoxInc/redvox-python-sdk):
- For RedVox SDK >= 3.1.0, RedPandas >= 1.3.0 
- For RedVox SDK < 3.1.0, RedPandas = 1.2.15
