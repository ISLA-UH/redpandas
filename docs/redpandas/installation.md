# RedPandas Installation

This section covers the basics on how to install and upgrade the RedVox Pandas (RedPandas) library.


## Table of Contents

<!-- toc -->

- [Installing and/or upgrading RedVox RedPandas with pip](#installing-andor-upgrading-redvox-redpandas-with-pip)
- [Verifying the installation](#verifying-the-installation)

<!-- tocstop -->


### Installing and/or upgrading RedVox RedPandas with pip

The RedPandas library can be installed using [pip](https://pip.pypa.io/en/stable/). The pip distribution and current RedPandas 
version can be found at [PyPI RedPandas](https://pypi.org/project/redvox-pandas/).

The following command can be used to install and/or upgrade RedPandas:
```shell script
pip install redvox-pandas --upgrade
```

Return to _[Table of Contents](#table-of-contents)_.

#### Verifying the installation

To check if the RedPandas library has been installed correctly:
```shell script
pip show redvox-pandas
```
The terminal should return the name of the module, version, summary, home-page, author, author email, license, location of the 
installed module and requires. An example is shown below:

```shell script
Name: redvox-pandas
Version: 1.1.5
Summary: Library to streamline preprocessing of RedVox API 900 and API 1000 data
Home-page: https://github.com/RedVoxInc/redpandas
Author: RedVox
Author-email: dev@redvoxsound.com
License: Apache
Location: /path/where/module/is/installed
Requires: matplotlib, scipy, libquantum, redvox, fastkml, obspy, librosa, pymap3d, numpy, pandas, libwwz
Required-by: 
```
Return to _[Table of Contents](#table-of-contents)_.

Return to _[main page](https://github.com/RedVoxInc/redpandas#redpandas)_.