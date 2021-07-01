# RedVox RedPandas Installation

This section covers the basics on how to install and upgrade the RedVox RedPandas library.


## Table of Contents

<!-- toc -->

- [Installing and/or upgrading RedVox RedPandas with pip](#installing-and/or-upgrading-RedVox-RedPandas-with-pip)
- [Verifying the installation](#verifying-the-installation)

<!-- tocstop -->



### Installing and/or upgrading RedVox RedPandas with pip

The RedPandas library can be installed using [pip](https://pip.pypa.io/en/stable/). The pip distribution and current RedPandas 
version can be found in PyPI at https://pypi.org/project/redvox-pandas/.

The following command can be used to install and/or upgrade RedPandas:
```shell script
pip install redvox-pandas --upgrade
```

Return to _[Table of Contents](#table-of-contents)_.

#### Verifying the installation

You can run the following command in your terminal to check if the RedPandas library has been installed correctly:
```shell script
pip show redvox-pandas
```
It should return the name of the module, version, summary, home-page, author, author email, license, location of the 
installed module and requires. An example is shown below:

```shell script
Name: redvox-pandas
Version: 1.1.4
Summary: Library to streamline preprocessing of RedVox API 900 and API 1000 data
Home-page: https://github.com/RedVoxInc/redpandas
Author: RedVox
Author-email: dev@redvoxsound.com
License: Apache
Location: /path/where/module/is/installed
Requires: matplotlib, scipy, libquantum, redvox, fastkml, obspy, librosa, pymap3d, numpy, pandas, libwwz
Required-by: 
```

Return to main _[Table of Contents](https://github.com/RedVoxInc/redpandas/blob/master/docs/README.md)_.