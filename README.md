<!--
 * @Author: Beihao Xia
 * @Date: 2023-03-01 15:58:16
 * @LastEditors: Beihao Xia
 * @LastEditTime: 2023-03-03 10:55:32
 * @Description: file content
 * @Github: https://cocoon2wong.github.io
 * Copyright 2023 Beihao Xia, All Rights Reserved.
-->
# MSN
Official implementation of the paper "MSN: Multi-Style Network for Trajectory Prediction"

## Requirements

The codes are developed with python 3.9.
Additional packages used are included in the `requirements.txt` file.

{: .box-note}
We recommend installing python packages in a virtual environment (like the `conda` environment).
Otherwise, there *COULD* be other problems due to the package version conflicts.

### Create the new python environment

Run the following command to create your new python environment:

```bash
conda create -n NAME python=3.9
```

`NAME` denotes the name of the new environment.

Make sure your `NAME` appears in the screen after running the following command.
If not, please follow the former step once again.

```bash
conda env list
```

### Activate the python environment

If your system is windows, please run the following command:

```bash
activate NAME
```

Or yours is linux/map, please run

```bash
source activate NAME
```

### Install the requirements

Then please run the command to install the required packages listed in `requirements.txt` in your python environment:

```bash
pip install -r requirements.txt
```
