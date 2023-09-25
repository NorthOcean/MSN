---
layout: pageWithLink
title: Codes for the Trajectory Prediction Network MSN
subtitle: "Official implementation of the paper \"MSN: Multi-Style Network for Trajectory Prediction\""
# cover-img: /assets/img/2022-07-01/eccv2022.png
gh-repo: NorthOcean/MSN
gh-badge: [star, fork]
---
<!--
 * @Author: Conghao Wong
 * @Date: 2023-02-27 11:24:39
 * @LastEditors: Beihao Xia
 * @LastEditTime: 2023-08-16 10:30:49
 * @Description: file content
 * @Github: https://cocoon2wong.github.io
 * Copyright 2023 Conghao Wong, All Rights Reserved.
-->

![MSN](../assets/img/main.png)

## Get Started

---

You can clone [this repository](https://github.com/northocean/msn) by the following command:

```bash
git clone https://github.com/northocean/msn.git
```

Since the repository contains all the dataset files, this operation may take a longer time.
Or you can just download the zip file from [here](https://codeload.github.com/northocean/msn/zip/refs/heads/master).

## Requirements

The codes are developed with python 3.9.
Additional packages used are included in the `requirements.txt` file.

{: .box-note}
**Note:** Please see [Environment Configuration Guidelines](https://cocoon2wong.github.io/2022-03-03-env/) for more details.

## Dataset Prepare and Process

---

Before training `MSN` on your own dataset, you should add your dataset information to the `datasets` directory.
See [this document](https://cocoon2wong.github.io/Project-Luna/) for details.

## Training

---

The `MSN` contains two main sub-networks, the style proposal sub-network, and the stylized prediction sub-network.
`MSN` forecast agents' multiple trajectories end-to-end.
Considering that most of the loss function terms used to optimize the model work within one sub-network alone, we divide `MSN` into `MSN-a` and `MSN-b`, and apply gradient descent separately for more accessible training.
You can train your own `MSN` weights on your datasets by training each of these two sub-networks.
After training, you can still use it as a regular end-to-end model.

### Stage-1 Subnetwork

It is the style proposal sub-network.
To train the subnetwork, you can pass the --model msna argument to run the `main.py`.
Please refer to section "Args Used" to learn how other args work when training and evaluating.

{: .box-warning}
**Warning:** Do not pass any value to `--load` when training, or it will start evaluating the loaded model.

For a quick start, you can train the subnetwork via the following minimum arguments:

```bash
python main.py --model msna --split sdd
```

### Stage-2 Subnetwork

It is the stylized prediction sub-network.
You can pass the `--model msnb` to run the training with the following minimum arguments:

```bash
python main.py --model msnb --split sdd
```

## Evaluation

---

You can use the following command to evaluate the `MSN` performance end-to-end:

```bash
python main.py \
  --model MKII \
  --loada A_MODEL_PATH \
  --loadb B_MODEL_PATH
```

Where `A_MODEL_PATH` and `B_MODEL_PATH` are the folders of the two sub-networks' weights.

## Pre-Trained Models

---

We have provided our pre-trained model weights to help you quickly evaluate the `MSN` performance.
Our pre-trained models contain model weights trained on `ETH-UCY` by the `leave-one-out` strategy and model weights trained on `SDD` via the dataset split method from [SimAug](https://github.com/JunweiLiang/Multiverse).

{: .box-note}
**Note:** We do not use dataset split files like TrajNet for several reasons.
For example, the frame rate problem in `ETH-eth` sub-dataset, and some of these splits only consider the `pedestrians` in the SDD dataset.
We process the original full-dataset files from these datasets with observations = 3.2 seconds (or 8 frames) and predictions = 4.8 seconds (or 12 frames) to train and test the model.
Detailed process codes are available in `./scripts/add_ethucy_datasets.py`, `./scripts/add_sdd.py`, and `./scripts/sdd_txt2csv.py`.
See details in [this issue](https://github.com/cocoon2wong/Vertical/issues/1).

In order to start validating the effects of our pre-trained models, please follow these steps to prepare dataset files and model weights:

{: .box-warning}
**Warning:** Since the code in this repository does not contain dataset files, you need to execute some of the following commands before you run the code for the first time, otherwise the program will not run correctly.
These commands only need to be executed once.

1. As this repository contains only codes, you may need to download the original dataset files first.
   If you have cloned this repository with `git clone` command, you can download the dataset files by the following command:

   ```bash
   git submodule update --init --recursive
   ```

   <!-- Or you can just download them from [here]([TODO]), then rename the folder as `dataset_original` and put it into the repository's root path. -->

2. You need to process these original dataset files so that they are in a format that our code can read and process.
   You can run the following lines to process the `ETH-UCY` and the `SDD` dataset files:

   ```bash
   cd dataset_original
   python main_ethucysdd.py
   ```

3. Create soft links to these folders:

   ```bash
   cd ..
   ln -s dataset_original/dataset_processed ./
   ln -s dataset_original/dataset_configs ./
   ```

4. After these steps, you can find and download our model weights file and put them into the `./weights` folder.

    <div style="text-align: center;">
        <a class="btn btn-info btn-lg" href="https://github.com/NorthOcean/MSN/releases">⬇️ Download Weights</a>
    </div>

You can start the quick evaluation via the following commands:

```bash
for dataset in eth hotel univ zara1 zara2 sdd
  python main.py \
    --model MKII \
    --loada ./weights/msn/a_${dataset} \
    --loadb ./weights/msn/b_${dataset}
```

### Linear-Interpolation Models

You can also start testing the fast version of our pre-trained models with the argument `--loadb l` instead of the `--loadb MSN_PATH`.
The `--loadb l` will replace the original stage-2 sub-network with the simple linear interpolation method.
Although it may reduce the prediction performance, the model will implement much faster.
You can start testing these linear-interpolation models with the following command:

```bash
python main.py --model MKII --loada $SOME_MODEL_PATH --loadb l
```

Here, `$SOME_MODEL_PATH` is still the path of model weights of the stage-1 sub-networks.

{: .box-note}
**Note:** You can also try to use other different interpolation methods with the following command:
`--loadb speed` or `--loadb newton` instead of `--loadb l`.

### Visualization

If you have the dataset videos and put them into the `videos` folder, you can draw the visualized results by adding the `--draw_reuslts $SOME_VIDEO_CLIP` argument.

{: .box-warning}
**Warning:** You must put videos according to the `video_path` item in the clip's `plist` config file in the `./dataset_configs` folder if you want to draw visualized results on them.

If you want to draw visualized trajectories like what our paper shows, you can add the additional `--draw_distribution 1` argument.
For example, if you have put the video `zara1.mp4` into `./videos/zara1.mp4`, you can draw the MSN results with the following commands:

```bash
python main.py --model MKII \
    --loada ./weights/msn/a_zara1 \
    --loadb ./weights/msn/b_zara1 \
    --draw_results zara1 \
    --draw_distribution 1
```

## Args Used

---

Please specify your customized args when training or testing your model in the following way:

```bash
python main.py --ARG_KEY1 ARG_VALUE2 --ARG_KEY2 ARG_VALUE2 --ARG_KEY3 ARG_VALUE3 ...
```

where `ARG_KEY` is the name of args, and `ARG_VALUE` is the corresponding value.
All args and their usages are listed below.

About the `argtype`:

- Args with argtype=`static` can not be changed once after training.
  When testing the model, the program will not parse these args to overwrite the saved values.
- Args with argtype=`dynamic` can be changed anytime.
  The program will try to first parse inputs from the terminal and then try to load from the saved JSON file.
- Args with argtype=`temporary` will not be saved into JSON files.
  The program will parse these args from the terminal at each time.

<!-- DO NOT CHANGE THIS LINE -->

### Basic args

- `--K_train`: type=`int`, argtype=`static`.
  The number of multiple generations when training. This arg only works for multiple-generation models. 
  The default value is `10`.
- `--K`: type=`int`, argtype=`dynamic`.
  Number of multiple generations when testing. This arg only works for multiple-generation models. 
  The default value is `20`.
- `--anntype`: type=`str`, argtype=`static`.
  Model's predicted annotation type. Can be `'coordinate'` or `'boundingbox'`. 
  The default value is `coordinate`.
- `--auto_dimension`: type=`int`, argtype=`temporary`.
  Choose whether to handle the dimension adaptively. It is now only used for silverballers models that are trained with annotation type `coordinate` but want to test on datasets with annotation type `boundingbox`. 
  The default value is `0`.
- `--batch_size` (short for `-bs`): type=`int`, argtype=`dynamic`.
  Batch size when implementation. 
  The default value is `5000`.
- `--dataset`: type=`str`, argtype=`static`.
  Name of the video dataset to train or evaluate. For example, `'ETH-UCY'` or `'SDD'`. NOTE: DO NOT set this argument manually. 
  The default value is `Unavailable`.
- `--dim`: type=`int`, argtype=`static`.
  Dimension of the `trajectory`. For example, - coordinate (x, y) -> `dim = 2`; - boundingbox (xl, yl, xr, yr) -> `dim = 4`. 
  The default value is `-1`.
- `--draw_distribution` (short for `-dd`): type=`int`, argtype=`temporary`.
  Controls if draw distributions of predictions instead of points. If `draw_distribution == 0`, it will draw results as normal coordinates; If `draw_distribution == 1`, it will draw all results in the distribution way, and points from different time steps will be drawn with different colors. 
  The default value is `0`.
- `--draw_index`: type=`str`, argtype=`temporary`.
  Indexes of test agents to visualize. Numbers are split with `_`. For example, `'123_456_789'`. 
  The default value is `all`.
- `--draw_results` (short for `-dr`): type=`str`, argtype=`temporary`.
  Controls whether to draw visualized results on video frames. Accept the name of one video clip. The codes will first try to load the video file according to the path saved in the `plist` file (saved in `dataset_configs` folder), and if it loads successfully it will draw the results on that video, otherwise it will draw results on a blank canvas. Note that `test_mode` will be set to `'one'` and `force_split` will be set to `draw_results` if `draw_results != 'null'`. 
  The default value is `null`.
- `--draw_videos`: type=`str`, argtype=`temporary`.
  Controls whether draw visualized results on video frames and save as images. Accept the name of one video clip. The codes will first try to load the video according to the path saved in the `plist` file, and if successful it will draw the visualization on the video, otherwise it will draw on a blank canvas. Note that `test_mode` will be set to `'one'` and `force_split` will be set to `draw_videos` if `draw_videos != 'null'`. 
  The default value is `null`.
- `--epochs`: type=`int`, argtype=`static`.
  Maximum training epochs. 
  The default value is `500`.
- `--force_clip`: type=`str`, argtype=`temporary`.
  Force test video clip (ignore the train/test split). It only works when `test_mode` has been set to `one`. 
  The default value is `null`.
- `--force_dataset`: type=`str`, argtype=`temporary`.
  Force test dataset (ignore the train/test split). It only works when `test_mode` has been set to `one`. 
  The default value is `null`.
- `--force_split`: type=`str`, argtype=`temporary`.
  Force test dataset (ignore the train/test split). It only works when `test_mode` has been set to `one`. 
  The default value is `null`.
- `--gpu`: type=`str`, argtype=`temporary`.
  Speed up training or test if you have at least one NVidia GPU. If you have no GPUs or want to run the code on your CPU, please set it to `-1`. NOTE: It only supports training or testing on one GPU. 
  The default value is `0`.
- `--interval`: type=`float`, argtype=`static`.
  Time interval of each sampled trajectory point. 
  The default value is `0.4`.
- `--load` (short for `-l`): type=`str`, argtype=`temporary`.
  Folder to load model (to test). If set to `null`, the training manager will start training new models according to other given args. 
  The default value is `null`.
- `--log_dir`: type=`str`, argtype=`static`.
  Folder to save training logs and model weights. Logs will save at `args.save_base_dir/current_model`. DO NOT change this arg manually. (You can still change the path by passing the `save_base_dir` arg.) 
  The default value is `Unavailable`.
- `--lr`: type=`float`, argtype=`static`.
  Learning rate. 
  The default value is `0.001`.
- `--model_name`: type=`str`, argtype=`static`.
  Customized model name. 
  The default value is `model`.
- `--model`: type=`str`, argtype=`static`.
  The model type used to train or test. 
  The default value is `none`.
- `--obs_frames` (short for `-obs`): type=`int`, argtype=`static`.
  Observation frames for prediction. 
  The default value is `8`.
- `--pmove`: type=`int`, argtype=`static`.
  Index of the reference point when moving trajectories. 
  The default value is `-1`.
- `--pred_frames` (short for `-pred`): type=`int`, argtype=`static`.
  Prediction frames. 
  The default value is `12`.
- `--protate`: type=`float`, argtype=`static`.
  Reference degree when rotating trajectories. 
  The default value is `0.0`.
- `--pscale`: type=`str`, argtype=`static`.
  Index of the reference point when scaling trajectories. 
  The default value is `autoref`.
- `--restore_args`: type=`str`, argtype=`temporary`.
  Path to restore the reference args before training. It will not restore any args if `args.restore_args == 'null'`. 
  The default value is `null`.
- `--restore`: type=`str`, argtype=`temporary`.
  Path to restore the pre-trained weights before training. It will not restore any weights if `args.restore == 'null'`. 
  The default value is `null`.
- `--save_base_dir`: type=`str`, argtype=`static`.
  Base folder to save all running logs. 
  The default value is `./logs`.
- `--split` (short for `-s`): type=`str`, argtype=`static`.
  The dataset split that used to train and evaluate. 
  The default value is `zara1`.
- `--start_test_percent`: type=`float`, argtype=`static`.
  Set when (at which epoch) to start validation during training. The range of this arg should be `0 <= x <= 1`. Validation may start at epoch `args.epochs * args.start_test_percent`. 
  The default value is `0.0`.
- `--step`: type=`int`, argtype=`dynamic`.
  Frame interval for sampling training data. 
  The default value is `1`.
- `--test_mode`: type=`str`, argtype=`temporary`.
  Test settings. It can be `'one'`, `'all'`, or `'mix'`. When setting it to `one`, it will test the model on the `args.force_split` only; When setting it to `all`, it will test on each of the test datasets in `args.split`; When setting it to `mix`, it will test on all test datasets in `args.split` together. 
  The default value is `mix`.
- `--test_step`: type=`int`, argtype=`static`.
  Epoch interval to run validation during training. 
  The default value is `3`.
- `--update_saved_args`: type=`int`, argtype=`temporary`.
  Choose whether to update (overwrite) the saved arg files or not. 
  The default value is `0`.
- `--use_extra_maps`: type=`int`, argtype=`dynamic`.
  Controls if uses the calculated trajectory maps or the given trajectory maps. The training manager will load maps from `./dataset_npz/.../agent1_maps/trajMap.png` if set it to `0`, and load from `./dataset_npz/.../agent1_maps/trajMap_load.png` if set this argument to `1`. 
  The default value is `0`.

### Silverballers args

- `--Kc`: type=`int`, argtype=`static`.
  The number of style channels in `Agent` model. 
  The default value is `20`.
- `--feature_dim`: type=`int`, argtype=`static`.
  Feature dimensions that are used in most layers. 
  The default value is `128`.
- `--key_points`: type=`str`, argtype=`static`.
  A list of key time steps to be predicted in the agent model. For example, `'0_6_11'`. 
  The default value is `0_6_11`.
- `--loada` (short for `-la`): type=`str`, argtype=`temporary`.
  Path to load the first-stage agent model. 
  The default value is `null`.
- `--loadb` (short for `-lb`): type=`str`, argtype=`temporary`.
  Path to load the second-stage handler model. 
  The default value is `null`.
- `--msn_hotel_fix`: type=`int`, argtype=`static`.
  Fix the training process of MSN on ETH-hotel by applying rotation. 
  The default value is `0`.
- `--preprocess`: type=`str`, argtype=`static`.
  Controls whether to run any pre-process before the model inference. It accepts a 3-bit-like string value (like `'111'`): - The first bit: `MOVE` trajectories to (0, 0); - The second bit: re-`SCALE` trajectories; - The third bit: `ROTATE` trajectories. 
  The default value is `111`.

### First-stage silverballers args

- `--depth`: type=`int`, argtype=`static`.
  Depth of the random noise vector. 
  The default value is `16`.

### Second-stage silverballers args

- `--points`: type=`int`, argtype=`static`.
  The number of keypoints accepted in the handler model. 
  The default value is `1`.
<!-- DO NOT CHANGE THIS LINE -->

## Thanks

---

Codes of the Transformers used in this model come from [TensorFlow.org](https://www.tensorflow.org/tutorials/text/transformer);  
Dataset CSV files of ETH-UCY come from [SR-LSTM (CVPR2019) / E-SR-LSTM (TPAMI2020)](https://github.com/zhangpur/SR-LSTM);  
Original dataset annotation files of SDD come from [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/), and its split file comes from [SimAug (ECCV2020)](https://github.com/JunweiLiang/Multiverse);  
All contributors of the repository [Vertical](https://github.com/cocoon2wong/Vertical).

## Contact us

---

Conghao Wong ([@cocoon2wong](https://github.com/cocoon2wong)): conghaowong@icloud.com  
Beihao Xia ([@NorthOcean](https://github.com/NorthOcean)): xbh_hust@hust.edu.cn
