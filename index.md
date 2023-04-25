---
layout: page
title: E-Vertical
subtitle: "Another Vertical View: A Hierarchical Network for Heterogeneous Trajectory Prediction via Spectrums"
# cover-img: /assets/img/2022-03-03/cat.jpeg
# tags: [guidelines]
comments: true
---
<!--
 * @Author: Conghao Wong
 * @Date: 2023-03-21 17:52:21
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2023-04-23 15:28:48
 * @Description: file content
 * @Github: https://cocoon2wong.github.io
 * Copyright 2023 Conghao Wong, All Rights Reserved.
-->

<link rel="stylesheet" type="text/css" href="./assets/css/user.css">

## Information

This paper is an expanded journal version of our conference paper "View vertically: A hierarchical network for trajectory prediction via fourier spectrums" ([homepage](https://cocoon2wong.github.io/Vertical/)).
The paper is publicly available on arXiv.
Click the buttons below for more information.

<div style="text-align: center;">
    <a class="btn btn-colorful btn-lg" href="https://arxiv.org/abs/2304.05106">📖 Paper</a>
    <a class="btn btn-colorful btn-lg" href="./supplementalMaterials">📝 Supplemental Materials</a>
    <br><br>
    <a class="btn btn-colorful btn-lg" href="https://github.com/cocoon2wong/E-Vertical">🛠️ Codes</a>
    <a class="btn btn-colorful btn-lg" href="./howToUse">💡 Codes Guidelines</a>
</div>

## Abstract

<div style="text-align: center;">
    <img style="width: 45%;" src="./assets/img/EV_fig1.png">
    <img style="width: 45%;" src="./assets/img/EV_fig2.png">
</div>

With the fast development of AI-related techniques, the applications of trajectory prediction are no longer limited to easier scenes and trajectories.
More and more heterogeneous trajectories with different representation forms, such as 2D or 3D coordinates, 2D or 3D bounding boxes, and even high-dimensional human skeletons, need to be analyzed and forecasted.
Among these heterogeneous trajectories, interactions between different elements within a frame of trajectory, which we call the "Dimension-Wise Interactions", would be more complex and challenging.
However, most previous approaches focus mainly on a specific form of trajectories, which means these methods could not be used to forecast heterogeneous trajectories, not to mention the dimension-wise interaction.
Besides, previous methods mostly treat trajectory prediction as a normal time sequence generation task, indicating that these methods may require more work to directly analyze agents' behaviors and social interactions at different temporal scales.
In this paper, we bring a new "view" for trajectory prediction to model and forecast trajectories hierarchically according to different frequency portions from the spectral domain to learn to forecast trajectories by considering their frequency responses.
Moreover, we try to expand the current trajectory prediction task by introducing the dimension M from "another view", thus extending its application scenarios to heterogeneous trajectories vertically.
Finally, we adopt the bilinear structure to fuse two factors, including the frequency response and the dimension-wise interaction, to forecast heterogeneous trajectories via spectrums hierarchically in a generic way.
Experiments show that the proposed model outperforms most state-of-the-art methods on ETH-UCY benchmark, Stanford Drone Dataset and nuScenes with heterogeneous trajectories, including 2D coordinates, 2D and 3D bounding boxes.

## Citation

If you find this work useful, it would be grateful to cite our paper!

```bib
@article{wong2023another,
    title={Another Vertical View: A Hierarchical Network for Heterogeneous Trajectory Prediction via Spectrums},
    author={Wong, Conghao and Xia, Beihao and Peng, Qinmu and You, Xinge},
    journal={arXiv preprint arXiv:2304.05106},
    year={2023}
}
```

## Thanks

All contributors of the repository [Vertical](https://github.com/cocoon2wong/Vertical).

## Contact us

Conghao Wong ([@cocoon2wong](https://github.com/cocoon2wong)): conghaowong@icloud.com  
Beihao Xia ([@NorthOcean](https://github.com/NorthOcean)): xbh_hust@hust.edu.cn
