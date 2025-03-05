# DexRepNet (Based on IsaacGym)
This is the official code for "**DexRepNet: Learning Dexterous Robotic Grasping Network with Geometric and Spatial Hand-Object Representation**" _(IROS 2023)_. This repository demonstrates how to use **DexRep** in the Isaac simulator for grasping tasks.

[Project Page](https://lqts.github.io/DexRepNet/) | [Paper](https://arxiv.org/abs/2303.09806) | [Video](https://www.bilibili.com/video/BV1bP411b7jh/?spm_id_from=333.999.0.0)

- [ ] We will release the **MuJoCo version** used in the original paper before May 2025, including:
    - [ ] Release demonstrations
    - [ ] Release behavior cloning (BC) code
    - [ ] Release reinforcement learning (RL) code
    - [ ] Release evaluation code
    - [ ] Release trained models

## Dependencies
- Create a conda environment
    ```shell
    conda create -n dexrep_isaac python==3.8
    conda activate dexrep_isaac
    ```
- Install torch
    ```shell
    pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
    ```
- Install IsaacGym

    1. Download [isaacgym](https://developer.nvidia.com/isaac-gym/download) 
    2. Extract the downloaded files to the main directory of the project
    3. Use the following commands to install isaacgym  
    ```shell
    cd isaacgym/python
    pip install -e .
    ```
- Install DexRep
    ```shell
    cd dexgrasp
    pip install -e .
    ```
- Install pytorch3d
    ```shell
    git clone https://github.com/facebookresearch/pytorch3d.git
    cd pytorch3d
    pip install -e .
    ```

## Run the scripts
We provide two tasks: **ShadowHandGraspDexRep** and **ShadowHandGraspDexRepDexgrasp**. You can find them in the `dexgrasp/tasks` folder. The former uses objects from `GRAB`, and the latter uses the same object settings as [UniDexGrasp](https://github.com/PKU-EPIC/UniDexGrasp/tree/main).

### For _ShadowHandGraspDexRep_
```shell
cd dexgrasp
python train.py --task=ShadowHandGraspDexRep --algo=ppo1 --seed=0 --rl_device=cuda:0 --sim_device=cuda:0 --logdir=logs/dexrep  -headless
```

### For _ShadowHandGraspDexRepDexgrasp_

For this task, we need to add extra objects from UniDexGrasp. You can download the object set `meshdatav3_scaled.tar.xz` from the [website](https://mirrors.pku.edu.cn/dl-release/UniDexGrasp_CVPR2023/dexgrasp_policy/assets/). After downloading, you can run the following command to extract the objects:
```shell    
tar -xvf meshdatav3_scaled.tar.xz -C assets/
```
`dexgrasp/cfg/train_set_modify.yaml` and `dexgrasp/cfg/test_set_modify.yaml` provide the object list for training and testing.

Then you can run the following command to train the model:
```shell
cd dexgrasp
python train.py --task=ShadowHandGraspDexRepDexgrasp --algo=ppo1 --seed=0 --rl_device=cuda:0 --sim_device=cuda:0 --logdir=logs/dexrep_dexgrasp --headless
```

Notes:
- If you want to open the simulator windows, remove **--headless** 
- More parameters can be found in **dexgrasp/cfg/shadow_hand_grasp_dexrep.yaml** and **dexgrasp/cfg/shadow_hand_grasp_dexrep_dexgrasp.yaml**.

## Evaluation

Add `--test` to the training command to evaluate the model. We release the trained models in the `log/dexrep` and `log/dexrep_dexgrasp` folders. You can run the following command to evaluate the model.

```shell
cd dexgrasp
python train.py --task=ShadowHandGraspDexRep --algo=ppo1 --seed=0 --rl_device=cuda:0 --sim_device=cuda:0 --logdir=logs/dexrep --test
python train.py --task=ShadowHandGraspDexRepDexgrasp --algo=ppo1 --seed=0 --rl_device=cuda:0 --sim_device=cuda:0 --logdir=logs/dexrep_dexgrasp --test
```

If needed, add `--headless` to close the simulator window.

## Troubleshooting

If you encounter any issues during setup or training, please refer to the following steps:

- Ensure all dependencies are installed correctly.
- Verify the paths in the configuration files.
- Check the compatibility of your hardware with the required software versions.

You can also send an email to [Qingtao Liu](mailto:l_qingtao@zju.edu.cn) or [Qi Ye](mailto:qi.ye@zju.edu.cn) for help.

## Bibtex
```bibtex
@inproceedings{liu2023dexrepnet,
title={Dexrepnet: Learning dexterous robotic grasping network with geometric and spatial hand-object representations},
author={Liu, Qingtao and Cui, Yu and Ye, Qi and Sun, Zhengnan and Li, Haoming and Li, Gaofeng and Shao, Lin and Chen, Jiming},
booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
pages={3153--3160},
year={2023},
organization={IEEE}
} 
```

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgments

This project is built upon [IsaacGym](https://developer.nvidia.com/isaac-gym) and [UniDexGrasp](https://github.com/PKU-EPIC/UniDexGrasp).