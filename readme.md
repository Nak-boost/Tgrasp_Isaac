# DexRep for Dexterous Grasping (Based on IsaacGym)
This repository contains the official code for "**DexRepNet++: Learning Dexterous Robotic Manipulation With Geometric and Spatial Hand-Object Representations**" _(T-RO 2026)_. It demonstrates the use of **DexRep** in the Isaac simulator for robotic grasping tasks.

[Project Page](https://lqts.github.io/DexRepNet/) | [Paper](https://arxiv.org/abs/2303.09806) | [Video](https://www.bilibili.com/video/BV1bP411b7jh/?spm_id_from=333.999.0.0)

## Dependencies
- Create a conda environment:
    ```shell
    conda create -n dexrep_isaac python==3.8
    conda activate dexrep_isaac
    ```
- Install PyTorch:
    ```shell
    pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
    ```
- Install IsaacGym:

    1. Download [IsaacGym](https://developer.nvidia.com/isaac-gym/download) 
    2. Extract the downloaded files to the main directory of the project
    3. Use the following command to install IsaacGym:  
    ```shell
    pip install -e path/to/isaacgym/python
    ```
- Install DexRep:
    ```shell
    pip install -e .
    ```
- Install PyTorch3D:
    ```shell
    git clone https://github.com/facebookresearch/pytorch3d.git
    cd pytorch3d
    pip install -e .
    ```

## Running the Scripts
We provide two tasks: **ShadowHandGraspDexRep** and **ShadowHandGraspDexRepDexgrasp**. These can be found in the `dexgrasp/tasks` folder. The former uses objects from `GRAB`, while the latter uses the same object settings as [UniDexGrasp](https://github.com/PKU-EPIC/UniDexGrasp/tree/main).

### For _ShadowHandGraspDexRep_
```shell
cd dexgrasp
python train.py --task=ShadowHandGraspDexRep --algo=ppo1 --seed=0 --rl_device=cuda:0 --sim_device=cuda:0 --logdir=logs/dexrep --headless
```

### For _ShadowHandGraspDexRepDexgrasp_

For this task, additional objects from UniDexGrasp are required. You can download the object set `meshdatav3_scaled.tar.xz` from the [website](https://mirrors.pku.edu.cn/dl-release/UniDexGrasp_CVPR2023/dexgrasp_policy/assets/). After downloading, extract the objects using the following command:
```shell    
tar -xvf meshdatav3_scaled.tar.xz -C assets/
```
The files `dexgrasp/cfg/train_set_modify.yaml` and `dexgrasp/cfg/test_set_modify.yaml` provide the object lists for training and testing.

Then, run the following command to train the model:
```shell
cd dexgrasp
python train.py --task=ShadowHandGraspDexRepDexgrasp --algo=ppo1 --seed=0 --rl_device=cuda:0 --sim_device=cuda:0 --logdir=logs/dexrep_dexgrasp --headless
```

Notes:
- If you want to open the simulator windows, remove the **--headless** flag.
- Additional parameters can be found in **dexgrasp/cfg/shadow_hand_grasp_dexrep.yaml** and **dexgrasp/cfg/shadow_hand_grasp_dexrep_dexgrasp.yaml**.

## Evaluation

To evaluate the model, add the `--test` flag to the training command. The trained models are available in the `log/dexrep` and `log/dexrep_dexgrasp` folders. Use the following commands to evaluate the models:

```shell
cd dexgrasp
python train.py --task=ShadowHandGraspDexRep --algo=ppo1 --seed=0 --rl_device=cuda:0 --sim_device=cuda:0 --logdir=logs/dexrep --test
python train.py --task=ShadowHandGraspDexRepDexgrasp --algo=ppo1 --seed=0 --rl_device=cuda:0 --sim_device=cuda:0 --logdir=logs/dexrep_dexgrasp --test
```

If needed, add the `--headless` flag to close the simulator window.

## Troubleshooting

If you encounter any issues during setup or training, please refer to the following steps:

- Ensure all dependencies are installed correctly.
- Verify the paths in the configuration files.
- Check the compatibility of your hardware with the required software versions.

Note that we will not release the code for the MuJoCo simulator due to the old version of MuJoCo used in our experiments. However, the code for the Isaac simulator is fully functional and can be used to reproduce the results in the paper. And if you want to use **dexrep** in other simulators, you can refer to the code in this repository for implementation details.


For further assistance, you can contact [Qingtao Liu](mailto:l_qingtao@zju.edu.cn) or [Qi Ye](mailto:qi.ye@zju.edu.cn).

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
```bibtex
@ARTICLE{liu2026dexrepnet++,
  author={Liu, Qingtao and Sun, Zhengnan and Cui, Yu and Li, Haoming and Li, Gaofeng and Shao, Lin and Chen, Jiming and Ye, Qi},
  journal={IEEE Transactions on Robotics}, 
  title={DexRepNet++: Learning Dexterous Robotic Manipulation With Geometric and Spatial Hand-Object Representations}, 
  year={2026},
  volume={42},
  number={},
  pages={799-818},
  keywords={Hands;Geometry;Grasping;Robots;Encoding;Handover;Training;Shape;Feature extraction;Visualization;Deep learning in robotics and automation;dexterous manipulation;hand-object representation;reinforcement learning (RL)},
  doi={10.1109/TRO.2026.3651669}}

```

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Acknowledgments

This project is built upon [IsaacGym](https://developer.nvidia.com/isaac-gym) and [UniDexGrasp](https://github.com/PKU-EPIC/UniDexGrasp).