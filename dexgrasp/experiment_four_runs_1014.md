# 2026-10-14：四组 E3 实验

四组从头训练，统一使用 seed 42、1024 个环境、每轮采样 32 步、10000 iteration。第 1、2 组 episode 上限为 1600 步；第 3、4 组改为 800 步，旧训练记录已删除并重新开始。全部使用传感器—桌面碰撞屏蔽、取消触觉高度过滤后的版本。

| GPU | 项目 | XY 随机范围 | 全局 voxel | 局部 voxel | 物体数 | 新 voxel / 首次区域访问奖励 |
|---|---|---|---|---|---:|---|
| 0 | Tgrasp | ±0.25 m | 32×32×16 | 开启 | 2 | 0.02 / 0.2 |
| 1 | Tgrasp | ±0.20 m | 32×32×16 | 关闭 | 2 | 0.01 / 0.1 |
| 2 | Tgrasp | ±0.15 m | 32×32×16 | 关闭 | 20 | 0.01 / 0.1 |
| 3 | TGrasp_B | ±0.15 m | 16×16×8 | 开启 | 20 | 0.01 / 0.1 |

## 实现改动

- 主项目补齐 B 已有的 `tactile.local_voxel.enabled` 开关：控制观测维度、局部地图初始化和观测拼接，默认仍开启。
- 编码器根据实际观测分支创建，无需新增网络结构；关闭局部地图时不创建其输入和编码分支。
- 四组参数分别保存在独立 YAML，原 E3 基础配置不替换。
- 第 1 组将新 voxel 奖励 0.01→0.02，首次区域访问奖励 0.1→0.2；区域距离进展系数维持 1.0。
- 新 voxel 奖励和九宫格的 XY 区域随物体随机范围配置缩放，沿用各项目现有的高度层规则。
- 第 4 组使用 B 已有的局部地图裁剪，8 通道、12×12×8 格。它继承粗全局地图的分辨率，不是独立维护的高分辨率地图。

## 共用的 20 个物体

从原项目 `cfg/shadow_hand_grasp_dexrep.yaml` 的物体列表按顺序取前 20 个：

`hammer`、`cubesmall`、`banana`、`stamp`、`cubemedium`、`toruslarge`、`duck`、`spherelarge`、`mouse`、`airplane`、`cylinderlarge`、`stapler`、`flute`、`piggybank`、`alarmclock`、`eyeglasses`、`gamecontroller`、`phone`、`torusmedium`、`knife`。

已核对两个项目对应 URDF 和碰撞网格内容一致；两组物体顺序一致，1024 个环境按相同规则分配（前 4 类各 52 个，其余各 51 个）。

## 配置与运行记录

- 实验 1（Tgrasp）：配置 `cfg/shadow_hand_grasp_dexrep_E3_1014_exp1.yaml`；日志 `logs/tactile_1014_four_experiments/exp1_r025_local_2obj_seed42/`。
- 实验 2（Tgrasp）：配置 `cfg/shadow_hand_grasp_dexrep_E3_1014_exp2.yaml`；日志 `logs/tactile_1014_four_experiments/exp2_r020_no_local_2obj_seed42/`。
- 实验 3（Tgrasp）：配置 `cfg/shadow_hand_grasp_dexrep_E3_1014_exp3.yaml`；日志 `logs/tactile_1014_four_experiments/exp3_r015_no_local_20obj_seed42/`。
- 实验 4（TGrasp_B）：配置 `cfg/shadow_hand_grasp_dexrep_E3_1014_exp4.yaml`；日志 `logs/tactile_1014_four_experiments/exp4_B_r015_local_20obj_seed42/`。

每组日志目录包含 `launch.sh`、完整训练输出、实际配置、源码快照、Git HEAD 与工作区补丁。总清单位于两项目的 `logs/tactile_1014_four_experiments/manifest.json`。

tmux 会话：`TGRASP_E3_1014_FOUR`；窗口：`exp1_gpu0`、`exp2_gpu1`、`exp3_gpu2`、`exp4_gpu3`。查看：`tmux attach -t TGRASP_E3_1014_FOUR`。

这四组用于比较指定方案，并非单变量消融：第 3、4 组除局部地图开关外，全局分辨率、局部地图物理覆盖范围及项目原有探索高度层规则也不同。每组仅一个 seed，最终效果需结合多 seed 复核。
