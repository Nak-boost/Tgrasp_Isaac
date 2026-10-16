# 2026-10-15：四张 GPU 训练安排

全部使用1024个并行环境、32步PPO rollout；新日志位于各项目的 `dexgrasp/logs/tactile_1015_four_experiments/`，原训练日志和权重保留。

| GPU | 项目/实验 | Seed | XY随机范围 | 物体 | Episode | Iteration | 日志子目录 |
|---|---|---:|---|---:|---:|---|---|
| 0 | Tgrasp，实验1调整奖励后续训 | 42 | ±0.25 | 2 | 1600 | 10000→15000，新增5000轮 | exp1_resume_seed42 |
| 1 | Tgrasp，新E2，从头训练 | 42 | ±0.15 | 20 | 800 | 0→10000 | E2_r015_20obj_seed42 |
| 2 | Tgrasp，新E2，从头训练 | 43 | ±0.15 | 20 | 800 | 0→10000 | E2_r015_20obj_seed43 |
| 3 | TGrasp_B，实验4原配置续训 | 42 | ±0.15 | 20 | 800 | 10000→15000，新增5000轮 | exp4_B_resume_seed42 |

## 实验1调整

| 参数 | 原实验1 | 本次 |
|---|---:|---:|
| 新voxel奖励系数 | 0.02 | 0.05 |
| 九宫格首次点亮奖励 | 0.2 | 1.0 |
| 区域距离进展系数 | 1.0 | 2.0 |
| 向上动作奖励系数 | 0.075 | 0.075（保持） |
| 向上辅助监督loss系数 | 0.1 | 0.03 |

沿用现有中心到达判定，不改为面积覆盖率。实际物体抬升、成功奖励及网络结构不变。使用独立配置 `cfg/shadow_hand_grasp_dexrep_E3_1015_exp1_resume.yaml` 和 `cfg/ppo1/config_E3_1015_exp1_resume.yaml`，不修改E3默认配置或原实验1配置。

## 两组E2

配置为 `cfg/shadow_hand_grasp_dexrep_E2_1015_r015_20obj.yaml`。用户已确认采用±0.15、与实验3/4相同的20个物体。沿用新E2：32×32×16全局voxel，无局部voxel；130维×8帧历史，采样步距3，历史MLP＋单个128维持续记忆GRU。其余奖励沿用新E2默认配置。

20个物体按相同顺序分配：hammer、cubesmall、banana、stamp、cubemedium、toruslarge、duck、spherelarge、mouse、airplane、cylinderlarge、stapler、flute、piggybank、alarmclock、eyeglasses、gamecontroller、phone、torusmedium、knife。

## 续训与运行管理

实验1、4分别从各自1014目录中的 `checkpoint/model_10000.pt` 加载。现有checkpoint保存模型权重及向上监督进度，不保存优化器、仿真状态或GRU隐藏状态；续训时后三者重新初始化。本次不改变保存格式。

PPO的 `--max_iterations` 是最终迭代编号，所以续训指定15000才能从10000再执行5000轮。新E2指定10000。

tmux会话：`TGRASP_1015_FOUR`；窗口为 `exp1_gpu0`、`E2_seed42_gpu1`、`E2_seed43_gpu2`、`exp4_gpu3`。查看命令：`tmux attach -t TGRASP_1015_FOUR`。

每个目录保留 `launch.sh`、`train.log`、源码与配置快照、git HEAD和工作区diff。两个项目的日志根目录均保存 `manifest.json`。

此前短测中，训练结束保存权重后曾发生原生退出段错误；本次没有修改训练退出逻辑。结束时需同时核对最后迭代日志及最终权重，不能只凭tmux退出码判断是否完成。
