# C分支：基座搜索BC

2026-10-12更新：当前本体输入已增加6维基座速度，完整观测为32,880维；欧拉角改为−π～π，手指输出初始化与示范一致。当前候选checkpoint、闭环结果及诊断见 [BC_FOLLOWUP_1012.md](BC_FOLLOWUP_1012.md)。下面的首批数据和3轮结果属于旧89维本体版本，其checkpoint不能直接加载到新版本；中间使用旧欧拉角的95维模型也需配合原配置。

## 版本与范围

- 项目：`/remote-home/ums_hanminghao/Projects/Tgrasp_C`，分支：`experiment-C`。
- 从主分支 `b054dc0` 创建；基线提交 `9b6d0c8`。
- E3保留两通道 `32×32×16` 全局voxel，关闭局部voxel；实现仍保留，可通过 `tactile.local_voxel.enabled` 开启。
- 观测共32,874维：本体89、触觉14、触觉估计相对位置3、全局voxel 32,768。沿用CNN、Transformer和GRU；不增加真实物体位置输入。
- 主项目、B项目及其训练没有修改。C的RL episode仍为1600，rollout仍为32；本次没有启动长期PPO训练。

## 示范如何生成

- 先保持原始掌心朝向，下降到搜索高度，再进入蛇形扫描。
- 以14片传感器中心的平均位置作为扫描参考，覆盖物体随机范围 `[-0.25, 0.25]²`；不是直接让掌心中心覆盖这个范围。
- 扫描行距10 cm，六条往返扫描线；传感器平均目标高度0.64 m。每段使用平滑加减速，最高速度约0.075 m/s，并在环境之间加入小幅速度变化。
- 手指保持接近初始张开姿态，关节目标加入±0.025 rad的小扰动；不规划闭合抓取。
- 平移采用位置/速度反馈加阻尼前馈，旋转采用姿态反馈，转换成现有的归一化力/力矩动作。仿真正常推进，不通过修改手的位置播放路径。
- 实测当前物理参数下的阻尼较大，控制参数为位置增益4000、速度增益80、阻尼前馈2650；这些是当前仿真的经验参数，不是现实硬件参数。
- 默认把物体移至工作区外，采集无物体完整扫描。桌面、手指碰撞、触觉过滤和voxel更新均照常运行。
- `--bc_with_objects` 保留随机物体，用于检查接触发现能力。触觉判定仍使用当前传感器过滤，并不等于碰撞对象身份认证。

## 数据与BC

每条 `trajectory_XXXX.npz` 保存：

| 字段 | 含义 |
|---|---|
| `observations` | 动作执行前的完整策略观测，float16存储，训练转float32 |
| `actions` | 实际执行的24维归一化动作，只有前6维参加BC损失 |
| `bc_mask` | 本轮此前尚未接触，允许搜索BC；导致首次接触的动作保留 |
| `next_contact` | 执行动作后的触觉标记，不是“停止动作”标签 |
| `base_state`、`sensor_positions` | 动作前的掌心状态与传感器位置，用于核查 |
| `target_position`、`tracking_error` | 规划位置和执行后的跟踪误差 |
| `orientation_error`、`region_coverage` | 姿态偏差和当前九宫格覆盖统计 |

- 连续保存初始化后60步稳定过程及完整扫描，GRU可以从轨迹起点重放。
- BC只学习基座动作均值，不监督手指动作、不训练critic或动作噪声头。
- 训练时不打乱轨迹内部顺序；GRU每条轨迹清零，每32步截断反向传播，整条轨迹累积梯度后再更新。
- 留出最后一条完整轨迹验证，保存验证误差最低的 `model_0.pt`；可用现有PPO的 `--model_dir` 加载。
- 数据中保留手指状态不代表已经学会所有手型下的搜索。当前仅有小幅张开姿态扰动，不能保证对大幅闭合手型泛化。
- `--bc_checkpoint` 只让BC网络控制基座，手指仍由示范姿态控制。这是基座闭环验证，不是完整抓取策略评估；这类记录禁止作为专家BC数据训练。

## 首批采集结果

数据目录：`logs/bc_search_1011/serpentine_v1/`。

| 指标 | 结果 |
|---|---:|
| 完整轨迹数 | 8 |
| 状态—动作样本数 | 38,493 |
| 每条长度，含60步稳定过程 | 4,718–4,909步 |
| 位置跟踪RMSE | 0.93–0.97 mm |
| 无物体示范中的触觉触发 | 0 |
| 最终九宫格平均覆盖率 | 91.2%–94.7% |

完整扫描约79–82秒，当前1600步约26.7秒。现有episode不足以完成这套保守扫描，不能因BC学会轨迹就认为1600步内可搜索完全部区域。当前覆盖率是原项目的体素统计，不代表真实物体发现率。

数据已检查有限值、维度和时间对齐：观测中的上一动作与前一条实际动作一致，最大差异约0.000244，来自float16存储；空场触觉及位置估计输入均为零。

随机物体检查：`logs/bc_search_1011/object_check_v1/`，seed 43，16个环境。前1600步检测到接触8/16（50%），完整扫描检测到14/16（87.5%）。这是规划控制器的小样本接触结果，不是BC策略的结果，也不是抓取成功率。带物体的轨迹仅用于诊断，没有混入本次空场BC训练。

完整数据上的短程BC：7条轨迹训练、1条验证，共3个epoch。归一化基座动作的验证MSE为：

| 初始化 | 第1轮 | 第2轮 | 第3轮 |
|---:|---:|---:|---:|
| 0.06122 | 0.03705 | 0.02377 | 0.01676 |

输出：`logs/bc_search_1011/pretrain_v1/model_0.pt`。这里只证明训练与验证链路有效。

短程闭环结果：seed 44，2个环境，60步初始化加500步扫描；位置跟踪RMSE为20.8–21.1 cm，探索覆盖为零。3轮预训练尚未学会及时下降并复现扫描，不能把这个checkpoint当作已经可用的搜索策略。后续需要充分BC训练并复查闭环表现，离线MSE下降并不充分。

PPO兼容性检查：8个环境、1次iteration，成功加载上述权重并完成更新和日志导出；进程在结束时触发原项目的Isaac Gym析构段错误，退出码139。未修改主训练入口的退出流程，也未启动长期RL实验。

## 运行方式

先进入C项目，并使用已有Tgrasp环境：

```bash
cd /remote-home/ums_hanminghao/Projects/Tgrasp_C/dexgrasp
conda activate Tgrasp
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

采集新数据；输出目录必须不存在，避免覆盖旧数据：

```bash
OMP_NUM_THREADS=4 python collect_search_bc.py --bc_output logs/bc_search_new --task ShadowHandGraspDexRep --algo ppo1 --cfg_env cfg/shadow_hand_grasp_dexrep_E3.yaml --cfg_train cfg/ppo1/config.yaml --num_envs 8 --seed 42 --sim_device cuda:1 --rl_device cuda:1 --headless
```

BC预训练，GPU编号按空闲情况调整：

```bash
python pretrain_search_bc.py --dataset logs/bc_search_1011/serpentine_v1 --output logs/bc_pretrain_new --device cuda:1 --epochs 10
```

基座闭环检查：

```bash
OMP_NUM_THREADS=4 python collect_search_bc.py --bc_output logs/bc_closed_loop_new --bc_checkpoint logs/bc_pretrain_new/model_0.pt --task ShadowHandGraspDexRep --algo ppo1 --cfg_env cfg/shadow_hand_grasp_dexrep_E3.yaml --cfg_train cfg/ppo1/config.yaml --num_envs 2 --seed 44 --sim_device cuda:1 --rl_device cuda:1 --headless
```

后续PPO可以增加 `--model_dir logs/bc_pretrain_new/model_0.pt`。本次只提供BC初始化，不增加PPO训练期间的辅助BC损失。初次接入PPO时仍需检查手指动作及基座采样噪声：它们没有被搜索BC训练，不能把离线误差下降等同于完整策略成功。

## 文件与运行说明

- `utils/search_demonstration.py`：蛇形路径、平滑采样和力/力矩控制器。
- `collect_search_bc.py`：仿真采集、数据校验、轨迹图；元数据保存在 `metadata.json`，配置在 `resolved_config.yaml`。
- `pretrain_search_bc.py`：连续轨迹BC、独立验证轨迹、兼容PPO的权重文件。
- 大体积数据和checkpoint保存在Git忽略的 `logs/` 中，不上传GitHub。
- 当前Isaac Gym在仿真销毁阶段有段错误；采集程序仅在数据、元数据和图表完整写入、CUDA同步成功后直接退出，避免该析构错误。运行中异常仍会报错退出，不会被当作成功。
