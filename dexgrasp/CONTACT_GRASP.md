# Tgrasp_C：扫掠触发抓取

本轮任务使用 `ShadowHandContactGrasp`。环境入口为 `cfg/shadow_hand_contact_grasp.yaml`，训练入口为 `cfg/ppo1/config_contact_grasp.yaml`。原 E2/E3 配置不是本轮入口。C 原有代码和未提交修改已备份到 `local_evaluation/contact_grasp_1104_backup/`；Tgrasp 和 TGrasp_B 的源代码没有随此实现修改。

## 控制与训练

1. 重置手和物体，固定手型，物体随机 yaw。手沿 +X 做直线扫掠；不使用 BC。
2. 扫掠期间策略仍读取每步观测、更新 E2 历史和 GRU，但执行的动作来自位置/姿态 PD 控制器，历史记录实际执行指令。
3. 任一触觉传感器越过原有触觉阈值后锁存抓取阶段，下一控制步交给 RL。交接不清空历史或 GRU；短暂失触也不退回脚本。
4. PPO 的策略、价值、熵、KL、向上辅助监督以及优势归一化均排除脚本动作对应的转移。包含首次接触的脚本转移也被排除。历史编码仍可从后续抓取损失学习。
5. 继续使用原抓取奖励与终止条件：有效手指接触、靠近物体进展、向上动作、抬升及保持；连续失触 100 步结束，越出桌面或根部高于桌面 50 cm 结束。没有 voxel 或九宫格探索奖励。

默认总 episode 上限 800 步，**包含扫掠阶段**。脚本最长 360 步，无接触时记录 `sweep_missed` 并重置，不强行生成接触样本。稳定成功仍采用原有抬升 12 cm、有效抓取连续保持 10 步的口径；首次抬升奖励和稳定成功是不同指标。

## 位置、姿态与观测

- 手的根部初始位置为 `(-0.20, 0, 桌面高度 + 0.158)` 米；根部不是指尖或掌心参考点。
- 欧拉角 roll=135°、pitch=0°、yaw=-90°。掌心外法向约为 `(0.707, 0, -0.707)`，即朝前下方 45°；手指略弯、略张开，具体 22 关节默认值见 `contact_grasp.joint_positions`，未列出的关节为零。
- 固定姿态的最低触觉传感器中心约高于桌面 1.35 cm。首先稳定 30 步，再以 0.08 m/s 扫掠，目标行程最多 0.40 m。
- 物体绕 Z 随机旋转，按碰撞网格底部放置于桌面以上 2 mm。其旋转后包围盒前沿位于手根部前方 `4 cm + U(2.5, 6) cm`；横向扰动 ±2 cm。这是近手接触训练分布，不是大区域搜索分布。
- Actor：8×130 维 E2 历史（间隔 3 个控制步，即隔两帧取一帧），以及 3 维触觉估计的物体相对位置；估计无效时该分支为零。训练与推理一致，不给真实物体位置。
- 每帧 130 维：14 个传感器位置 42、二值触觉 14、22 关节位置和速度 44、手参考位置与欧拉角 6、基座控制指令 6、18 受控关节目标误差 18。
- 位置历史以本次扫掠起点 XY、桌面 Z 为参考，保持世界轴方向。平移整段局部轨迹不改变这些输入。历史窗口 MLP 和原 E2 的单个 128 维持久 GRU 共同输入策略；无全局/局部 voxel。
- Critic：保留 77 维特权状态，包括真实物体相对位置、姿态、速度、任务阶段，以及 14 个传感器对 256 点物体表面的距离与法向。

将来大区域遍历可平移此局部控制器。当前测试使用固定 +X 扫掠，尚未验证任意扫掠朝向、邻近多个物体、边缘碰撞、或长距离历史的泛化；更换扫掠方向时需要一致转换观测和动作，不能仅旋转手。

## 验证与运行

固定姿态的两个测试 seed 各 200 回合、20 种物体，每种合计 20 回合，均触发触觉。结果位于 `local_evaluation/contact_sweep_1104/{pose135_h158,final_seed42}/sweep.json`，同目录保存姿态截图。首次接触前物体位移中位数约 1.7 mm，90 分位约 1.3–1.5 cm；少数案例可达到约 8.6 cm。因此这证明本轮分布可产生接管样本，不证明任意物体必然接触，也不代表抓取成功。

本地诊断脚本 `local_evaluation/validate_contact_sweep.py` 同时检查无 voxel 分配、1043 维 actor 输入、接触前奖励屏蔽、交接训练 mask、历史保留和位置历史平移不变性。该脚本及其结果不纳入 Git；训练和可视化不依赖它。短程训练和纯 actor 推理用于检查完整训练/保存/加载流程。

```bash
python launch_geometry_grasp.py --gpus 0 1 --experiments ContactE2 --seeds 42 43 --iterations 10000
```

启动器保存独立代码/配置快照至 `logs/contact_grasp_1104/runtime`，每个 seed 训练后自动用 3 个评估 seed、每次 200 环境，分别进行均值动作和采样动作纯 actor 推理。关注扫掠失触率、触觉接管率、接触后的稳定抓取率及所有回合的稳定抓取率，不能把脚本扫掠接触率当成 RL 学到的搜索能力。

```bash
python train.py --task ShadowHandContactGrasp --algo ppo1 --cfg_env cfg/shadow_hand_contact_grasp.yaml --cfg_train cfg/ppo1/config_contact_grasp.yaml --model_dir logs/contact_grasp_1104/ContactE2_seed42/checkpoint/model_10000.pt --logdir logs/visualize_contact_grasp --sim_device cuda:0 --rl_device cuda:0 --seed 42 --test
```

后续修改源码后，应使用对应实验 runtime 中的脚本和配置复现，而非混用新源码与旧 checkpoint。
