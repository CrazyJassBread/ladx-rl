# 声音作为强化学习观测：2025-2026 年研究调研

检索截止日期：2026-08-27

## 结论摘要

近期研究已经明确证明声音可以作为策略学习的有效输入，但“接入声音”至少包含四种不同问题：

1. **原始音频直接进入策略**：把一小段双声道波形变成 STFT、log-Mel、IPD/ILD 等时频特征，再与图像或其他状态融合，由 PPO/DD-PPO 等算法端到端训练。代表工作是 NLDL 2025 的游戏音频智能体、AAAI 2025 的 ENMuS3 和 CVPR 2026 的 MAGNet。
2. **先做声学感知，再让 RL 使用结构化结果**：先估计声源类别、方位、距离或跟踪不确定性，再把这些中间量输入 actor-critic。ENMuS3、MAGNet 和 2026 年的双基地声呐 AUV 都属于这一类。它通常比直接拼接波形更容易训练，也更容易解释。
3. **声音不是当前模拟器已有的模态，先生成再训练策略**：CoRL 2025 的 MultiGen 用视频条件生成模型为机器人模拟轨迹补齐声音，再训练视觉-音频扩散策略，实现零样本 sim-to-real。
4. **声音作为指令或辅助任务**：ICLR 2025 的 VLAS 直接理解语音指令；ICCV 2025 的 DescRL 在音视频导航策略上增加动作语言描述辅助目标。它们说明声音既可以描述环境物理状态，也可以携带任务语义或提供训练正则化。

最稳健的共同设计不是“把一张频谱图和一帧 RGB 粗暴拼起来”，而是：**独立编码各模态、保留时间记忆、显式处理模态缺失/噪声，并通过消融实验验证声音确实带来额外信息**。

对本仓库最直接的起点是：从 PyBoy 获取每个 emulator frame 的立体声音频，按一个 agent step 聚合为固定长度窗口，先实现 `pixels + compact audio features`，再比较 pixel-only、audio-only 和 audio-visual 三组策略。不要一开始就上大型预训练音频模型；LADX 的合成音频域与自然音频差异很大，轻量 CNN 或事件化 APU 特征更可能成为可靠基线。

## 调研范围与判定标准

本调研优先纳入 2025 年 1 月至 2026 年 8 月正式发表于会议或期刊的论文。核心纳入标准是满足以下至少一项：

- 声音或声学测量进入控制策略的 observation；
- 声音参与具身策略、机器人策略或游戏策略的训练；
- 提出可直接迁移到 RL 环境的音视频融合、记忆或声音模拟方法。

为避免混淆，本文将以下两类工作单独标注：

- **策略学习但不是 RL**：例如行为克隆、扩散策略或 VLA 指令微调；
- **使用 RL 优化多模态推理模型，但不与可交互环境闭环**：方法有借鉴价值，但不能作为“声音 observation 改善控制”的直接证据。

本次检索发现，2025-2026 年高质量工作仍主要集中在三类环境：SoundSpaces/Habitat 室内导航、机器人操作、以及游戏/声呐仿真。直接把音频接入通用 Atari 或复古游戏 RL benchmark 的正式论文仍然很少。

## 核心论文对比

| 论文 | 类型 | 研究环境 | 声音输入 | 与策略的结合 | 训练方式 | 直接结论 |
|---|---|---|---|---|---|---|
| [Towards concurrent real-time audio-aware agents with deep reinforcement learning](https://proceedings.mlr.press/v265/debner25a.html), NLDL 2025 | 直接 RL | Unity + ML-Agents；室内 hide-and-seek 声源定位 | 0.2 s 立体声；STFT；`20 x 42 x 42` 浮点输入 | 3 层 CNN 后接 FC；实验中只给音频，不给图像 | PPO，1000 万 samples | 音频单模态可以学会朝静态发声目标导航；多 listener 音频渲染决定并行训练吞吐量 |
| [Towards Audio-Visual Navigation in Noisy Environments](https://ojs.aaai.org/index.php/AAAI/article/view/33608), AAAI 2025 | 直接 RL + 声学辅助任务 | Matterport3D + 修改后的 SoundSpaces；BeDAViN 多声源/噪声 benchmark | 16 kHz 双耳波形；STFT、IPD、ILD；声源类别与方向 | 声音事件描述器 + RGB-D/pose/action encoder + 多尺度 scene-memory transformer + actor-critic | 端到端导航策略；声源描述器有监督训练 | 多声源和背景噪声下，显式提取目标声音的语义与空间信息优于简单音视频融合 |
| [Embodied Navigation with Auxiliary Task of Action Description Prediction](https://openaccess.thecvf.com/content/ICCV2025/html/Kondoh_Embodied_Navigation_with_Auxiliary_Task_of_Action_Description_Prediction_ICCV_2025_paper.html), ICCV 2025 | RL 辅助任务 | 多种 embodied navigation benchmark；包括 SoundSpaces 的 semantic audio-visual navigation | 沿用音视频导航策略的听觉 observation | 在共享策略表示上增加过去/未来动作的语言描述预测；描述标签由预训练 VLM 蒸馏产生 | 导航 RL + action-description auxiliary loss | 动作描述不是声音编码器本身，但能改善含音频的导航策略并提高可解释性 |
| [The Sound of Simulation: Learning Multimodal Sim-to-Real Robot Policies with Generative Audio](https://proceedings.mlr.press/v305/wang25a.html), CoRL 2025 | 非 RL 策略学习；高度相关 | RoboVerse 倒液体仿真；Kinova Gen3 + Robotiq 2F-85 实机 | 16 kHz log-Mel；AST encoder；真实执行时由末端麦克风采集 | RGB、proprioception、语言命令、audio 分别编码，输入 Diffusion Policy | 运动规划生成 demonstrations；扩散策略模仿学习 | 生成式音频使模拟器获得原本难以物理建模的模态；视觉+音频相对纯视觉平均降低 23.3% NMAE |
| [VLAS: Vision-Language-Action Model with Speech Instructions](https://proceedings.iclr.cc/paper_files/paper/2025/hash/804a1a9de5787e3597b4bb64e1a48ec3-Abstract-Conference.html), ICLR 2025 | 非 RL VLA；语音指令 | CALVIN 与定制化机器人操作 benchmark | 原始语音中的文本语义与说话人 voiceprint | speech encoder 对齐 LLM/VLA；三阶段 speech alignment、speech QA、机器人任务训练；voice RAG 支持个性化 | 多模态监督微调/策略学习 | 不必先 ASR 转文字；非语义声学信息可决定“为谁执行”这类个性化任务 |
| [Semantic Audio-Visual Navigation in Continuous Environments](https://openaccess.thecvf.com/content/CVPR2026/html/Zeng_Semantic_Audio-Visual_Navigation_in_Continuous_Environments_CVPR_2026_paper.html), CVPR 2026 | 直接 RL + 声学辅助任务 | SoundSpaces 2.0 + Habitat/Matterport3D；SAVN-CE 连续 3D 环境 | 每 0.25 s 4000 点、16 kHz 双耳波形；STFT、平均幅度、IPD sin/cos、ILD | 音频/视觉/pose/action 独立编码；目标描述器维护 episodic memory；policy 维护 scene memory | 目标描述器监督学习 + DD-PPO（最多 2.4 亿步） | 连续位置、短时发声、长时间静音下，声音必须与 self-motion 和记忆联合更新；clean 场景 SR 由 SAVi 的 25.6 提升到 37.7 |
| [AUV decision-making in bistatic sonar target tracking via deep reinforcement learning](https://doi.org/10.1016/j.robot.2025.105262), Robotics and Autonomous Systems 197, 2026 | 结构化声学状态进入 RL | 静止声源 + 携线阵接收机的 AUV；含传播、杂波和测量噪声的仿真 | 不把原始声呐波形直接给 policy；输入 tracker 的目标状态估计及不确定性 | 声学信号先经目标跟踪器，PPO 决定 AUV heading | PPO；OSPA 跟踪误差构造 reward | 分层的“声学 estimator -> RL policy”可实时决策，仿真中相对 RHC 平均提升 14.5% 跟踪精度 |

## 各路线如何把声音与研究结合

### 1. 原始波形到策略：短时频谱作为 observation

NLDL 2025 的设计最接近为游戏环境添加音频 observation：

- Unity 以 50 FPS 运行；音频传感器累计 10 个 simulation ticks，形成约 0.2 s 的立体声窗口。
- 对左右声道做 STFT，得到固定大小的二维时频 observation。
- 用 3 层 CNN 编码音频，再由全连接网络输出二维移动方向。
- 为隔离声音价值，实验不提供视觉与 raycast，只使用音频。
- PPO 在中等复杂度地图训练 1000 万 samples，并在空旷、中等、复杂三种室内地图评测。

这一路线实施简单，适合建立第一条可靠 baseline。它的主要风险是频谱窗口和 agent step 的时间尺度不匹配，以及网络通过音量等偶然特征过拟合固定地图或音效。

### 2. 双耳空间线索：IPD/ILD 和声音事件描述器

ENMuS3 与 MAGNet 都不满足于把左右声道当作普通图像，而是显式提取双耳听觉线索：

- **IPD（Interaural Phase Difference）**：左右声道相位差，主要提供方向信息；
- **ILD（Interaural Level Difference）**：左右声道能量差，也可帮助判断方向；
- **声音事件类别**：用于从多个同时发声的对象中识别目标；
- **DOA/距离**：把声学输入转换成可解释的目标方位与距离。

ENMuS3 的 observation embedding 包含声音事件特征、RGB-D、当前 pose 与上一动作，再进入 scene memory。MAGNet 更进一步，在目标静音后用上一动作和当前位置更新目标相对方位，并保留两套记忆：短期目标 episodic memory 与较长期 scene memory。

这说明声音不是“多一张图”而已。对导航任务，声源相对方向会随 agent 转身和移动而系统性变化；显式输入 self-motion 或让模型预测声源方位，通常比只依赖隐式端到端学习更有效。

### 3. 多声源、噪声和静音：训练时主动制造模态失效

真实声音不是持续、干净、唯一的。近两年的 benchmark 已把这些问题当作核心变量：

- BeDAViN 包含 2258 个音频样本、20 类目标声音和 4 类背景噪声，总时长约 10.8 小时；实验区分单声源、多声源和持续背景噪声。
- SAVN-CE 中目标开始发声的时间随机，发声时长服从随机分布，之后可能长期完全静音；还可加入与目标同期发声的 distractor。
- MAGNet 的消融结果表明，episodic memory 与 self-motion 两者都有贡献，组合效果最好。

因此，一个可信的 audio-RL 实验至少应包含：随机静音、音量扰动、左右声道扰动、无关声音、不同音效实例，以及 audio-only / visual-only / audio-visual 消融。否则模型可能只是把特定音效 ID 当作状态标签。

### 4. 难以模拟声音时：生成式音频扩展 simulator

MultiGen 解决的是另一个关键问题：许多物理模拟器能渲染 RGB 和动力学，却不能高效地产生可信声音。其流程是：

1. 在真实倒液体视频上微调视频到音频模型 MMAudio；
2. 用语义分割 mask 缩小真实视频与仿真 RGB 的 domain gap；
3. 为 RoboVerse 中运动规划生成的视觉/动作轨迹合成同步音频；
4. 用 RGB、proprioception、语言命令和生成音频训练 Diffusion Policy；
5. 在实机上由末端麦克风提供真实音频，零样本执行。

策略侧将音频降采样到 16 kHz，转换为 64-bin log-Mel spectrogram，再由 Audio Spectrogram Transformer 编码。结果显示音频对不透明容器收益更大，符合“视觉信息不足时声音更有价值”的预期。

严格来说这不是强化学习论文，但它提供了对 RL 同样适用的环境工程方法：当模拟器缺少某种 observation 时，可以离线生成与轨迹同步的模态，而不必先完成高保真声学物理模拟。

### 5. 声音先变成 belief，再输入 RL

双基地声呐 AUV 论文没有把高维声呐波形直接送给 PPO。声呐测量先由 tracker 估计目标运动状态与不确定性，policy 再依据 belief 决定下一航向。这是 POMDP 中很实用的分层设计：

```text
声学测量 -> 声源/目标估计器 -> belief state + uncertainty -> RL policy -> action
```

它牺牲了一部分端到端表达能力，换取更低样本复杂度、更强解释性和更稳定的仿真到现实迁移。如果声音只用于提示“门打开了”“拿到物品”“敌人受击”这类离散事件，LADX 也可先采用类似结构，把音频变成事件概率或 APU 状态，再给策略使用。

### 6. 语音与辅助目标：声音也可以承载任务语义

VLAS 表明，语音 observation 不只等于 ASR 文本。说话人特征等非文本信息也可决定定制化操作。它用三阶段训练把 speech encoder 接入 VLA，并在 CALVIN speech instructions 上评测。

DescRL 则从另一个方向提供辅助监督：让 RL policy 同时预测对过去或未来动作的自然语言描述。描述数据由预训练 VLM 蒸馏，避免人工标注。它在 semantic audio-visual navigation 中取得了最强结果，说明对共享 latent state 增加与规划相关的辅助任务，可以迫使策略保留更有用的多模态语义。

## 融合架构的可复用模式

### 早期融合

将像素、频谱和状态在较浅层拼接。实现最简单，但对不同尺度、不同噪声特性的模态不友好，且容易让高维视觉淹没音频。

### 独立编码后融合

```text
RGB -> visual encoder ----\
audio window -> audio encoder -> fusion -> recurrent/transformer memory -> actor/critic
pose/action -> MLP -------/
```

这是当前最稳妥的默认方案。音频编码器可从轻量 CNN 开始；视觉保持现有 CNN；在低维 embedding 层拼接或做 gated attention。

### 动态或门控融合

策略根据声音置信度、静音状态或视觉可见性改变各模态权重。虽然本调研核心正式论文更常采用显式描述器与 memory，但其效果实质上也是让音频在“可用时”更新 belief，在静音后依赖历史和 self-motion，而不是永久等权拼接。

### 辅助任务融合

除 action/value loss 外增加：

- 声音事件分类；
- 声源方向/距离预测；
- 下一段音频预测；
- 音视频对齐或对比学习；
- 动作描述或未来状态描述。

辅助任务既能改善 representation，也能提供诊断指标。若策略失败，可以区分是“没听懂”“定位错了”还是“听懂但控制错误”。

## 对 Zelda-LADX 环境的建议

### 当前可行性

当前 `ZeldaEnv` 的 observation 是 `(144, 160, 3)` 像素，`info` 中提供语义状态、事件和 reward terms；`PyBoyBackend.advance()` 每次逐帧调用 `pyboy.tick()`，尚未读取声音。

仓库固定依赖 PyBoy 2.7.0。PyBoy 官方 API 已提供 `pyboy.sound.ndarray`，返回最新 emulator frame 的立体声样本；文档强调需要在保存时 `.copy()`，因为底层 buffer 会复用。其采样率可在构造器设置，且应能被 60 整除。[PyBoy sound API](https://docs.pyboy.dk/api/sound.html)

这意味着无需修改 ROM 或外接录音设备，就能在 backend 层获取与每一帧严格对齐的 Game Boy APU 输出。

### 推荐的分阶段实验

#### Phase 0：验证音频数据链

- 在每次 `pyboy.tick()` 后复制 `pyboy.sound.ndarray`；
- 将一个 agent step 内 `frame_skip` 帧的 stereo buffers 按时间拼接；
- 固定 `sound_sample_rate`，记录窗口长度、dtype、峰值、静音比例；
- 用几段已知事件验证同步：挥剑、受伤、开宝箱、进入房间、音乐切换；
- reset/load-state 后确认音频缓冲区不会混入上一个 episode 的残留。

#### Phase 1：建立三组最小 baseline

1. `pixels-only`：现有 observation；
2. `audio-only`：log-Mel 或小型 STFT + 2D CNN；
3. `pixels+audio`：视觉和音频独立编码，在低维层拼接，再送入 recurrent policy。

音频窗口建议先与 agent step 一一对应，不做过长上下文。当前 `frame_skip=4` 时，一个 step 约为 1/15 秒；可以叠加最近 4-8 个 step，使音频上下文约 0.27-0.53 秒。窗口过短难以区分旋律/音效，过长则增加动作-声音 credit assignment 延迟。

#### Phase 2：加入事件化音频辅助任务

LADX 的音频是确定性的 APU 合成信号，适合增加轻量辅助头：

- 预测当前是否出现攻击、受击、菜单、宝箱、门、拾取等音效；
- 预测音乐/音效状态是否变化；
- 预测下一步音频 embedding；
- 用现有 `events` 作为训练标签，但不要把这些 privileged labels 放进评测时 observation。

这能检验音频 encoder 是否捕获了任务相关信号，而不是只学习背景音乐 ID。

#### Phase 3：做缺失模态和泛化实验

- 随机静音连续若干 step；
- 音量缩放、轻微时间偏移、左右声道交换；
- 叠加与任务无关的其他房间音效；
- 在训练未见的地图、音乐和 save state 上评测；
- 比较是否加入 LSTM/GRU memory；
- 分别报告成功率、样本效率、回报、episode 长度和音频辅助任务准确率。

### 推荐 observation API

不要直接把音频塞进现有像素 `Box`。更自然的 Gymnasium 接口是 `spaces.Dict`：

```python
spaces.Dict({
    "pixels": spaces.Box(0, 255, shape=(144, 160, 3), dtype=np.uint8),
    "audio": spaces.Box(-1.0, 1.0, shape=(num_samples, 2), dtype=np.float32),
})
```

也可以让基础环境继续保持 pixel-only，并提供 `AudioObservation` wrapper，以避免破坏现有训练脚本和测试。wrapper 可支持 `waveform`、`log_mel` 和 `event_embedding` 三种模式。

### 建议首先回答的研究问题

1. 在画面基本可见的 LADX 中，音频是否提高样本效率，还是只在部分可观测/遮挡/低帧率条件下有用？
2. 声音收益来自瞬时音效，还是来自背景音乐对区域/状态的提示？
3. 原始频谱与结构化 APU/音频事件特征，哪一种更容易泛化到未见状态？
4. 音频在 reward 稀疏时能否作为 auxiliary prediction signal 改善 representation，而不进入 policy observation？
5. 当画面降采样、frame stack 减少或部分遮挡时，声音的边际价值如何变化？

第五个问题尤其重要。若完整像素已经包含几乎全部 Markov 信息，音频的增益可能很小；人为构造合理的视觉不完整条件，才能检验声音作为互补感官而不是冗余标签的价值。

## 实验设计中的常见陷阱

- **时间不同步**：音效往往由前一帧动作触发。应明确 observation 是动作前还是动作后窗口，并在 replay 中保存时间戳/帧号。
- **窗口长度不固定**：模拟器每帧音频样本数可能轻微波动，应先聚合再 pad/crop，或在 encoder 前重采样。
- **音频泄漏任务 ID**：背景音乐可能直接标识房间或关卡。需要音乐替换、静音和未见音乐测试。
- **只报告最终回报**：至少提供 audio-only、visual-only、fusion 消融和噪声/静音鲁棒性。
- **把 privileged state 当 observation**：可以用 RAM events 标注辅助任务或 reward，但评测时必须说明策略是否能看到这些标签。
- **并行环境开销**：音频模拟和 STFT 会降低环境吞吐；应分别测 emulator FPS、feature extraction 时间和 learner 时间。
- **保存状态后的音频历史不一致**：普通 emulator save state 未必包含外部堆叠窗口，wrapper 自己的音频历史必须在 reset/load 时清空或恢复。

## 进一步参考：边界案例

[AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video](https://openaccess.thecvf.com/content/CVPR2026/html/Kulkarni_AVATAR_Reinforcement_Learning_to_See_Hear_and_Reason_Over_Video_CVPR_2026_paper.html)（CVPR 2026）使用 off-policy replay 和 Temporal Advantage Shaping，以 RL 提升长视频音视频推理。它在 MMVU、OmniBench、Video-Holmes 等 benchmark 上有效，但没有可交互环境、控制动作或环境回报，因此不能证明声音改善 embodied control。可借鉴的是：多模态长时序任务的 credit assignment 不应对所有 reasoning step 等权。

若未来将范围扩展到 2024 年以前，SoundSpaces、SAVi、CAVEN、Sound Adversarial Audio-Visual Navigation、Noisy Agents 等是理解本领域演化所需的前置工作；本报告没有把它们计入“2025 年至今”的核心结果。

## 最终判断

近期证据支持以下判断：

- 声音在**目标不可见、视觉歧义、持续过程估计、部分可观测或语音交互**任务中价值最大；
- 对控制策略而言，**时间记忆和 self-motion 建模**通常与音频编码器同样重要；
- **显式声学辅助目标**能改善训练稳定性、可解释性和故障诊断；
- 模拟器没有声音时，**生成式音频**已成为可行的数据构造手段；
- 并非所有“audio + RL”论文都在做声音 observation 控制，必须区分 embodied policy、离线策略学习和多模态推理 RL。

对 LADX，最有研究价值且风险最低的路线是：先完成 PyBoy 同步立体声采集与 Dict/wrapper observation，建立轻量 log-Mel CNN baseline，再加入事件预测辅助头和静音/音乐泄漏消融。若这一步能证明音频在低视觉信息或长时序任务上提高样本效率，再考虑更复杂的 cross-attention、预训练音频 encoder 或世界模型。

