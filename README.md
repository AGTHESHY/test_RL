# 强化学习对齐 (RLHF / DPO / PPO / GRPO) 学习项目

这是 [`test-lora`](../test-lora) 的**续篇**。上一个仓库我们把 **SFT(监督微调)** 用 LoRA / QLoRA 跑通了,
并在最后一章 `04_衍生_从SFT走向强化学习` 埋下了伏笔:SFT 只会「模仿标准答案」,学不到「答案 A 比答案 B 好」这种**偏好**。

本仓库就来动手实现**用强化学习对齐大模型**,循序渐进地走完主流三条路线:

- **奖励模型 (Reward Model)**:用成对偏好数据训练一个「打分器」。
- **DPO(Direct Preference Optimization)**:跳过显式奖励模型,直接从偏好对里学(**推荐入门**,稳、省资源)。
- **PPO(Proximal Policy Optimization)**:经典 RLHF,走一遍「采样 → 打分 → 带 KL 惩罚地更新策略」的闭环。
- **GRPO(Group Relative Policy Optimization)**:面向可验证任务(如数学题)的新方法,去掉价值网络、用组内相对优势。

> **承上启下:** `test-lora` 训练出的 SFT 模型,正是本仓库里强化学习的**起点(策略模型的初始化)**。
> 本仓库可独立运行(直接用 Qwen2.5 官方 Instruct 底座起步);若你已完成 `test-lora`,也可把策略模型指向那边合并好的模型。

## 运行环境

本项目在以下环境开发与验证(与 `test-lora` 完全一致):

- Windows 10/11
- NVIDIA RTX 5080,16GB 显存(**Blackwell 架构 / sm_120**)
- Conda(miniconda)

> **重要:Blackwell (RTX 50 系) 用户必看**
>
> RTX 50 系是 Blackwell 架构(计算能力 sm_120)。普通稳定版 PyTorch 常常没有编译对应 kernel,运行时会报
> `CUDA error: no kernel image is available for execution on the device`。
> 因此**必须安装 CUDA 12.8 (cu128) 的 PyTorch(≥ 2.7)**,并使用较新的 `bitsandbytes`(≥ 0.45)。
> 具体安装命令见 `notebooks/00_环境准备与GPU检测.ipynb`。

## 环境安装

遵循本机的 conda 约定,新建独立环境 `test-rl`,不污染系统 Python。

```bash
# 1) 创建并激活环境
conda env create -f environment.yml
conda activate test-rl

# 2) 安装 Blackwell 可用的 PyTorch(cu128)——注意这一步不能用普通 pip install torch
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# 3) 安装其余依赖
pip install -r requirements.txt
```

安装完成后,先运行 `notebooks/00_环境准备与GPU检测.ipynb` 验证 GPU、4-bit 与 trl 训练器是否可用。

## 阅读顺序

| 顺序 | Notebook | 内容 | 演示模型 |
| --- | --- | --- | --- |
| 0 | `00_环境准备与GPU检测.ipynb` | 环境安装、GPU/4-bit 检测、trl 版本自检 | - |
| 1 | `01_强化学习对齐总览与RLHF全景.ipynb` | 策略/奖励/参考模型、KL、策略梯度直觉、三条路线对比 | 仅概念 |
| 2 | `02_奖励模型RewardModel原理与实战.ipynb` | Bradley-Terry、`RewardTrainer` + LoRA 训练打分模型 | Qwen2.5-0.5B-Instruct |
| 3 | `03_DPO原理与实战.ipynb` | DPO 损失 / beta,`DPOTrainer` + QLoRA 偏好对齐 | Qwen2.5-1.5B-Instruct |
| 4 | `04_PPO原理与实战.ipynb` | 4 模型闭环、`PPOTrainer`、reward/KL 曲线 | Qwen2.5-0.5B-Instruct |
| 5 | `05_GRPO原理与实战.ipynb` | 组内相对优势、`GRPOTrainer` + 自定义奖励函数 | Qwen2.5-0.5B-Instruct |
| 6 | `06_对齐前后对比与评估.ipynb` | SFT/DPO 生成对比、奖励均分、KL、win-rate、选型总结 | 承接 02/03 |

## 三条路线速览

| 路线 | 要不要单独的奖励模型 | 要不要在线采样 | 维护几个模型 | 稳定性/资源 | 适合场景 |
| --- | --- | --- | --- | --- | --- |
| RM + PPO | 要 | 要 | 4(策略/参考/奖励/价值) | 复杂、吃显存 | 经典 RLHF、上限高 |
| DPO | 不要 | 不要 | 2(策略/参考) | 稳、省资源 | 有成对偏好数据、入门首选 |
| GRPO | 用奖励函数替代 | 要 | 2~3(策略/参考[/奖励]) | 中等 | 有可验证奖励(数学/代码/格式) |

## 显存与规模(16GB 参考)

| 章节 | 方法 | 底座 | 加载精度 | 大致显存 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 02 | Reward Model + LoRA | 0.5B | bf16 | ~3-6GB | 序列分类头 |
| 03 | DPO + QLoRA | 1.5B | 4-bit NF4 | ~8-12GB | 一次要过 chosen+rejected |
| 04 | PPO | 0.5B | bf16 | ~8-14GB | 同时有策略/参考/奖励/价值 |
| 05 | GRPO | 0.5B | bf16 | ~8-14GB | 每个 prompt 采样一组 |

> 所有 notebook 都刻意把训练步数、batch、序列长度设得很小,保证**单张 16GB 卡能真跑通**;
> 注释里说明了如何加大以获得更明显的效果。

## 数据来源(HuggingFace 优先 + 本地兜底)

各 notebook **优先从 HuggingFace 下载真实、优质的数据集**(比仓库里手写的少量样例效果好得多);
如果下载不便(断网 / 无代理),代码会**自动回退**到 `data/` 下的本地 jsonl,保证离线也能跑通。

| 章节 | 首选 HuggingFace 数据集 | 用途 | 本地兜底 |
| --- | --- | --- | --- |
| 02 / 03 | [`llamafactory/DPO-En-Zh-20k`](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k)(`zh` 子集) | 中文成对偏好(RM / DPO) | `data/preference.jsonl` |
| 04 | [`llamafactory/alpaca_gpt4_zh`](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh) | PPO 采样用的中文指令 prompt | `data/prompts.jsonl`(chat) |
| 05 | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k)(`main`) | GRPO 可验证的数学题(带标准答案) | `data/prompts.jsonl`(math) |

> 每个数据 cell 顶部都有 `N_SAMPLES` / `N_PROMPTS` 之类的开关,演示只取一小部分;想要好效果就调大它。
> 下载慢可在 cell 里取消注释 `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"` 走国内镜像。

## 目录结构

```
test_RL/
├── README.md
├── environment.yml
├── requirements.txt
├── data/                       # 仅作离线兜底;默认优先从 HuggingFace 下载(见上表)
│   ├── preference.jsonl        # 成对偏好数据:prompt + chosen + rejected(RM / DPO 共用)
│   └── prompts.jsonl           # 采样用纯 prompt(PPO / GRPO;GRPO 条目含可验证答案)
└── notebooks/
    ├── 00_环境准备与GPU检测.ipynb
    ├── 01_强化学习对齐总览与RLHF全景.ipynb
    ├── 02_奖励模型RewardModel原理与实战.ipynb
    ├── 03_DPO原理与实战.ipynb
    ├── 04_PPO原理与实战.ipynb
    ├── 05_GRPO原理与实战.ipynb
    └── 06_对齐前后对比与评估.ipynb
```

## 说明

- 每个 notebook 顶部都注明前置条件与本机(16GB / Blackwell)的注意事项。
- 训练产物默认写到 `outputs/`(已在运行时创建),多个 notebook 之间会互相引用(如 04 用 02 的奖励模型)。
- 延续 `test-lora` 的习惯:**中文讲解 + Blackwell 环境 + 尽量在单张 16GB 卡上跑通**。
