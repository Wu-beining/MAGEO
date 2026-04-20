<p align="center">
  <h1 align="center">MAGEO: Memory-Augmented Multi-Agent Generative Engine Optimization</h1>
</p>

<p align="center">
  <em>基于记忆增强与多智能体协作的生成式引擎优化框架 — ACL 2026</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ACL-2026-blue" alt="ACL 2026"/>
  <img src="https://img.shields.io/badge/Python-3.12%2B-green" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/LiteLLM-Support-orange" alt="LiteLLM"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

---

## 📌 概述

**MAGEO** 是一种面向生成式搜索引擎（Generative Engines, GE）的自动化内容优化框架。当前仓库已按论文的方法部分收束为**Preference / Planner / Editor / Evaluator** 四智能体闭环，并使用论文中的 **DSV-CF** 指标族作为默认优化目标。

### ✨ 核心特点

- 🤖 **四智能体协作流**：偏好建模器（Preference）、规划器（Planner）、编辑器（Editor）与评估器（Evaluator）协同优化内容。
- ⚖️ **DSV-CF 双轴评估体系**：采用论文方法部分定义的 8 个核心指标：`WLV`, `DPA`, `CP`, `SI`, `AA`, `FA`, `KC`, `AD`。
- 🧠 **双层记忆增强机制（Hierarchical Memory）**：
  - *步骤级记忆 (Step-level)*：存储单次迭代的局部有效策略。
  - *创作者级记忆 (Creator-level)*：跨实例沉淀全系统全局视角的强效编辑范式。
- 🚦 **基于 Fidelity Gate 的早停与选择机制**：在保证 `AA/FA` 不失真的前提下，按 DSV-CF 目标选择候选并在分数平台期早停。

### 🔎 论文对齐说明

- 仓库当前以 `pipeline/geo_optimizer.py` 为论文主链路实现，闭环包含 `Preference -> Planner -> Editor -> Evaluator -> Selector -> Hierarchical Memory`。
- 我们补充整理了一份更直接的“论文模块 vs 代码入口”说明，见 [docs/paper_alignment.md](docs/paper_alignment.md)。
- `tool/web_search.py` 依赖可选的 `zai-sdk` 与 `WEB_SEARCH_API_KEY`；即使未安装该依赖，核心模块现在也能正常导入，不会再阻塞整个项目。

---

## 🏗️ 方法框架

### 范式转移

<p align="center">
  <img src="background.png" alt="SEO to GEO paradigm shift" width="100%"/>
</p>
<p align="center"><b>图 1.</b> SEO到GEO：从面向排名的目标转向基于综合的影响，突出了四个基本挑战：不透明的呈现、未定义的指标、不清晰的优化路径和模糊的偏好。</p>

### MAGEO整体流程

<p align="center">
  <img src="model.png" alt="MAGEO optimization workflow" width="100%"/>
</p>
<p align="center"><b>图 2.</b> MAGEO 多智能体协同优化整体框架。涵盖从偏好分析、智能规划、多样性编辑到 DSV-CF 多维评估及融合闭环。</p>

---

## 📁 项目结构

```
MAGEO/
├── 📄 README.md                    # 本文档
├── 📄 MAGEO_开发文档.md             # 论文对齐后的核心组件说明
├── 📄 requirements.txt             # 依赖环境配置
│
├── 📂 agent/                       # 🤖 智能体核心模块
│   ├── 📄 preference_agent.py      # 引擎偏好画像构建智能体
│   ├── 📄 planner_agent.py         # 战略规划智能体
│   ├── 📄 editor_agent.py          # 候选版本编辑智能体
│   └── 📄 evaluation_agent.py      # DSV-CF 评估与 fidelity gate
│
├── 📂 evaluation/                  # 📏 评估与指标体系 (DSV-CF)
│   ├── 📄 metrics.py               # DSV-CF 指标与目标函数
│   ├── 📄 candidate_selector.py    # 最优候选抉择与早停判断
│   └── 📄 simulated_evaluator.py   # RAG 本地沙盒仿真生成引擎
│
├── 📂 memory/                      # 🧠 双层知识存储库
│   ├── 📄 memory_bank.py           # Step-level / Creator-level 记忆引擎
│   └── 📄 schema.py                # 序列化规范定义
│
├── 📂 pipeline/                    # 🚀 主控循环与链路串联
│   ├── 📄 geo_optimizer.py         # MAGEO 主控回路
│   └── 📄 interactive_optimize.py  # 终端交互测试主程序
│
├── 📂 prompt/                      # 📝 Role-playing System Prompts
├── 📂 config/                      # ⚙️ 环境及大模型定义
└── 📂 scripts/                     # 🛠️ 自动化测试集批处理
```

---

## 🔧 环境安装

### 前置要求

- Python 3.12+
- `uv` 包管理器 (推荐使用)

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/Wu-Beining/MAGEO.git
cd MAGEO

# 2. 创建并激活虚拟环境 (推荐使用 uv 加速)
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安装依赖库
uv pip install -r requirements.txt

# 4. 配置 API Key (重要)
# 项目目录下提供 `.example_env`，请复制为 `.env` 并填入：
cp .example_env .env
# OPENAI_API_KEY="sk-..."
# WEB_SEARCH_API_KEY="..." 
```

### 关键架构依赖

| 库 | 版本 | 用途 |
|---|------|------|
| LiteLLM | 1.80.11 | 大语言模型统一代理（支持定制 OpenAI/Claude/Zhipu 等请求）|
| Pydantic | 2.12.5 | 智能体交互架构的强类型数据验证 |
| aiohttp | 3.13.3 | 底层异步高并发引擎支持 |
| AnyIO / asyncio | 4.12.0 | 智能体并行调度与子线程安全管理 |

---

## 🚀 快速开始

### 交互式流端到端优化体验

只需提供一条需要改善排名的 Query，框架即可自动完成搜索、RAG对齐仿真、局部规划与迭代升级：

```bash
python -m pipeline.interactive_optimize --query "Best AI coding agents" --auto
```

<details>
<summary>📋 交互式入口可选参数</summary>

| 参数 | 选项 | 说明 |
|------|--------|------|
| `--query` | `-q` | 指定优化目标的搜索引擎查询意图 |
| `--auto` | `-a` | 自动化模式（搜索完后默认采用 Top-1 结果作为改造目标）|
| `--yes` | `-y` | 自动化跳过确认，一键式持续执行直到早停 |

</details>

### 集群/自动化指标批处理

对企业内部网络或批量内容分析实验进行 Benchmark 测试：

```bash
python scripts/batch_optimize_v2.py --json path/to/test_queries.json
```
该脚本需要一个查询列表 JSON，例如：
```json
[
  {
    "index": 1,
    "query": "Best AI coding agents",
    "is_optimized": false,
    "log": ""
  }
]
```
运行结果会写入 `log/optimization_web/`。如果需要逐条串行执行，可使用 `python scripts/batch_optimize_sequential.py --json path/to/test_queries.json`。

---

## 🧩 智能体与核心配置

本项目采用与论文方法部分一致的多智能体优化策略：

```
┌─────────────────  Planning Level  ───────────────┐
│ Preference 归纳引擎偏好，Planner 生成修订计划       │
└──────────────────────────────────────────────────┘
                             ↓
┌─────────────────  Execution Level  ──────────────┐
│ Editor 依据方案生成 $K$ 个多样化增强分支          │
└──────────────────────────────────────────────────┘
                             ↓
┌─────────────────  Evaluation Level ──────────────┐
│ 1) QA 代理在固定 retrieval list 下生成响应        │
│ 2) Evaluation 依照 DSV-CF (8指标) 进行评分        │
│ 3) Fidelity Gate: AA 与 FA 不得失真               │
└──────────────────────────────────────────────────┘
```

**评估指标核心约束 (DSV-CF)**：
- **SSV 表面可见性**：`WLV`, `DPA`, `CP`, `SI`
- **ISI 内在影响力**：`AA`, `FA`, `KC`, `AD`
- **安全门**：`AA` 与 `FA` 不得退化，且 `FA` 需高于 fidelity gate 阈值

---

## 📊 实验结果回顾

### 性能跃迁与安全屏障 (MSME-GEO-Bench)

相比于单纯激进的词频强加式 SEO，MAGEO 通过**多角色动态制衡机制（评价与融合）** 达成了跨平台的适应性平衡，特别体现在：
- **表面可见性 (WLV / DPA)** 与 **回答主导权 (AD)** 指标相较传统启发式 GEO 有显著提升。
- **事实幻觉风险 (1-FA)** 抑制在趋近极微幅度，完全超越单大模型无经验改写的安全基线。

### 数据集概览

<p align="center">
  <img src="data.png" alt="Dataset overview" width="100%"/>
</p>

### 实用 Pareto Frontier

<p align="center">
  <img src="mageo_pareto_frontier_comic.png" alt="Practical Pareto frontier" width="70%"/>
</p>

---

## 📄 License

This project is licensed under the MIT License.
