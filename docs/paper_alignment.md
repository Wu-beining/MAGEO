# MAGEO 论文对齐说明

本说明只保留与论文方法部分和评价指标直接对应的实现，不再混入早期 demo 代码。

## 1. 论文方法对应的最小实现

- Twin-Branch Evaluation Protocol
  入口在 [evaluation/simulated_evaluator.py](../evaluation/simulated_evaluator.py)。
  在固定 retrieval list 的前提下，仅替换目标文档内容，重新生成响应并评估改动效果。
- Preference Agent
  入口在 [agent/preference_agent.py](../agent/preference_agent.py)。
  负责把原始 `engine_rules` 规范化为可复用的 Preference Profile。
- Planner Agent
  入口在 [agent/planner_agent.py](../agent/planner_agent.py)。
  输入 query、当前文档、Preference Profile 和 memory examples，输出 `plan_steps`。
- Editor Agent
  入口在 [agent/editor_agent.py](../agent/editor_agent.py)。
  根据 `plan_steps` 生成多个候选版本。
- Evaluator Agent
  入口在 [agent/evaluation_agent.py](../agent/evaluation_agent.py)。
  基于 DSV-CF 预测候选质量，并执行 fidelity gate 所需的关键分数判断。
- Dual-Layer Memory
  入口在 [memory/memory_bank.py](../memory/memory_bank.py) 和 [memory/schema.py](../memory/schema.py)。
  Step-level memory 记录单轮有效编辑轨迹；
  Creator-level memory 记录跨实例可复用模式。
- Optimization Loop
  入口在 [pipeline/geo_optimizer.py](../pipeline/geo_optimizer.py)。
  按照 `Preference -> Planner -> Editor -> Evaluator -> Select -> Memory` 闭环执行。

## 2. 论文指标对应的当前实现

论文方法部分中，DSV-CF 采用两条主轴共 8 个核心指标：

- SSV: `WLV`, `DPA`, `CP`, `SI`
- ISI: `AA`, `FA`, `KC`, `AD`

当前仓库中的实现位置：

- 指标定义与 DSV-CF 公式：
  [evaluation/metrics.py](../evaluation/metrics.py)
- 候选筛选与目标函数排序：
  [evaluation/candidate_selector.py](../evaluation/candidate_selector.py)

代码中的目标函数与论文一致：

`S_DSV-CF = λ * SSV + (1 - λ) * ISI - γ * (10 - AA)`

默认使用：

- `lambda = 0.5`
- `gamma = 0.5`

## 3. 这次清理掉的旧实现

以下内容不属于论文主方法，已经从主仓库实现中剥离：

- `FusionAgent`
- `ReactAgent`
- `ToolAgent`
- `SummaryMemory`
- 通用 tool 注册器与 save_json 小工具
- DAG 风格的 `pipeline/base.py`
- 依赖在线模型的旧 demo 测试脚本

这些删除是为了避免“论文主方法”和“历史实验脚手架”混在一起。
