# MAGEO 开发文档

这份文档只描述当前与论文方法部分和评价指标保持一致的实现。

## 1. 方法主链路

MAGEO 当前实现对应论文中的以下闭环：

1. Preference Agent 规范化 `engine_rules`
2. Twin-Branch 条件下对当前文档建立基线
3. 从 dual-layer memory 检索可复用经验
4. Planner 生成高层修订计划
5. Editor 生成多个候选版本
6. Evaluator 按 DSV-CF 预测质量，并执行 fidelity gate 所需判断
7. 选择 DSV-CF 增益最大的安全候选
8. 回写 Step-level / Creator-level memory
9. 在 DSV-CF plateau 时早停

核心入口文件：

- [pipeline/geo_optimizer.py](pipeline/geo_optimizer.py)

## 2. 四类核心 Agent

- Preference Agent
  [agent/preference_agent.py](agent/preference_agent.py)
- Planner Agent
  [agent/planner_agent.py](agent/planner_agent.py)
- Editor Agent
  [agent/editor_agent.py](agent/editor_agent.py)
- Evaluator Agent
  [agent/evaluation_agent.py](agent/evaluation_agent.py)

## 3. 指标体系

当前实现不再使用旧版 13 维评分，而是直接收束到论文方法部分的 DSV-CF：

- SSV
  `WLV`, `DPA`, `CP`, `SI`
- ISI
  `AA`, `FA`, `KC`, `AD`

公式实现位于：

- [evaluation/metrics.py](evaluation/metrics.py)

选择逻辑位于：

- [evaluation/candidate_selector.py](evaluation/candidate_selector.py)

## 4. 已移除内容

为避免偏离论文主方法，当前仓库已移除：

- FusionAgent
- ReactAgent / ToolAgent
- SummaryMemory
- 通用 tool 注册器
- 与主方法无关的 DAG pipeline 基类
- 旧版在线 demo 测试
