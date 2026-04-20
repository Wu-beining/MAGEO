# MAGEO 测试与验证说明

当前仓库只保留与论文方法部分一致的最小验证路径。

## 1. 核心验证目标

- 验证 Twin-Branch 设定下，目标文档替换是否能改变生成回答中的暴露度与影响力。
- 验证候选选择是否按论文中的 DSV-CF 目标进行，而不是旧版的维度计数逻辑。
- 验证 memory 是否记录真实有效的局部编辑轨迹和跨实例模式。

## 2. 当前保留的自动化测试

- [test/test_candidate_selector_unit.py](../test/test_candidate_selector_unit.py)
- [test/test_config_base_unit.py](../test/test_config_base_unit.py)
- [test/test_memory_bank.py](../test/test_memory_bank.py)

## 3. 当前论文对齐的指标集

- SSV
  `WLV`, `DPA`, `CP`, `SI`
- ISI
  `AA`, `FA`, `KC`, `AD`
- Overall
  `DSV-CF`

## 4. 推荐验证命令

```bash
pytest -q test\test_candidate_selector_unit.py test\test_config_base_unit.py test\test_memory_bank.py
```
