# 自然语言处理大作业 - 反诈骗检测对抗实验

本项目旨在研究针对大模型反诈骗检测系统的对抗攻击策略。通过对诈骗对话进行不同层级（词汇级、语义级）的改写，评估检测模型的鲁棒性。

## 项目结构

- **final.py**: 核心代码。包含：
  - `detect_fraud`: 基于 DeepSeek 的诈骗检测器。
  - `rewrite_lexical`: 策略 A - 同义词替换攻击。
  - `rewrite_semantic`: 策略 B - 语义伪装/风格迁移攻击。
  - 并发处理逻辑，用于批量跑实验。
- **dataset.csv**: 实验数据集，包含对话内容及诈骗标签。
- **ablation_experiment_results_parallel.csv**: 生成的实验结果文件，包含攻击前后的文本及检测结果。

## 环境依赖

需要安装以下 Python 库：

```bash
pip install pandas requests tqdm
```

此外，代码依赖于本地部署的 DeepSeek 模型 API（兼容 OpenAI 格式）：
- 模型：`DeepSeek-R1-Distill-Qwen-7B`
- 默认地址：`http://127.0.0.1:8000/v1/chat/completions`
- 如果你的 API 地址不同，请修改 `final.py` 中的 `INSTANCE_IP` 或 `API_URL`。

## 运行说明

1. 确保本地大模型服务已启动。
2. 确保 `dataset.csv` 与脚本在同一目录下。
3. 运行脚本：

   ```bash
   python final.py
   ```

4. 运行结束后，结果将保存至 `ablation_experiment_results_parallel.csv`。

## 实验逻辑

对于数据集中的每一条记录：
1. **基准测试**：使用检测器判断原始文本是否为诈骗。
2. **对抗攻击**（仅针对诈骗样本）：
   - **策略 A (Lexical)**：保持句式，仅替换敏感词（如“转账”->“办理”）。
   - **策略 B (Semantic)**：重写句式，模拟专业客服语气，进行深度伪装。
3. **攻击评估**：再次检测改写后的文本，判断攻击是否成功（即原本被识别为诈骗，改写后被识别为正常）。
