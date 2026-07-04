# C同学任务说明

本目录用于放置深度学习建模和报告整合材料。

## 已实现内容

- 使用 `AmesHousing_Cleaned.csv` 作为清洗后数据源。
- 使用 `LogSalePrice` 作为回归目标，并从输入特征中删除 `SalePrice` 与 `LogSalePrice`，避免目标泄露。
- 使用 PyTorch 搭建 MLP 回归网络，并通过验证集比较多组超参数。
- 输出测试集 RMSE、R2、训练曲线、预测-真实散点图、残差图和模型性能对比柱状图。
- 将 C 部分可直接写入项目报告的内容保存到 `report/C_deep_learning_report.md`。

## 运行方式

```bash
conda activate Ames-Housing
python deep_model.py
```

或者直接运行 `Deep_Model.ipynb`。
