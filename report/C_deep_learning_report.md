# C同学任务：PyTorch深度学习建模与结果可视化

## 数据与防泄露处理

本部分基于A同学提交的 `AmesHousing_Cleaned.csv`，共 `2930` 条样本。目标变量使用清洗阶段生成的 `LogSalePrice`；建模输入中同时删除 `SalePrice` 和 `LogSalePrice`，避免把目标值或其等价变换作为特征造成数据泄露。

数据划分采用固定随机种子 `42`，并按目标变量分位数分箱后进行训练集、验证集、测试集划分，使各集合的房价分布尽量一致。数值特征使用 `StandardScaler` 标准化，分类特征使用 `OneHotEncoder(handle_unknown="ignore")` 编码。

## MLP模型设计

使用PyTorch搭建多层感知机回归网络，结构由全连接层、BatchNorm、ReLU和Dropout组成。训练时对目标变量做标准化，预测后还原到对数房价尺度计算指标。

本次尝试了 `3` 组超参数，最优模型为 `MLP_medium`：

- 隐藏层：`(256, 128, 64)`
- Dropout：`0.15`
- 学习率：`0.0008`
- 权重衰减：`1e-05`
- Batch size：`64`
- Early stopping patience：`35`

## 测试集结果

| 模型 | RMSE(log) | R2(log) | RMSE(原始价格) | R2(原始价格) |
|---|---:|---:|---:|---:|
| MLP_medium | 0.113032 | 0.914603 | 21741.51 | 0.913334 |

全部MLP调参结果已保存到 `output/deep_model/mlp_tuning_results.csv`，与B同学传统模型的对比结果保存到 `output/deep_model/model_comparison.csv`。

## 图表清单

- `output/deep_model/01_training_curve.png`：训练损失和验证集RMSE曲线
- `output/deep_model/02_prediction_vs_true.png`：预测值与真实值散点图
- `output/deep_model/03_residual_plot.png`：残差图
- `output/deep_model/04_model_comparison.png`：传统模型与MLP性能对比柱状图

## 报告整合说明

B分支的 `model_results.csv` 被用于生成性能对比图。需要注意：A的清洗数据同时包含 `SalePrice` 和 `LogSalePrice`，传统模型脚本若只删除 `SalePrice`，可能把 `LogSalePrice` 当作输入特征，造成结果异常偏高。最终报告建议说明C部分已显式删除两个目标相关列，以保证深度学习评估不发生目标泄露。
