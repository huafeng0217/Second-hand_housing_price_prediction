import joblib
import pandas as pd
import numpy as np

# 加载训练好的Lasso模型
model = joblib.load("lasso_model.pkl")
df = pd.read_csv("AmesHousing_Cleaned.csv")
X = df.drop("SalePrice",axis=1)

# 任选一行房源做预测
sample = X.iloc[0:1,:]
log_pred = model.predict(sample)[0]
real_price = np.exp(log_pred)
print(f"对数预测房价：{log_pred:.4f}")
print(f"还原原始成交房价：{real_price:.2f}")