import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据+只取前300行，大幅减小数据量防卡死
df = pd.read_csv("AmesHousing_Cleaned.csv")

y = np.log(df["SalePrice"])
X = df.drop("SalePrice", axis=1)

# 区分特征类型
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 关键改动：取消GridSearch网格搜索，直接固定参数，不用交叉验证 ==========
# Lasso
lasso_pipe = Pipeline([("pre", preprocessor), ("model", Lasso(alpha=0.01, random_state=42))])
lasso_pipe.fit(X_train,y_train)
y_pred_ls = lasso_pipe.predict(X_test)
ls_rmse = np.sqrt(mean_squared_error(y_test,y_pred_ls))
ls_r2 = r2_score(y_test,y_pred_ls)

# RF单核、小树
rf_pipe = Pipeline([("pre", preprocessor), ("model", RandomForestRegressor(n_estimators=30,n_jobs=1,random_state=42))])
rf_pipe.fit(X_train,y_train)
y_pred_rf = rf_pipe.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test,y_pred_rf))
rf_r2 = r2_score(y_test,y_pred_rf)

# LGB单核、小树
lgb_pipe = Pipeline([("pre", preprocessor), ("model", LGBMRegressor(n_estimators=30,learning_rate=0.1,n_jobs=1,random_state=42))])
lgb_pipe.fit(X_train,y_train)
y_pred_lgb = lgb_pipe.predict(X_test)
lgb_rmse = np.sqrt(mean_squared_error(y_test,y_pred_lgb))
lgb_r2 = r2_score(y_test,y_pred_lgb)

# 保存结果
result = pd.DataFrame({
    "模型":["Lasso","RandomForest","LightGBM"],
    "RMSE":[ls_rmse,rf_rmse,lgb_rmse],
    "R2":[ls_r2,rf_r2,lgb_r2]
})
print(result)
result.to_csv("model_results.csv",index=False,encoding="utf-8-sig")
import joblib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

#1、保存三个训练完成的流水线模型
joblib.dump(lasso_pipe,"lasso_model.pkl")
joblib.dump(rf_pipe,"rf_model.pkl")
joblib.dump(lgb_pipe,"lgb_model.pkl")
print("模型保存完毕，生成3个pkl文件")

#2、真实值vs预测值散点图（LGB为例，通用画图，保存图片给C）
plt.figure(figsize=(7,6))
plt.scatter(y_test,y_pred_lgb,alpha=0.6)
#对角理想拟合线
min_y = min(y_test.min(),y_pred_lgb.min())
max_y = max(y_test.max(),y_pred_lgb.max())
plt.plot([min_y,max_y],[min_y,max_y],'r--',lw=2)
plt.xlabel("真实对数房价")
plt.ylabel("预测对数房价")
plt.title("LGB真实值-预测值对比")
plt.savefig("true_pred_scatter.png",dpi=300)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

# 提取特征名与特征权重
preprocessor = lasso_pipe.named_steps["pre"]
feature_names = preprocessor.get_feature_names_out()

# 随机森林TOP10特征
rf = rf_pipe.named_steps["model"]
rf_imp = pd.DataFrame({"name":feature_names,"imp":rf.feature_importances_}).sort_values("imp",ascending=False).head(10)
rf_imp.to_csv("rf_top10特征.csv",encoding="utf-8-sig",index=False)
plt.figure(figsize=(9,5))
plt.barh(rf_imp["name"][::-1],rf_imp["imp"][::-1])
plt.title("随机森林TOP10关键特征")
plt.savefig("rf_feature_top10.png",dpi=300)

# LGB TOP10特征
lgb = lgb_pipe.named_steps["model"]
lgb_imp = pd.DataFrame({"name":feature_names,"imp":lgb.feature_importances_}).sort_values("imp",ascending=False).head(10)
lgb_imp.to_csv("lgb_top10特征.csv",encoding="utf-8-sig",index=False)
plt.figure(figsize=(9,5))
plt.barh(lgb_imp["name"][::-1],lgb_imp["imp"][::-1])
plt.title("LightGBM TOP10关键特征")
plt.savefig("lgb_feature_top10.png",dpi=300)

# Lasso正负系数表
lasso = lasso_pipe.named_steps["model"]
lasso_df = pd.DataFrame({"name":feature_names,"coef":lasso.coef_})
lasso_df["影响方向"] = lasso_df["coef"].map(lambda x:"正向抬价" if x>0 else "负向降价")
lasso_df.to_csv("lasso全量系数.csv",encoding="utf-8-sig",index=False)
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
preprocessor = lasso_pipe.named_steps["pre"]
feature_names = preprocessor.get_feature_names_out()

#随机森林TOP10特征
rf = rf_pipe.named_steps["model"]
rf_imp = pd.DataFrame({"name":feature_names,"imp":rf.feature_importances_}).sort_values("imp",ascending=False).head(10)
rf_imp.to_csv("rf_top10特征.csv",encoding="utf-8-sig",index=False)
plt.figure(figsize=(9,5))
plt.barh(rf_imp["name"][::-1],rf_imp["imp"][::-1])
plt.title("随机森林TOP10关键特征")
plt.savefig("rf_feature_top10.png",dpi=300)

#LGB TOP10特征
lgb = lgb_pipe.named_steps["model"]
lgb_imp = pd.DataFrame({"name":feature_names,"imp":lgb.feature_importances_}).sort_values("imp",ascending=False).head(10)
lgb_imp.to_csv("lgb_top10特征.csv",encoding="utf-8-sig",index=False)
plt.figure(figsize=(9,5))
plt.barh(lgb_imp["name"][::-1],lgb_imp["imp"][::-1])
plt.title("LightGBM TOP10关键特征")
plt.savefig("lgb_feature_top10.png",dpi=300)

#Lasso系数
lasso = lasso_pipe.named_steps["model"]
lasso_df = pd.DataFrame({"name":feature_names,"coef":lasso.coef_})
lasso_df["影响方向"] = lasso_df["coef"].map(lambda x:"正向抬价" if x>0 else "负向降价")
lasso_df.to_csv("lasso全量系数.csv",encoding="utf-8-sig",index=False)