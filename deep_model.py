"""PyTorch MLP regression for the Ames Housing cleaned dataset.

This script is the C-task deliverable:
- uses A's cleaned CSV instead of the raw dataset;
- prevents target leakage by removing both SalePrice and LogSalePrice from X;
- trains several MLP settings with early stopping;
- reports RMSE/R2 and exports figures for the final project report.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RANDOM_STATE = 42
DATA_PATH = Path("AmesHousing_Cleaned.csv")
TRADITIONAL_RESULTS_PATH = Path("model_results.csv")
OUTPUT_DIR = Path("output/deep_model")
REPORT_DIR = Path("report")


@dataclass(frozen=True)
class MLPConfig:
    name: str
    hidden_dims: tuple[int, ...]
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int = 300
    patience: int = 35


CONFIGS = [
    MLPConfig(
        name="MLP_small",
        hidden_dims=(128, 64),
        dropout=0.10,
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=64,
    ),
    MLPConfig(
        name="MLP_medium",
        hidden_dims=(256, 128, 64),
        dropout=0.15,
        learning_rate=8e-4,
        weight_decay=1e-5,
        batch_size=64,
    ),
    MLPConfig(
        name="MLP_regularized",
        hidden_dims=(256, 128),
        dropout=0.25,
        learning_rate=8e-4,
        weight_decay=1e-4,
        batch_size=64,
    ),
]


class MLPRegressor(nn.Module):
    """Fully connected network for tabular regression."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_regression_bins(y: pd.Series, bins: int = 10) -> pd.Series | None:
    """Create quantile bins so regression splits keep target distribution stable."""

    try:
        return pd.qcut(y, q=bins, labels=False, duplicates="drop")
    except ValueError:
        return None


def split_data(
    x: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    bins = make_regression_bins(y)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=bins,
    )

    train_val_bins = make_regression_bins(y_train_val)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_val_bins,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def load_clean_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Missing cleaned dataset: {path}")

    df = pd.read_csv(path)
    if "LogSalePrice" in df.columns:
        target = df["LogSalePrice"].astype("float32")
    elif "SalePrice" in df.columns:
        target = np.log(df["SalePrice"].astype("float32"))
    else:
        raise ValueError("The cleaned dataset must contain SalePrice or LogSalePrice.")

    leakage_columns = ["SalePrice", "LogSalePrice"]
    features = df.drop(columns=[c for c in leakage_columns if c in df.columns])
    return features, target


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = x_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in x_train.columns if c not in categorical_cols]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def prepare_arrays() -> dict[str, object]:
    features, target = load_clean_data()
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(features, target)

    preprocessor = build_preprocessor(x_train)
    x_train_arr = preprocessor.fit_transform(x_train).astype("float32")
    x_val_arr = preprocessor.transform(x_val).astype("float32")
    x_test_arr = preprocessor.transform(x_test).astype("float32")

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    y_train_scaled = ((y_train - y_mean) / y_std).to_numpy(dtype="float32")
    y_val_scaled = ((y_val - y_mean) / y_std).to_numpy(dtype="float32")
    y_test_scaled = ((y_test - y_mean) / y_std).to_numpy(dtype="float32")

    return {
        "x_train": x_train_arr,
        "x_val": x_val_arr,
        "x_test": x_test_arr,
        "y_train": y_train.to_numpy(dtype="float32"),
        "y_val": y_val.to_numpy(dtype="float32"),
        "y_test": y_test.to_numpy(dtype="float32"),
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "y_test_scaled": y_test_scaled,
        "y_mean": y_mean,
        "y_std": y_std,
        "preprocessor": preprocessor,
        "feature_count": x_train_arr.shape[1],
        "raw_feature_count": features.shape[1],
        "row_count": features.shape[0],
    }


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def inverse_target(y_scaled: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    return y_scaled * y_std + y_mean


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(
    model: nn.Module,
    x: np.ndarray,
    y_log: np.ndarray,
    y_mean: float,
    y_std: float,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(x).to(device)).cpu().numpy()
    pred_log = inverse_target(pred_scaled, y_mean, y_std)
    rmse = rmse_score(y_log, pred_log)
    r2 = r2_score(y_log, pred_log)
    return float(rmse), float(r2), pred_log


def train_one_config(data: dict[str, object], config: MLPConfig, device: torch.device) -> dict[str, object]:
    set_global_seed()
    model = MLPRegressor(
        input_dim=int(data["feature_count"]),
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = make_loader(
        data["x_train"],
        data["y_train_scaled"],
        batch_size=config.batch_size,
        shuffle=True,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    patience_left = config.patience
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_rmse, val_r2, _ = evaluate_model(
            model,
            data["x_val"],
            data["y_val"],
            float(data["y_mean"]),
            float(data["y_std"]),
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_rmse_log": val_rmse,
            "val_r2_log": val_r2,
        }
        history.append(row)

        if val_rmse < best_val_rmse - 1e-5:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            patience_left = config.patience
        else:
            patience_left -= 1

        if patience_left <= 0:
            break

    model.load_state_dict(best_state)
    test_rmse, test_r2, test_pred_log = evaluate_model(
        model,
        data["x_test"],
        data["y_test"],
        float(data["y_mean"]),
        float(data["y_std"]),
        device,
    )

    return {
        "config": config,
        "model": model,
        "history": pd.DataFrame(history),
        "best_val_rmse_log": best_val_rmse,
        "test_rmse_log": test_rmse,
        "test_r2_log": test_r2,
        "test_pred_log": test_pred_log,
    }


def plot_training_curve(history: pd.DataFrame, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(history["epoch"], history["train_loss"], label="Train loss", color="#1f77b4")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Scaled MSE loss")
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["val_rmse_log"], label="Validation RMSE", color="#d62728")
    ax2.set_ylabel("Validation RMSE (log)")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.set_title("MLP training curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_predictions(y_true_log: np.ndarray, y_pred_log: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_log, y_pred_log, alpha=0.65, edgecolor="none")
    low = min(y_true_log.min(), y_pred_log.min())
    high = max(y_true_log.max(), y_pred_log.max())
    ax.plot([low, high], [low, high], linestyle="--", color="#d62728", linewidth=2)
    ax.set_xlabel("True log SalePrice")
    ax.set_ylabel("Predicted log SalePrice")
    ax.set_title("MLP prediction vs true values")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_residuals(y_true_log: np.ndarray, y_pred_log: np.ndarray, output_path: Path) -> None:
    residuals = y_pred_log - y_true_log
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred_log, residuals, alpha=0.65, edgecolor="none")
    ax.axhline(0, linestyle="--", color="#d62728", linewidth=2)
    ax.set_xlabel("Predicted log SalePrice")
    ax.set_ylabel("Residual (predicted - true)")
    ax.set_title("MLP residual plot")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def build_comparison(best_result: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if TRADITIONAL_RESULTS_PATH.exists():
        traditional = pd.read_csv(TRADITIONAL_RESULTS_PATH)
        for _, row in traditional.iterrows():
            rows.append(
                {
                    "model": row["模型"] if "模型" in row else row.get("model", "Traditional"),
                    "rmse_log": float(row["RMSE"]),
                    "r2_log": float(row["R2"]),
                    "source": "traditional-model branch",
                }
            )

    config: MLPConfig = best_result["config"]
    rows.append(
        {
            "model": config.name,
            "rmse_log": float(best_result["test_rmse_log"]),
            "r2_log": float(best_result["test_r2_log"]),
            "source": "deep-model branch",
        }
    )
    return pd.DataFrame(rows)


def plot_model_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = ["#4c78a8" if source != "deep-model branch" else "#f58518" for source in comparison["source"]]

    axes[0].bar(comparison["model"], comparison["rmse_log"], color=colors)
    axes[0].set_title("Reported RMSE comparison (lower is better)")
    axes[0].set_ylabel("RMSE on log SalePrice")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(comparison["model"], comparison["r2_log"], color=colors)
    axes[1].set_title("Reported R2 comparison (higher is better)")
    axes[1].set_ylabel("R2 on log SalePrice")
    axes[1].tick_params(axis="x", rotation=25)

    fig.text(
        0.5,
        0.01,
        "Traditional scores are imported from B branch; confirm target-leakage handling before final submission.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_report(
    best_result: dict[str, object],
    data: dict[str, object],
    comparison: pd.DataFrame,
    all_results: pd.DataFrame,
) -> None:
    config: MLPConfig = best_result["config"]
    y_true = data["y_test"]
    y_pred = best_result["test_pred_log"]
    rmse_price = rmse_score(np.exp(y_true), np.exp(y_pred))
    r2_price = r2_score(np.exp(y_true), np.exp(y_pred))

    report = f"""# C同学任务：PyTorch深度学习建模与结果可视化

## 数据与防泄露处理

本部分基于A同学提交的 `AmesHousing_Cleaned.csv`，共 `{data["row_count"]}` 条样本。目标变量使用清洗阶段生成的 `LogSalePrice`；建模输入中同时删除 `SalePrice` 和 `LogSalePrice`，避免把目标值或其等价变换作为特征造成数据泄露。

数据划分采用固定随机种子 `42`，并按目标变量分位数分箱后进行训练集、验证集、测试集划分，使各集合的房价分布尽量一致。数值特征使用 `StandardScaler` 标准化，分类特征使用 `OneHotEncoder(handle_unknown="ignore")` 编码。

## MLP模型设计

使用PyTorch搭建多层感知机回归网络，结构由全连接层、BatchNorm、ReLU和Dropout组成。训练时对目标变量做标准化，预测后还原到对数房价尺度计算指标。

本次尝试了 `{len(CONFIGS)}` 组超参数，最优模型为 `{config.name}`：

- 隐藏层：`{config.hidden_dims}`
- Dropout：`{config.dropout}`
- 学习率：`{config.learning_rate}`
- 权重衰减：`{config.weight_decay}`
- Batch size：`{config.batch_size}`
- Early stopping patience：`{config.patience}`

## 测试集结果

| 模型 | RMSE(log) | R2(log) | RMSE(原始价格) | R2(原始价格) |
|---|---:|---:|---:|---:|
| {config.name} | {best_result["test_rmse_log"]:.6f} | {best_result["test_r2_log"]:.6f} | {rmse_price:.2f} | {r2_price:.6f} |

全部MLP调参结果已保存到 `output/deep_model/mlp_tuning_results.csv`，与B同学传统模型的对比结果保存到 `output/deep_model/model_comparison.csv`。

## 图表清单

- `output/deep_model/01_training_curve.png`：训练损失和验证集RMSE曲线
- `output/deep_model/02_prediction_vs_true.png`：预测值与真实值散点图
- `output/deep_model/03_residual_plot.png`：残差图
- `output/deep_model/04_model_comparison.png`：传统模型与MLP性能对比柱状图

## 报告整合说明

B分支的 `model_results.csv` 被用于生成性能对比图。需要注意：A的清洗数据同时包含 `SalePrice` 和 `LogSalePrice`，传统模型脚本若只删除 `SalePrice`，可能把 `LogSalePrice` 当作输入特征，造成结果异常偏高。最终报告建议说明C部分已显式删除两个目标相关列，以保证深度学习评估不发生目标泄露。
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "C_deep_learning_report.md").write_text(report, encoding="utf-8")

    summary = {
        "best_model": config.name,
        "best_config": {
            "hidden_dims": config.hidden_dims,
            "dropout": config.dropout,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
        },
        "test_rmse_log": float(best_result["test_rmse_log"]),
        "test_r2_log": float(best_result["test_r2_log"]),
        "test_rmse_price": float(rmse_price),
        "test_r2_price": float(r2_price),
        "feature_count_after_encoding": int(data["feature_count"]),
        "raw_feature_count": int(data["raw_feature_count"]),
        "all_results": all_results.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "deep_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    set_global_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = prepare_arrays()
    print(
        f"Rows: {data['row_count']}, raw features: {data['raw_feature_count']}, "
        f"encoded features: {data['feature_count']}"
    )

    results = []
    for config in CONFIGS:
        print(f"\nTraining {config.name}...")
        result = train_one_config(data, config, device)
        results.append(result)
        print(
            f"{config.name}: val RMSE={result['best_val_rmse_log']:.6f}, "
            f"test RMSE={result['test_rmse_log']:.6f}, test R2={result['test_r2_log']:.6f}"
        )

    best_result = min(results, key=lambda item: item["best_val_rmse_log"])
    best_config: MLPConfig = best_result["config"]

    tuning_results = pd.DataFrame(
        [
            {
                "model": item["config"].name,
                "hidden_dims": str(item["config"].hidden_dims),
                "dropout": item["config"].dropout,
                "learning_rate": item["config"].learning_rate,
                "weight_decay": item["config"].weight_decay,
                "batch_size": item["config"].batch_size,
                "epochs_ran": len(item["history"]),
                "best_val_rmse_log": item["best_val_rmse_log"],
                "test_rmse_log": item["test_rmse_log"],
                "test_r2_log": item["test_r2_log"],
            }
            for item in results
        ]
    )
    tuning_results.to_csv(OUTPUT_DIR / "mlp_tuning_results.csv", index=False, encoding="utf-8-sig")

    best_result["history"].to_csv(OUTPUT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")
    prediction_df = pd.DataFrame(
        {
            "true_log_saleprice": data["y_test"],
            "pred_log_saleprice": best_result["test_pred_log"],
            "true_saleprice": np.exp(data["y_test"]),
            "pred_saleprice": np.exp(best_result["test_pred_log"]),
            "residual_log": best_result["test_pred_log"] - data["y_test"],
        }
    )
    prediction_df.to_csv(OUTPUT_DIR / "deep_model_predictions.csv", index=False, encoding="utf-8-sig")

    comparison = build_comparison(best_result)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

    plot_training_curve(best_result["history"], OUTPUT_DIR / "01_training_curve.png")
    plot_predictions(data["y_test"], best_result["test_pred_log"], OUTPUT_DIR / "02_prediction_vs_true.png")
    plot_residuals(data["y_test"], best_result["test_pred_log"], OUTPUT_DIR / "03_residual_plot.png")
    plot_model_comparison(comparison, OUTPUT_DIR / "04_model_comparison.png")

    torch.save(best_result["model"].state_dict(), OUTPUT_DIR / "mlp_state_dict.pt")
    joblib.dump(data["preprocessor"], OUTPUT_DIR / "preprocessor.pkl")
    write_report(best_result, data, comparison, tuning_results)

    print(f"\nBest model: {best_config.name}")
    print(f"Test RMSE(log): {best_result['test_rmse_log']:.6f}")
    print(f"Test R2(log): {best_result['test_r2_log']:.6f}")
    print(f"Artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
