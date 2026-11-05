"""Compare Bagging and Boosting regressors on Vega's wheat dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from vega_datasets import data


OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42


def main():
    df = data.wheat().dropna(subset=["year", "wheat", "wages"]).copy()
    X = df[["year", "wages"]]
    y = df["wheat"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    models = {
        "Referencia (media)": DummyRegressor(strategy="mean"),
        "Bagging": BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE),
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }

    predictions = {}
    metric_rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        metric_rows.append(
            {
                "Modelo": name,
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R2": r2_score(y_test, y_pred),
            }
        )

    results = pd.DataFrame(metric_rows).set_index("Modelo").sort_values("RMSE")
    results.to_csv(OUTPUT_DIR / "wheat_model_results.csv", float_format="%.6f")

    prediction_results = pd.DataFrame(
        {
            "year": X_test["year"].to_numpy(),
            "wages": X_test["wages"].to_numpy(),
            "wheat_real": y_test.to_numpy(),
            **{f"wheat_estimado_{name}": pred for name, pred in predictions.items()},
        }
    ).sort_values("year")
    prediction_results.to_csv(
        OUTPUT_DIR / "wheat_test_predictions.csv", index=False, float_format="%.6f"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, y_pred in predictions.items():
        axes[0].scatter(y_test, y_pred, alpha=0.8, label=name)

    limits = [
        min(float(y_test.min()), *(float(values.min()) for values in predictions.values())),
        max(float(y_test.max()), *(float(values.max()) for values in predictions.values())),
    ]
    axes[0].plot(limits, limits, "k--", label="Estimativa perfeita")
    axes[0].set(
        xlabel="Valor real de wheat",
        ylabel="Valor estimado de wheat",
        title="Valores reais x estimados",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    results["RMSE"].plot(kind="bar", ax=axes[1], color="#4472C4")
    axes[1].set(
        xlabel="Modelo",
        ylabel="RMSE (unidade de wheat)",
        title="Erro por modelo",
    )
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    for position, value in enumerate(results["RMSE"]):
        axes[1].text(position, value, f" {value:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "wheat_model_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Dataset: {len(df)} observacoes; treino: {len(X_train)}; teste: {len(X_test)}")
    print("\nMetricas no conjunto de teste:")
    print(results.round(3).to_string())
    print(f"\nMenor RMSE neste teste: {results['RMSE'].idxmin()}")
    print("\nArquivos gerados:")
    print("  wheat_model_results.csv")
    print("  wheat_test_predictions.csv")
    print("  wheat_model_comparison.png")


if __name__ == "__main__":
    main()
