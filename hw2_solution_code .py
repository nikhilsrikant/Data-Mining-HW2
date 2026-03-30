from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "titanic_train.csv"
OUTPUT_DIR = HERE


def ensure_data() -> Path:
    if DATA_PATH.exists():
        return DATA_PATH
    print(f"Downloading Titanic training data to {DATA_PATH} ...")
    urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).fillna("Unknown")
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss").replace("Mme", "Mrs")
    rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["Pclass"] = df["Pclass"].astype(str)
    return df


def build_preprocessor() -> ColumnTransformer:
    categorical = ["Pclass", "Sex", "Embarked", "Title"]
    numeric = ["Age", "Fare", "FamilySize", "IsAlone", "CabinKnown"]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def build_dataset() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(ensure_data())
    X = add_features(data.drop(columns=["Survived"]))
    X = X[["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone", "CabinKnown", "Title"]]
    y = data["Survived"]
    return X, y


def tune_decision_tree(X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> GridSearchCV:
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", DecisionTreeClassifier(random_state=42)),
    ])
    param_grid = {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [3, 4, 5, 6],
        "classifier__min_samples_split": [2, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__ccp_alpha": [0.0, 0.001, 0.005],
    }
    search = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=1)
    search.fit(X, y)
    return search


def tune_random_forest(X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> GridSearchCV:
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=1)),
    ])
    param_grid = {
        "classifier__n_estimators": [200],
        "classifier__max_depth": [4, 6, 8],
        "classifier__min_samples_split": [2, 10],
        "classifier__min_samples_leaf": [1, 2],
        "classifier__max_features": ["sqrt"],
    }
    search = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=1)
    search.fit(X, y)
    return search


def save_decision_tree_plot(best_estimator: Pipeline) -> None:
    best_estimator.fit(X, y)
    feature_names = best_estimator.named_steps["preprocessor"].get_feature_names_out()
    clf = best_estimator.named_steps["classifier"]
    plt.figure(figsize=(28, 14))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["Did not survive", "Survived"],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=True,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decision_tree_plot.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_task6_plot() -> None:
    points = [(-1, 1, -1), (-1, -1, +1), (1, -1, +1), (1, 1, -1)]
    plt.figure(figsize=(6, 6))
    for z1, z2, label in points:
        marker = "o" if label == 1 else "s"
        plt.scatter(z1, z2, s=150, marker=marker)
        sign = "+" if label == 1 else "-"
        plt.text(z1 + 0.05, z2 + 0.08, f"{sign} ({z1}, {z2})", fontsize=10)
    plt.axhline(0, linewidth=2, linestyle="-", label="Maximal margin separator: x1x2 = 0")
    plt.axhline(1, linewidth=1, linestyle="--", label="Margin boundaries")
    plt.axhline(-1, linewidth=1, linestyle="--")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel("z1 = x1")
    plt.ylabel("z2 = x1x2")
    plt.title("Task 6: Points in mapped space (x1, x1x2)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task6_svm_mapped_space.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_accuracy_plot(dt_mean: float, rf_mean: float) -> None:
    plt.figure(figsize=(6, 4.5))
    plt.bar(["Decision Tree", "Random Forest"], [dt_mean, rf_mean])
    plt.ylim(0.75, 0.86)
    plt.ylabel("5-fold CV accuracy")
    plt.title("Task 1: Model comparison on Titanic training data")
    for i, value in enumerate([dt_mean, rf_mean]):
        plt.text(i, value + 0.002, f"{value:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task1_accuracy_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    X, y = build_dataset()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Tuning decision tree ...")
    dt_search = tune_decision_tree(X, y, cv)
    dt_scores = cross_val_score(dt_search.best_estimator_, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    print("Best decision tree parameters:")
    print(dt_search.best_params_)
    print("Decision tree 5-fold accuracies:", dt_scores)
    print("Decision tree mean accuracy:", dt_scores.mean())

    print("Tuning random forest ...")
    rf_search = tune_random_forest(X, y, cv)
    rf_scores = cross_val_score(rf_search.best_estimator_, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    print("Best random forest parameters:")
    print(rf_search.best_params_)
    print("Random forest 5-fold accuracies:", rf_scores)
    print("Random forest mean accuracy:", rf_scores.mean())

    save_decision_tree_plot(dt_search.best_estimator_)
    save_task6_plot()
    save_accuracy_plot(float(dt_scores.mean()), float(rf_scores.mean()))

    results = {
        "decision_tree_best_params": dt_search.best_params_,
        "decision_tree_scores": dt_scores.tolist(),
        "decision_tree_mean_accuracy": float(dt_scores.mean()),
        "random_forest_best_params": rf_search.best_params_,
        "random_forest_scores": rf_scores.tolist(),
        "random_forest_mean_accuracy": float(rf_scores.mean()),
    }
    with open(OUTPUT_DIR / "task1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nArtifacts written to: {OUTPUT_DIR}")
