from __future__ import annotations

MULTICLASS_LABELS = [0, 1, 2, 3]


def _import_library(module_name: str, class_name: str):
    import importlib
    import sys

    for name, mod in list(sys.modules.items()):
        mod_file = (getattr(mod, "__file__", "") or "").replace("\\", "/")
        if name == module_name and ("binarymodels/" in mod_file or "multiclassmodels/" in mod_file):
            del sys.modules[name]

    clean_path = [
        p for p in sys.path
        if not (p or "").replace("\\", "/").rstrip("/").endswith(("binarymodels", "multiclassmodels"))
    ]
    old_path = sys.path
    sys.path = clean_path
    try:
        module = importlib.import_module(module_name)
    finally:
        sys.path = old_path
    return getattr(module, class_name)


def build_lr(random_state: int, task: str = "binary"):
    from sklearn.linear_model import LogisticRegression

    solver = "lbfgs" if task == "multiclass" else "lbfgs"
    return LogisticRegression(
        C=10,
        solver=solver,
        max_iter=5000,
        class_weight="balanced",
        random_state=random_state,
    )


def build_svm(random_state: int, task: str = "binary"):
    from sklearn.svm import SVC

    params = {
        "C": 10,
        "kernel": "rbf",
        "gamma": "scale",
        "class_weight": "balanced",
        "probability": True,
        "random_state": random_state,
    }
    if task == "multiclass":
        params["decision_function_shape"] = "ovr"
    return SVC(**params)


def build_rf(random_state: int, task: str = "binary"):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
    )


def build_xgb(random_state: int, task: str = "binary"):
    XGBClassifier = _import_library("xgboost", "XGBClassifier")

    eval_metric = "logloss" if task == "binary" else "mlogloss"
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric=eval_metric,
        verbosity=0,
        random_state=random_state,
        n_jobs=-1,
    )


def build_catboost(random_state: int, task: str = "binary"):
    CatBoostClassifier = _import_library("catboost", "CatBoostClassifier")

    loss_function = "Logloss" if task == "binary" else "MultiClass"
    return CatBoostClassifier(
        iterations=250,
        learning_rate=0.05,
        depth=6,
        loss_function=loss_function,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        auto_class_weights="Balanced",
        l2_leaf_reg=3.0,
        thread_count=-1,
        od_type="Iter",
        od_wait=15,
    )


BUILDERS = {
    "lr": build_lr,
    "svm": build_svm,
    "rf": build_rf,
    "xgb": build_xgb,
    "catboost": build_catboost,
}


def build_model(name: str, task: str, random_state: int):
    if name not in BUILDERS:
        raise ValueError(f"Unknown model: {name}")
    return BUILDERS[name](random_state, task)
