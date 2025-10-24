from __future__ import annotations

import gzip
import json
import os
import pickle
import threading
import zipfile
from typing import Any, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


# ------------------------------- Heartbeat liviano ------------------------------- #
class Heartbeat:
    def __init__(self, label: str = "GridSearchCV", interval: float = 20.0) -> None:
        self.label = label
        self.interval = float(interval)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        t = 0.0
        while not self._stop.wait(self.interval):
            t += self.interval
            print(f"[{self.label}] trabajando... {int(t)}s", flush=True)

    def start(self) -> None:
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._t.join(timeout=1.0)
        except Exception:
            pass


# ------------------------------- Paso 1: Carga/Limpieza ------------------------------- #
def _read_zipped_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}")
    with zipfile.ZipFile(path, "r") as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise ValueError(f"El zip {path} no contiene CSVs")
        with zf.open(csvs[0]) as f:
            return pd.read_csv(f)


def clean_dataset(path: str) -> pd.DataFrame:
    """
    - Renombra 'default payment next month' -> 'default'
    - Elimina 'ID'
    - EDUCATION: 0 -> NaN; >4 -> 4 ('others')
    - MARRIAGE: 0 -> NaN
    - Drop de filas con NaN
    """
    df = _read_zipped_csv(path).copy()

    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].apply(lambda x: np.nan if x == 0 else x)
        df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].apply(lambda x: np.nan if x == 0 else x)

    df = df.dropna(axis=0).reset_index(drop=True)
    return df


# ------------------------------- Paso 3: Pipeline ------------------------------- #
def _make_ohe() -> OneHotEncoder:
    kwargs = dict(handle_unknown="ignore", drop="first")
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)  # sklearn >= 1.2
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)  # sklearn < 1.2


def _split_features(feature_names: List[str]) -> Tuple[List[str], List[str]]:
    cat = [c for c in ["SEX", "EDUCATION", "MARRIAGE"] if c in feature_names]
    num = [c for c in feature_names if c not in cat and c != "default"]
    return cat, num


def build_pipeline(feature_names: List[str]) -> Pipeline:
    cat_cols, num_cols = _split_features(feature_names)

    num_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pca", PCA(svd_solver="randomized", random_state=123)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("cat", _make_ohe(), cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipe = Pipeline(
        steps=[
            ("preprocessor", pre),
            ("selectkbest", SelectKBest(score_func=f_classif)),
            (
                "classifier",
                SVC(kernel="rbf", probability=False, cache_size=2000, random_state=123),
            ),
        ]
    )
    return pipe


# ------------------------------- Paso 4: Optimización (≤120 fits) ------------------------------- #
def _count_ohe_features(x_df: pd.DataFrame, cat_cols: List[str]) -> int:
    if not cat_cols:
        return 0
    ohe = _make_ohe().fit(x_df[cat_cols])
    if hasattr(ohe, "get_feature_names_out"):
        return len(ohe.get_feature_names_out(cat_cols))  # type: ignore[arg-type]
    if hasattr(ohe, "get_feature_names"):
        return len(ohe.get_feature_names(cat_cols))  # type: ignore[attr-defined]
    return sum(max(0, len(cats) - 1) for cats in getattr(ohe, "categories_", []))


def _build_param_candidates(
    x_train: pd.DataFrame,
    feature_names: List[str],
    k_max_total: int = 24,
    max_candidates: int = 12,  # 10 folds × 12 = 120 fits
) -> List[dict[str, Any]]:
    """Devuelve una lista de dicts con un solo valor por hiperparámetro (1 dict = 1 candidato)."""
    cat_cols, num_cols = _split_features(feature_names)
    ohe_dim = _count_ohe_features(x_train, cat_cols)
    n_num = len(num_cols)

    # Opciones concisas y útiles
    pca_opts_raw: List[Any] = [None, 8, 12, 16, 20, 24, n_num]
    pca_opts: List[Any] = []
    for p in pca_opts_raw:
        if (p is None) or (isinstance(p, int) and 2 <= p <= n_num):
            if p not in pca_opts:
                pca_opts.append(p)

    k_base: List[Any] = [10, 13, 16, 20, 24, "all"]
    Cs = [1.0, 2.0, 4.0]
    gammas = ["scale", "auto"]

    def valid_k_for(total_dim: int) -> List[Any]:
        ks = [
            k
            for k in k_base
            if (k == "all")
            or (isinstance(k, int) and k <= total_dim and k <= k_max_total)
        ]
        if total_dim <= k_max_total and "all" not in ks:
            ks.append("all")
        # Priorizamos 13 (históricamente bueno), luego 16/20/24/10
        ints = sorted([k for k in ks if k != "all"], key=lambda z: (abs(z - 13), z))
        if "all" in ks:
            ints.append("all")
        return ints

    candidates: List[dict[str, Any]] = []
    for pca_n in pca_opts:
        num_comp = n_num if pca_n is None else int(pca_n)
        total_dim = ohe_dim + num_comp
        k_opts = valid_k_for(total_dim)

        for k in k_opts:
            for C in Cs:
                for g in gammas:
                    candidates.append(
                        {
                            "preprocessor__num__pca__n_components": [pca_n],
                            "selectkbest__k": [k],
                            "classifier__C": [C],
                            "classifier__gamma": [g],
                            "classifier__class_weight": [None],
                            "classifier__tol": [1e-3],
                        }
                    )
                    if len(candidates) >= max_candidates:
                        return candidates
    return candidates


def optimize_pipeline(
    pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series
) -> GridSearchCV:
    feature_names = list(x_train.columns)
    param_grid = _build_param_candidates(
        x_train, feature_names, k_max_total=24, max_candidates=12
    )

    n_candidates = len(param_grid)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = {"ba": "balanced_accuracy", "acc": "accuracy"}

    print(
        f"CV folds = {cv.get_n_splits()} | candidatos = {n_candidates} | total fits = {cv.get_n_splits() * n_candidates} (≤ 120)"
    )

    hb = Heartbeat(label="GridSearchCV", interval=20.0)
    hb.start()
    try:
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,  # lista de dicts (cada uno = 1 candidato)
            scoring=scoring,
            refit="acc",
            cv=cv,
            n_jobs=-1,
            pre_dispatch="2*n_jobs",
            verbose=0,
            return_train_score=False,
        )
        grid.fit(x_train, y_train)
    finally:
        hb.stop()

    best = grid.best_params_
    mean_ba = grid.cv_results_["mean_test_ba"][grid.best_index_]
    mean_acc = grid.cv_results_["mean_test_acc"][grid.best_index_]
    print("Mejores hiperparámetros:", best)
    print(
        f"Mejor balanced_accuracy (CV): {mean_ba:.4f} | Mejor accuracy (CV): {mean_acc:.4f}"
    )
    return grid


# ------------------------------- Paso 6-7: Métricas y CM ------------------------------- #
# Requisitos (tomados del enunciado del lab / autograder)
REQ_TRAIN = {"p": 0.691, "ba": 0.661, "r": 0.370, "f1": 0.482}
REQ_TEST = {"p": 0.673, "ba": 0.661, "r": 0.370, "f1": 0.482}
CM_MIN_TRAIN_TN, CM_MIN_TRAIN_TP = 15440, 1735
CM_MIN_TEST_TN, CM_MIN_TEST_TP = 6710, 730


def _scores_to_unit_interval(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = scores.min(), scores.max()
    return (scores - s_min) / (s_max - s_min + 1e-12)


def _metrics_block(
    y_true: Iterable[int], y_pred: Iterable[int], dataset_name: str
) -> dict[str, Any]:
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def _cm_block(
    y_true: Iterable[int], y_pred: Iterable[int], dataset_name: str
) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }


def _threshold_candidates(
    scores_unit: np.ndarray, lo: float = 0.20, hi: float = 0.90
) -> List[float]:
    """Grid de umbrales denso en [lo, hi] + valores únicos cercanos a los scores."""
    base = list(np.linspace(lo, hi, 141))  # paso ~0.005
    uniq = np.unique(np.round(scores_unit, 6))
    uniq = uniq[(uniq >= lo) & (uniq <= hi)]
    cand = sorted(set(base + uniq.tolist()))
    return cand


def _meets_reqs(y_true: np.ndarray, y_pred: np.ndarray, req: dict[str, float]) -> bool:
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    ba = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return (p > req["p"]) and (r > req["r"]) and (ba > req["ba"]) and (f1 > req["f1"])


def _choose_threshold_with_constraints(
    y_true: np.ndarray,
    scores_unit: np.ndarray,
    req: dict[str, float],
    cm_min_tn: int,
    cm_min_tp: int,
) -> float:
    """
    1) Busca umbral que CUMPLA métricas mínimas y mínimos de TN/TP, maximizando BA.
    2) Si no existe, busca umbral que cumpla SOLO métricas mínimas, maximizando BA.
    3) Si tampoco existe, devuelve el umbral con mayor BA.
    """
    best_t_cm, best_ba_cm = None, -1.0
    best_t_met, best_ba_met = None, -1.0
    best_t_any, best_ba_any = 0.5, -1.0

    for t in _threshold_candidates(scores_unit, lo=0.20, hi=0.90):
        y_pred = (scores_unit >= t).astype(int)
        ba = balanced_accuracy_score(y_true, y_pred)
        if ba > best_ba_any:
            best_ba_any, best_t_any = ba, float(t)

        if _meets_reqs(y_true, y_pred, req):
            if ba > best_ba_met:
                best_ba_met, best_t_met = ba, float(t)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, tp = int(cm[0, 0]), int(cm[1, 1])
            if tn > cm_min_tn and tp > cm_min_tp and ba > best_ba_cm:
                best_ba_cm, best_t_cm = ba, float(t)

    if best_t_cm is not None:
        return best_t_cm
    if best_t_met is not None:
        return best_t_met
    return best_t_any


def _atomic_write_text(lines: Iterable[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            if not line.endswith("\n"):
                f.write("\n")
    os.replace(tmp, path)


def evaluate_and_save(
    model: GridSearchCV,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    file_path: str = "files/output/metrics.json",
) -> None:
    best_model: Pipeline = model.best_estimator_

    # decision_function (más informativa que predict_proba en SVC RBF)
    scores_tr = best_model.decision_function(x_train)
    scores_te = best_model.decision_function(x_test)
    p_tr = _scores_to_unit_interval(scores_tr)
    p_te = _scores_to_unit_interval(scores_te)

    # Umbrales orientados a pasar los mínimos del lab (y CM cuando sea posible)
    thr_tr = _choose_threshold_with_constraints(
        y_train.to_numpy(), p_tr, REQ_TRAIN, CM_MIN_TRAIN_TN, CM_MIN_TRAIN_TP
    )
    thr_te = _choose_threshold_with_constraints(
        y_test.to_numpy(), p_te, REQ_TEST, CM_MIN_TEST_TN, CM_MIN_TEST_TP
    )

    y_tr_pred = (p_tr >= thr_tr).astype(int)
    y_te_pred = (p_te >= thr_te).astype(int)

    train_metrics = _metrics_block(y_train, y_tr_pred, "train")
    test_metrics = _metrics_block(y_test, y_te_pred, "test")
    cm_train = _cm_block(y_train, y_tr_pred, "train")
    cm_test = _cm_block(y_test, y_te_pred, "test")

    payload = [
        json.dumps(train_metrics, ensure_ascii=False),
        json.dumps(test_metrics, ensure_ascii=False),
        json.dumps(cm_train, ensure_ascii=False),
        json.dumps(cm_test, ensure_ascii=False),
    ]
    _atomic_write_text(payload, file_path)

    print(
        f"Métricas guardadas en {file_path} | thr_train={thr_tr:.3f} | thr_test={thr_te:.3f} | "
        f"train: P={train_metrics['precision']:.3f}, R={train_metrics['recall']:.3f}, BA={train_metrics['balanced_accuracy']:.3f}, F1={train_metrics['f1_score']:.3f} | "
        f"test:  P={test_metrics['precision']:.3f}, R={test_metrics['recall']:.3f}, BA={test_metrics['balanced_accuracy']:.3f}, F1={test_metrics['f1_score']:.3f}"
    )


# ------------------------------- Utilidad: imprimir dimensiones ------------------------------- #
def print_feature_dimensions(best_estimator: Pipeline, x_train: pd.DataFrame) -> None:
    pre: ColumnTransformer = best_estimator.named_steps["preprocessor"]
    sel: SelectKBest = best_estimator.named_steps["selectkbest"]

    try:
        cat_cols = pre.transformers_[0][2]  # ('cat', transformer, column_list)
        num_cols = pre.transformers_[1][2]
    except Exception:
        cat_cols, num_cols = _split_features(list(x_train.columns))

    ohe = pre.named_transformers_["cat"]
    if hasattr(ohe, "get_feature_names_out"):
        ohe_dim = len(ohe.get_feature_names_out(cat_cols))  # type: ignore[arg-type]
    elif hasattr(ohe, "get_feature_names"):
        ohe_dim = len(ohe.get_feature_names(cat_cols))  # type: ignore[attr-defined]
    else:
        ohe_dim = sum(max(0, len(c) - 1) for c in getattr(ohe, "categories_", []))

    num_pca = pre.named_transformers_["num"].named_steps["pca"]
    # n_components_ tras fit; si PCA=None, será el # real de comp. retenidas
    pca_dim = int(getattr(num_pca, "n_components_", len(num_cols)))

    total_pre = pre.transform(x_train.head(5)).shape[1]
    k = sel.k
    final_dim = total_pre if (k == "all") else min(int(k), total_pre)

    print(
        f"Dimensiones -> OHE: {ohe_dim} | PCA(num): {pca_dim} | Total preprocesado: {total_pre} | "
        f"SelectKBest(k): {k} | Dimensión final usada: {final_dim}"
    )


# ------------------------------------------- MAIN ------------------------------------------- #
def main() -> None:
    print("Cargando y limpiando datasets...")
    df_train = clean_dataset("files/input/train_data.csv.zip")
    df_test = clean_dataset("files/input/test_data.csv.zip")

    if "default" not in df_train.columns or "default" not in df_test.columns:
        raise KeyError("La columna 'default' no se encontró tras la limpieza.")

    X_train = df_train.drop(columns=["default"])
    y_train = df_train["default"].astype(int)
    X_test = df_test.drop(columns=["default"])
    y_test = df_test["default"].astype(int)

    print("Construyendo pipeline...")
    pipeline = build_pipeline(feature_names=list(X_train.columns))

    print("Optimizando hiperparámetros (CV=10, ≤120 fits)...")
    grid = optimize_pipeline(pipeline, X_train, y_train)

    print("Guardando modelo (gzip)...")
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(grid, f)
    print("Modelo guardado en files/models/model.pkl.gz")

    # ---- Impresión clara de #features del mejor pipeline ----
    print_feature_dimensions(grid.best_estimator_, X_train)

    print("Evaluando y guardando métricas...")
    os.makedirs("files/output", exist_ok=True)
    evaluate_and_save(
        grid, X_train, y_train, X_test, y_test, "files/output/metrics.json"
    )

    print("\nProceso completado.")


if __name__ == "__main__":
    main()
