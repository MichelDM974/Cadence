"""Ticket #10 : poids d'ensemble LightGBM / SARIMAX propres à chaque série,
comparés au 50/50 de la page, sur les fenêtres du banc (3 récentes + an-1).

Pour noter honnêtement une fenêtre, les poids ne s'apprennent que sur des
fenêtres **antérieures** (`--train-windows` fenêtres de backtest qui la
précèdent, chacune entièrement connue à sa date de coupe) : c'est ce qu'on
aurait au moment de soumettre. Méthodes comparées :

- `fixe50` : la page (référence) ;
- `global` : un seul poids, le meilleur sur les fenêtres d'apprentissage ;
- `serie` : le meilleur poids de chaque série sur ces fenêtres, rétréci
  vers le poids global (16 jours par série, très bruité sinon) ;
- `fforma` : méta-modèle LightGBM qui prédit le poids d'une série à partir
  de descripteurs de son historique (niveau, bruit, zéros, saisonnalité
  hebdomadaire, tendance, écart entre les deux membres), à la FFORMA
  (Montero-Manso et al., 2020) — généralise d'une série à l'autre au lieu
  de mémoriser chacune.

Les prévisions des membres passent par le cache de benchmark_kaggle.py
(mêmes clés) : relancer ne réentraîne rien.

    python benchmark_ensemble_weights.py store-sales --cache /tmp/cadence-cache

`--submit DIR` : au lieu du banc, écrit le fichier de soumission FFORMA,
méta-modèle appris sur les `--train-windows` fenêtres les plus récentes
(celles du banc : déjà en cache).
"""

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import api
import kaggle_competition as kc
from benchmark_kaggle import DATASETS, cache_member_forecasts, load

MEMBERS = ("lightgbm", "sarimax")
GRID = np.round(np.linspace(0, 1, 21), 2)  # part de LightGBM
ZERO_DEAD_DAYS = 14
SHRINK = (0.25, 0.5, 1.0)  # part du poids propre à la série dans `serie`


def window_data(train, spec, keys, horizon, scored_days, exog_long, fold_end):
    """Prévisions des deux membres, vérité et descripteurs d'une fenêtre
    (même enchaînement que api.kaggle_backtest)."""
    sub = train[train[spec.date_col] <= fold_end]
    entities = keys[spec.entity_cols].drop_duplicates()
    history, truth = kc.backtest_split(sub, spec, horizon, scored_days)
    targets = truth.merge(entities, on=spec.entity_cols)
    history = kc.pad_to(history, spec, history[spec.date_col].max())
    df = member_frame(targets, history, spec, horizon, exog_long, entities)
    return df, descriptors(history, spec, df)


def member_frame(targets, history, spec, horizon, exog_long, entities) -> pd.DataFrame:
    """Une ligne par ligne de `targets` (ordre conservé) : prévision de
    chaque membre, série éteinte ou non (`_alive`) et clé de série."""
    frames = {}
    for member in MEMBERS:
        rows = api._kaggle_forecast_rows({}, history, spec, horizon, member, {}, exog_long, entities)
        frames[member] = kc.predictions_frame(rows, spec).rename(columns={"prevision": member})
    on = [*spec.entity_cols, "date"]
    df = targets.copy()
    for col in spec.entity_cols:
        df[col] = df[col].astype(str)
    for member in MEMBERS:
        df = df.merge(frames[member], on=on, how="left")
    # un membre absent : l'autre seul (comme blend_predictions)
    df["lightgbm"] = df["lightgbm"].fillna(df["sarimax"])
    df["sarimax"] = df["sarimax"].fillna(df["lightgbm"])
    # aucun des deux : repli de la page (moyenne récente), quel que soit le poids
    empty = kc.predictions_frame([], spec)
    fallback, _ = kc.attach_predictions(df[on], empty, history, spec, clip_zero=False)
    for member in MEMBERS:
        df[member] = df[member].fillna(pd.Series(fallback.to_numpy(), index=df.index))
    dead = kc.zero_dead_series(
        df[[*on]].assign(prevision=1.0), history, spec, ZERO_DEAD_DAYS)
    df = df.merge(dead.rename(columns={"prevision": "_alive"}), on=on, how="left")
    df["_alive"] = df["_alive"].fillna(1.0)
    df["_key"] = df[spec.entity_cols].astype(str).agg("|".join, axis=1)
    return df


def descriptors(history, spec, df) -> pd.DataFrame:
    """Descripteurs par série, calculés sur le seul historique connu."""
    daily = history.groupby([*spec.entity_cols, spec.date_col], observed=True)[spec.target_col].sum().reset_index()
    daily["_key"] = daily[spec.entity_cols].astype(str).agg("|".join, axis=1)
    end = daily[spec.date_col].max()
    feats = {}
    for key, g in daily.groupby("_key", sort=False):
        y = g.set_index(spec.date_col)[spec.target_col].sort_index()
        nz = y[y > 0]
        start = nz.index.min() if len(nz) else end
        y = y[y.index >= start]
        recent = y[y.index > end - pd.Timedelta(days=90)]
        log = np.log1p(recent.clip(lower=0))
        dow = log.groupby(log.index.dayofweek).transform("mean")
        var = log.var()
        last28, prev = y[y.index > end - pd.Timedelta(days=28)], y[(y.index <= end - pd.Timedelta(days=28))
                                                                   & (y.index > end - pd.Timedelta(days=112))]
        feats[key] = {
            "length": (end - start).days,
            "level": float(np.log1p(recent.mean())) if len(recent) else 0.0,
            "cv": float(recent.std() / recent.mean()) if len(recent) and recent.mean() > 0 else 0.0,
            "zeros": float((recent == 0).mean()) if len(recent) else 1.0,
            "weekly": float(1 - ((log - dow).var() / var)) if var > 0 else 0.0,
            "trend": float(np.log1p(last28.mean()) - np.log1p(prev.mean())) if len(prev) else 0.0,
        }
    out = pd.DataFrame.from_dict(feats, orient="index")
    pred = df.groupby("_key")[["lightgbm", "sarimax"]].mean()
    out["gap"] = np.log1p(pred["lightgbm"].clip(lower=0)) - np.log1p(pred["sarimax"].clip(lower=0))
    return out.fillna(0.0)


def blend(df, weight) -> np.ndarray:
    """`weight` : scalaire ou part de LightGBM par ligne."""
    values = weight * df["lightgbm"].to_numpy() + (1 - weight) * df["sarimax"].to_numpy()
    return np.clip(values * df["_alive"].to_numpy(), 0, None)


def series_sse(df) -> pd.DataFrame:
    """Somme des carrés d'erreur log par série et par poids de la grille."""
    log_actual = np.log1p(df["actual"].clip(lower=0).to_numpy())
    cols = {w: (np.log1p(blend(df, w)) - log_actual) ** 2 for w in GRID}
    return pd.DataFrame(cols).groupby(df["_key"].to_numpy()).sum()


def rmsle(df, weight) -> float:
    return kc.compute_metrics(df["actual"].to_numpy(), blend(df, weight))["rmsle"]


def evaluate(target, history_windows):
    """Score de chaque méthode sur `target`, poids appris sur `history_windows`."""
    df, feats = target
    # séries absentes d'une fenêtre (pas encore ouvertes) : sommées sur les autres
    sse = pd.concat([series_sse(d) for d, _ in history_windows]).groupby(level=0).sum()
    w_global = float(sse.sum().idxmin())
    w_series = sse.idxmin(axis=1)
    scores = {"fixe50": rmsle(df, 0.5), "global": rmsle(df, w_global)}
    for s in SHRINK:
        w = (s * w_series + (1 - s) * w_global).reindex(df["_key"]).fillna(w_global).to_numpy()
        scores[f"serie{int(s * 100)}"] = rmsle(df, w)

    meta = fit_fforma(history_windows)
    w_meta = pd.Series(np.clip(meta.predict(feats), 0, 1), index=feats.index)
    scores["fforma"] = rmsle(df, w_meta.reindex(df["_key"]).fillna(w_global).to_numpy())
    scores["oracle"] = float(np.sqrt(series_sse(df).min(axis=1).sum() / len(df)))
    return scores, w_global


def fit_fforma(history_windows) -> lgb.Booster:
    """FFORMA : une ligne par (série, fenêtre d'apprentissage) ; cible = poids
    optimal, pondérée par ce que le choix du poids change pour la série."""
    xs, ys, ws = [], [], []
    for d, f in history_windows:
        # séries ouvertes pendant la fenêtre notée : aucun historique à décrire
        s = series_sse(d)
        s = s.loc[s.index.intersection(f.index)]
        xs.append(f.loc[s.index])
        ys.append(s.idxmin(axis=1).astype(float))
        ws.append(s.max(axis=1) - s.min(axis=1))
    x, y, sw = pd.concat(xs), pd.concat(ys), pd.concat(ws)
    return lgb.train(
        {"objective": "regression", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 50,
         "feature_fraction": 0.8, "verbose": -1, "seed": 0},
        lgb.Dataset(x, y, weight=sw.clip(lower=1e-6)), num_boost_round=200,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--train-windows", type=int, default=3)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--submit", type=Path,
                        help="au lieu du banc : écrit dans ce dossier la soumission FFORMA")
    parser.add_argument("--with-exog", action="store_true",
                        help="garder les exogènes du jeu (défaut : sans, comme les soumissions retenues)")
    args = parser.parse_args()
    if args.cache:
        cache_member_forecasts(args.cache / args.dataset)

    train, spec, keys, horizon, scored_days, exog_long = load(args.dataset)
    if not args.with_exog:
        spec = kc.CompetitionSpec(**{**DATASETS[args.dataset]["spec"], "exog_cols": []})
        exog_long = None
    end = train[spec.date_col].max()
    targets = [(f"-{k * scored_days}j", k * scored_days) for k in range(args.folds)] + [("an-1", 364)]

    cache: dict[int, tuple] = {}

    def window(offset):
        if offset not in cache:
            start = time.time()
            cache[offset] = window_data(train, spec, keys, horizon, scored_days, exog_long,
                                        end - pd.Timedelta(days=offset))
            print(f"  fenêtre -{offset} j prête ({time.time() - start:.0f}s)", flush=True)
        return cache[offset]

    if args.submit:
        past = [window(k * scored_days) for k in range(args.train_windows)]
        sse = pd.concat([series_sse(d) for d, _ in past]).groupby(level=0).sum()
        w_global = float(sse.sum().idxmin())
        meta = fit_fforma(past)
        # même enchaînement que le mode « soumission » de la page (_run_kaggle_job)
        entities = keys[spec.entity_cols].drop_duplicates()
        history = kc.pad_to(train, spec, train[spec.date_col].max())
        df = member_frame(keys[[*spec.entity_cols, "date"]], history, spec, horizon, exog_long, entities)
        feats = descriptors(history, spec, df)
        w_meta = pd.Series(np.clip(meta.predict(feats), 0, 1), index=feats.index)
        w = w_meta.reindex(df["_key"]).fillna(w_global).to_numpy()
        args.submit.mkdir(parents=True, exist_ok=True)
        out = args.submit / f"{args.dataset}-fforma.csv"
        out.write_text(kc.submission_csv(keys, pd.Series(blend(df, w)), spec))
        q = np.quantile(w_meta, [0.1, 0.5, 0.9])
        print(f"-> {out}  (global {w_global:.2f}, part LightGBM par série p10/p50/p90 "
              f"{q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f})")
        return

    results = {}
    for label, offset in targets:
        past = []
        for k in range(1, args.train_windows + 1):
            try:
                past.append(window(offset + k * scored_days))
            except ValueError:
                break
        if not past:
            print(f"{label} : aucune fenêtre d'apprentissage antérieure, ignorée")
            continue
        scores, w_global = evaluate(window(offset), past)
        results[label] = scores
        print(f"{label} ({len(past)} fenêtre(s) d'apprentissage, global {w_global:.2f}) : "
              + " ".join(f"{k}={v:.5f}" for k, v in scores.items()), flush=True)

    table = pd.DataFrame(results).T
    recent = table.drop(index="an-1", errors="ignore").mean()
    print("\nméthode     récent   an-1")
    for method in table.columns:
        yoy = f"{table.loc['an-1', method]:.5f}" if "an-1" in table.index else "-"
        print(f"{method:<10} {recent[method]:.5f}  {yoy}")


if __name__ == "__main__":
    main()
