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

import numpy as np
import pandas as pd

import api
import ensemble_weights as ew
import kaggle_competition as kc
from benchmark_kaggle import DATASETS, cache_member_forecasts, load

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
    return df, ew.descriptors(history, spec, df)


def member_frame(targets, history, spec, horizon, exog_long, entities) -> pd.DataFrame:
    frames = {}
    for member in ew.MEMBERS:
        rows = api._kaggle_forecast_rows({}, history, spec, horizon, member, {}, exog_long, entities)
        frames[member] = kc.predictions_frame(rows, spec)
    return ew.align(targets, frames, history, spec, ZERO_DEAD_DAYS)


def rmsle(df, weight) -> float:
    return kc.compute_metrics(df["actual"].to_numpy(), ew.blend(df, weight))["rmsle"]


def evaluate(target, history_windows):
    """Score de chaque méthode sur `target`, poids appris sur `history_windows`."""
    df, feats = target
    sse = pd.concat([ew.series_sse(d) for d, _ in history_windows]).groupby(level=0).sum()
    w_global = ew.global_weight(history_windows)
    w_series = sse.idxmin(axis=1)
    scores = {"fixe50": rmsle(df, 0.5), "global": rmsle(df, w_global)}
    for s in SHRINK:
        w = (s * w_series + (1 - s) * w_global).reindex(df["_key"]).fillna(w_global).to_numpy()
        scores[f"serie{int(s * 100)}"] = rmsle(df, w)
    scores["fforma"] = rmsle(df, ew.row_weights(history_windows, df, feats))
    scores["oracle"] = float(np.sqrt(ew.series_sse(df).min(axis=1).sum() / len(df)))
    return scores, w_global


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
        w_global = ew.global_weight(past)
        # même enchaînement que le mode « soumission » de la page (_run_kaggle_job)
        entities = keys[spec.entity_cols].drop_duplicates()
        history = kc.pad_to(train, spec, train[spec.date_col].max())
        df = member_frame(keys[[*spec.entity_cols, "date"]], history, spec, horizon, exog_long, entities)
        feats = ew.descriptors(history, spec, df)
        w_meta = ew.series_weights(ew.fit(past), feats)
        w = w_meta.reindex(df["_key"]).fillna(w_global).to_numpy()
        args.submit.mkdir(parents=True, exist_ok=True)
        out = args.submit / f"{args.dataset}-fforma.csv"
        out.write_text(kc.submission_csv(keys, pd.Series(ew.blend(df, w)), spec))
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
