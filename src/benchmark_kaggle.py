"""Banc d'essai des pistes de précision, par le même chemin que la page
Kaggle (api.kaggle_backtest) : un gain mesuré ici est un gain de la page,
pas d'une réimplémentation à part.

Plusieurs fenêtres de backtest successives (`--folds`, défaut 3) : une
seule fenêtre de 16 jours s'est révélée trop bruitée pour trancher entre
deux configurations proches (cf. BENCHMARKS.md). La fenêtre k recule la
fin du train de k × (jours notés) ; horizon et entités restent ceux du vrai
test.

    python benchmark_kaggle.py store-sales base dead14 --folds 3
    python benchmark_kaggle.py recruit base dead14

Variantes : cf. VARIANTS ci-dessous (combinables avec « + », ex.
`dead14+lgb_fine`).
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd

import api
import kaggle_competition as kc

RACINE = Path(__file__).resolve().parent.parent
DATASETS = {
    "store-sales": {
        "train": Path(os.path.expanduser("~/Téléchargements/store-sales-time-series-forecasting/train.csv")),
        "test": Path(os.path.expanduser("~/Téléchargements/store-sales-time-series-forecasting/test.csv")),
        "spec": dict(date_col="date", target_col="sales", entity_cols=["store_nbr", "family"], id_col="id",
                     exog_cols=["onpromotion"]),
    },
    "recruit": {
        "train": RACINE / "data" / "raw" / "air_visit_data.csv",
        "test": RACINE / "data" / "raw" / "sample_submission.csv",
        "spec": dict(date_col="visit_date", target_col="visitors", entity_cols=["air_store_id"], id_col="id",
                     composite_sep="_"),
    },
}

# Chaque variante : (modèle, model_params, options de kaggle_predict).
VARIANTS = {
    "base": ("lightgbm", {}, {}),
    "noexog": ("lightgbm", {}, {"_no_exog": True}),
    "dead14": ("lightgbm", {}, {"zero_dead_days": 14}),
    "dead28": ("lightgbm", {}, {"zero_dead_days": 28}),
    "lgb_fine": ("lightgbm", {"lightgbm": {"feature_config": {
        "lag_days": (1, 2, 3, 4, 5, 6, 7, 14, 21, 28), "rolling_windows": (7, 14, 28, 56)}}}, {}),
    "round4": ("lightgbm", {"lightgbm": {"feature_config": {"round_decimals": 4}}}, {}),
    "seeds3": ("lightgbm", {"lightgbm": {"feature_config": {"seeds": [1, 2, 3]}}}, {}),
    # réglages du script dédié Store Sales (0.42475 officiel), en générique
    "lgb_big": ("lightgbm", {"lightgbm": {"learning_rate": 0.03, "num_leaves": 255, "min_data_in_leaf": 10,
                                          "lambda_l2": 0.05, "num_boost_round": 1200}}, {}),
    "sarimax": ("sarimax", {}, {}),
    "ens_lgb_sarimax": ("lightgbm", {}, {"ensemble": [["lightgbm", 0.5], ["sarimax", 0.5]]}),
}


def combine(name: str) -> tuple[str, dict, dict]:
    model, params, options = "lightgbm", {}, {}
    for part in name.split("+"):
        m, p, o = VARIANTS[part]
        if m != "lightgbm":
            model = m
        for key, value in p.items():
            merged = params.setdefault(key, {})
            for k, v in value.items():
                if k == "feature_config":
                    merged.setdefault(k, {}).update(v)
                else:
                    merged[k] = v
        options.update(o)
    return model, params, options


def load(dataset: str):
    cfg = DATASETS[dataset]
    train_raw, test_raw = pd.read_csv(cfg["train"]), pd.read_csv(cfg["test"])
    spec = kc.CompetitionSpec(**cfg["spec"])
    train = kc.prepare_train(train_raw, spec)
    keys = kc.submission_keys(test_raw, spec)
    horizon, scored_days = kc.horizon_days(train, spec, keys)
    exog_long = kc.exog_long_frame(train, test_raw, keys, spec)
    return train, spec, keys, horizon, scored_days, exog_long


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("variants", nargs="+")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--no-yoy", action="store_true", help="sans la fenêtre « même période l'an dernier »")
    args = parser.parse_args()

    train, spec, keys, horizon, scored_days, exog_long = load(args.dataset)
    end = train[spec.date_col].max()
    print(f"{args.dataset} : horizon {horizon} j, {scored_days} j notés, {args.folds} fenêtre(s)")

    # Fenêtres : les `folds` plus récentes, plus la même période un an plus
    # tôt (364 j, jours de semaine alignés). Sur Store Sales, la fenêtre
    # récente seule a classé à l'envers deux configs que Kaggle a
    # départagées (0.49947 local / 0.53171 officiel contre un script à
    # 0.42475) : la saison du test (rentrée scolaire) n'y figure pas.
    windows = [(f"-{k * scored_days}j", end - pd.Timedelta(days=k * scored_days)) for k in range(args.folds)]
    if not args.no_yoy:
        windows.append(("an-1", end - pd.Timedelta(days=364)))

    for name in args.variants:
        model, params, options = combine(name)
        # option du banc seulement : ignorer les exogènes du jeu
        no_exog = options.pop("_no_exog", False)
        run_spec = kc.CompetitionSpec(**{**DATASETS[args.dataset]["spec"], "exog_cols": []}) if no_exog else spec
        run_exog = None if no_exog else exog_long
        scores = {}
        start = time.time()
        for label, fold_end in windows:
            job = {"processed": 0, "total": 0, "combos": []}
            out = api.kaggle_backtest(
                job, train[train[spec.date_col] <= fold_end], run_spec, keys, horizon, scored_days,
                run_exog, model, params, options,
            )
            scores[label] = out["scores"]["model"]["rmsle"]
        recent = [v for k, v in scores.items() if k != "an-1"]
        detail = " ".join(f"{k}={v:.5f}" for k, v in scores.items())
        yoy = f"  an-1 {scores['an-1']:.5f}" if "an-1" in scores else ""
        print(f"{name:<24} récent {sum(recent) / len(recent):.5f}{yoy}  [{detail}]  {time.time() - start:.0f}s",
              flush=True)

if __name__ == "__main__":
    main()
