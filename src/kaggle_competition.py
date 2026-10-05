"""Workflow « compétition Kaggle » (séries temporelles) : de train.csv +
test.csv (ou sample_submission.csv) au fichier de soumission, avec un
backtest local qui rejoue exactement la même configuration (même horizon,
même écart éventuel entre fin du train et début du test) sur les derniers
jours connus — pour trier les essais sans consommer de soumission Kaggle.

Logique pure (pandas), sans FastAPI : la prévision elle-même est déléguée
par api.py à la génération en masse existante (bulk_forecast_rows), ce
module ne fait que préparer ses entrées et exploiter ses sorties.
"""

import gzip
import io
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Plafond appliqué au contenu *décompressé* : sans lui, un petit .zip
# hautement compressé suffirait à saturer la mémoire du process.
MAX_DECOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024

# Repli pour une entité sans prévision (historique trop court, absente du
# train) : moyenne de ses derniers jours connus, 0 si elle n'a aucun passé.
FALLBACK_WINDOW_DAYS = 14

METRICS = ("rmsle", "rmse", "mae", "smape")


@dataclass
class CompetitionSpec:
    """Correspondance des colonnes d'une compétition.

    `entity_cols` : 1 ou 2 colonnes identifiant une série (ex. magasin, ou
    magasin × famille de produits). `composite_sep` : le test ne contient
    qu'un identifiant composite « <entité><sep><date> » (format *Recruit*,
    sample_submission.csv) au lieu de colonnes séparées.
    """

    date_col: str
    target_col: str
    entity_cols: list[str]
    id_col: str
    test_date_col: str | None = None
    composite_sep: str | None = None
    exog_cols: list[str] = field(default_factory=list)
    submission_target: str | None = None

    def __post_init__(self):
        if not 1 <= len(self.entity_cols) <= 2:
            raise ValueError("Une ou deux colonnes d'entité attendues")
        if self.composite_sep and len(self.entity_cols) != 1:
            raise ValueError("Un identifiant composite ne peut porter qu'une seule colonne d'entité")
        self.test_date_col = self.test_date_col or self.date_col
        self.submission_target = self.submission_target or self.target_col


def _read_capped(stream) -> bytes:
    content = stream.read(MAX_DECOMPRESSED_BYTES + 1)
    if len(content) > MAX_DECOMPRESSED_BYTES:
        raise ValueError("Fichier décompressé trop volumineux")
    return content


def decompress_upload(content: bytes, filename: str | None) -> bytes:
    """Renvoie le CSV brut d'un upload tel que téléchargé depuis Kaggle :
    .csv tel quel, .gz décompressé, .zip réduit à son unique CSV (ou à celui
    dont le nom correspond à l'archive, ex. train.csv.zip -> train.csv).
    Détection par signature plutôt que par extension seule : un fichier
    renommé reste lisible.
    """
    name = (filename or "").lower()
    if content[:2] == b"\x1f\x8b" or name.endswith(".gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
            return _read_capped(gz)
    if content[:4] == b"PK\x03\x04" or name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csvs = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
            if not csvs:
                raise ValueError("Aucun CSV dans l'archive")
            if len(csvs) > 1:
                stem = name.rsplit("/", 1)[-1].removesuffix(".zip")
                matching = [n for n in csvs if n.rsplit("/", 1)[-1].lower() == stem]
                if len(matching) != 1:
                    raise ValueError(f"Archive à plusieurs CSV ({', '.join(csvs[:5])}) : extraire le bon fichier")
                csvs = matching
            with zf.open(csvs[0]) as member:
                return _read_capped(member)
    return content


def suggest_mapping(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Devine une correspondance de colonnes à proposer par défaut — l'UI la
    laisse entièrement modifiable. Heuristiques volontairement simples :
    noms usuels des jeux Kaggle (date, id, store, family...) et types.
    """
    common = [c for c in train.columns if c in test.columns]

    def looks_like_date(df: pd.DataFrame, col: str) -> bool:
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty:
            return False
        return pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() > 0.9

    date_candidates = [c for c in common if looks_like_date(train, c)]
    date_col = next((c for c in date_candidates if "date" in c.lower()), date_candidates[0] if date_candidates else None)

    test_only = [c for c in test.columns if c not in train.columns]
    id_col = next((c for c in test.columns if c.lower() == "id"), test_only[0] if test_only else None)

    numeric_train_only = [
        c for c in train.columns
        if c not in test.columns and pd.api.types.is_numeric_dtype(train[c]) and c.lower() != "id"
    ]
    # Cible = colonne numérique absente du test. Sinon (sample_submission
    # au format *Recruit*), la colonne cible factice y figure sous le même
    # nom que dans train, avec une valeur constante.
    target_col = numeric_train_only[0] if numeric_train_only else next(
        (c for c in test.columns if c in train.columns and c != date_col and test[c].nunique(dropna=True) <= 1
         and pd.api.types.is_numeric_dtype(train[c])),
        None,
    )

    composite = date_col not in test.columns and id_col is not None
    entity_cols = _detect_entity_cols(train, [c for c in common if c not in (date_col, target_col, id_col)], date_col)
    exog_cols = [
        c for c in common
        if c not in (date_col, target_col, id_col, *entity_cols) and pd.api.types.is_numeric_dtype(train[c])
    ]

    composite_sep = None
    entity_for_composite = None
    if composite:
        sample_id = str(test[id_col].iloc[0]) if len(test) else ""
        for sep in ("_", "-", "|", ";"):
            head, _, tail = sample_id.rpartition(sep)
            if head and pd.notna(pd.to_datetime(tail, errors="coerce")):
                composite_sep = sep
                break
        # entité à rechercher parmi les colonnes catégorielles du train
        # (le test n'a alors aucune colonne en commun)
        if composite_sep:
            head = sample_id.rpartition(composite_sep)[0]
            entity_for_composite = next(
                (c for c in train.columns if c not in (date_col, target_col) and (train[c].astype(str) == head).any()),
                None,
            )
            if entity_for_composite:
                entity_cols = [entity_for_composite]

    if composite and date_col is None:
        train_dates = [c for c in train.columns if looks_like_date(train, c)]
        date_col = train_dates[0] if train_dates else None

    return {
        "date_col": date_col,
        "target_col": target_col,
        "id_col": id_col,
        "entity_cols": entity_cols,
        "exog_cols": [c for c in exog_cols if c not in entity_cols],
        "composite_sep": composite_sep,
    }


def _detect_entity_cols(train: pd.DataFrame, candidates: list[str], date_col: str | None) -> list[str]:
    """Plus petite combinaison (1 ou 2 colonnes) qui, avec la date, identifie
    chaque ligne : c'est la définition même d'une clé de série. Un critère de
    cardinalité seul confondait une exogène entière à peu de valeurs
    (onpromotion) avec un identifiant de magasin."""
    if not date_col or date_col not in train.columns:
        return []
    sample = train.head(500_000)
    low_card = [c for c in candidates if 1 < sample[c].nunique(dropna=True) <= 50_000]

    def is_key(cols: list[str]) -> bool:
        return not sample.duplicated(subset=[date_col, *cols]).any()

    singles = sorted((sample[c].nunique(), c) for c in low_card if is_key([c]))
    if singles:
        return [singles[0][1]]
    pairs = []
    for i, a in enumerate(low_card):
        for b in low_card[i + 1:]:
            if is_key([a, b]):
                pairs.append((sample[a].nunique() * sample[b].nunique(), [a, b]))
    if pairs:
        # magasin avant famille : la dimension la plus large en premier
        return sorted(min(pairs)[1], key=lambda c: -sample[c].nunique())
    return []


def _parse_dates(raw: pd.Series) -> pd.Series:
    iso = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
    if iso.notna().all():
        return iso
    return pd.to_datetime(raw, errors="coerce", format="mixed")


def prepare_train(train: pd.DataFrame, spec: CompetitionSpec) -> pd.DataFrame:
    """Colonnes utiles seulement, dates parsées, entités en texte (mêmes clés
    que le regroupement de la génération en masse)."""
    cols = [spec.date_col, spec.target_col, *spec.entity_cols, *spec.exog_cols]
    missing = [c for c in cols if c not in train.columns]
    if missing:
        raise ValueError(f"Colonnes absentes du train : {', '.join(missing)}")
    df = train[cols].copy()
    df[spec.date_col] = _parse_dates(df[spec.date_col])
    df = df.dropna(subset=[spec.date_col])
    if df.empty:
        raise ValueError(f"Colonne '{spec.date_col}' : aucune date reconnue")
    for col in spec.entity_cols:
        df[col] = df[col].astype(str)
    return df


def submission_keys(test: pd.DataFrame, spec: CompetitionSpec) -> pd.DataFrame:
    """Une ligne par prévision à rendre : id, entité(s), date."""
    if spec.id_col not in test.columns:
        raise ValueError(f"Colonne identifiant '{spec.id_col}' absente du test")

    if spec.composite_sep:
        parts = test[spec.id_col].astype(str).str.rpartition(spec.composite_sep)
        keys = pd.DataFrame({spec.id_col: test[spec.id_col], spec.entity_cols[0]: parts[0], "date": parts[2]})
    else:
        missing = [c for c in (spec.test_date_col, *spec.entity_cols) if c not in test.columns]
        if missing:
            raise ValueError(f"Colonnes absentes du test : {', '.join(missing)}")
        keys = test[[spec.id_col, *spec.entity_cols]].copy()
        keys["date"] = test[spec.test_date_col]
        for col in spec.entity_cols:
            keys[col] = keys[col].astype(str)

    keys["date"] = _parse_dates(keys["date"])
    if keys["date"].isna().any():
        raise ValueError("Dates illisibles dans le test (ou séparateur d'identifiant composite incorrect)")
    return keys.reset_index(drop=True)


def horizon_days(train: pd.DataFrame, spec: CompetitionSpec, keys: pd.DataFrame) -> tuple[int, int]:
    """(horizon total depuis la fin du train, nombre de jours notés). Le
    second peut être plus court que le premier si le test démarre après un
    écart (jours à prévoir mais jamais notés)."""
    train_end = train[spec.date_col].max()
    test_start, test_end = keys["date"].min(), keys["date"].max()
    if test_start <= train_end:
        raise ValueError(
            f"Le test commence le {test_start.date()} mais le train va jusqu'au {train_end.date()} : "
            "dates qui se chevauchent, mauvaise colonne date ?"
        )
    return (test_end - train_end).days, (test_end - test_start).days + 1


def exog_long_frame(
    train: pd.DataFrame, test: pd.DataFrame, keys: pd.DataFrame, spec: CompetitionSpec
) -> pd.DataFrame | None:
    """Exogènes connues à l'avance (ex. onpromotion) au format long (date,
    entité(s), valeurs) : historique du train + vraies valeurs futures du
    test, au lieu du report de la dernière valeur connue."""
    if not spec.exog_cols:
        return None
    missing = [c for c in spec.exog_cols if c not in test.columns]
    if missing:
        raise ValueError(f"Exogènes absentes du test (inconnues dans le futur) : {', '.join(missing)}")
    hist = train[[spec.date_col, *spec.entity_cols, *spec.exog_cols]].rename(columns={spec.date_col: "date"})
    future = keys[["date", *spec.entity_cols]].copy()
    for col in spec.exog_cols:
        future[col] = pd.to_numeric(test[col], errors="coerce").to_numpy()
    long = pd.concat([hist, future], ignore_index=True)
    for col in spec.exog_cols:
        long[col] = pd.to_numeric(long[col], errors="coerce")
    return long


def pad_to(train: pd.DataFrame, spec: CompetitionSpec, last_date: pd.Timestamp) -> pd.DataFrame:
    """Prolonge par des zéros les entités dont l'historique s'arrête avant
    `last_date` : la génération en masse prévoit à partir de la fin propre à
    chaque entité, qui ne couvrirait sinon pas les dates du test. Les jours
    manquants valent déjà 0 à l'intérieur d'une série (cf. _daily_series)."""
    last_by_entity = train.groupby(spec.entity_cols, observed=True)[spec.date_col].max()
    short = last_by_entity[last_by_entity < last_date]
    if short.empty:
        return train
    pad = short.reset_index()[spec.entity_cols].copy()
    pad[spec.date_col] = last_date
    pad[spec.target_col] = 0
    return pd.concat([train, pad], ignore_index=True)


def predictions_frame(rows: list[dict], spec: CompetitionSpec) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[*spec.entity_cols, "date", "prevision"])
    preds = pd.DataFrame(rows)[[*spec.entity_cols, "date", "prevision"]]
    preds["date"] = pd.to_datetime(preds["date"])
    for col in spec.entity_cols:
        preds[col] = preds[col].astype(str)
    return preds


def attach_predictions(
    targets: pd.DataFrame, preds: pd.DataFrame, history: pd.DataFrame, spec: CompetitionSpec, clip_zero: bool
) -> tuple[pd.Series, int]:
    """Prévision alignée sur chaque ligne de `targets` (entité(s) + date),
    repli sur la moyenne récente de l'entité quand le modèle n'en a pas
    produit. Renvoie (valeurs, nombre de lignes en repli)."""
    merged = targets[[*spec.entity_cols, "date"]].merge(preds, on=[*spec.entity_cols, "date"], how="left")

    missing = merged["prevision"].isna()
    n_fallback = int(missing.sum())
    if n_fallback:
        daily = history.groupby([*spec.entity_cols, spec.date_col], observed=True)[spec.target_col].sum().reset_index()
        daily = daily.sort_values(spec.date_col)
        recent = daily.groupby(spec.entity_cols, observed=True).tail(FALLBACK_WINDOW_DAYS)
        fallback = recent.groupby(spec.entity_cols, observed=True)[spec.target_col].mean().rename("_fallback")
        merged = merged.merge(fallback.reset_index(), on=spec.entity_cols, how="left")
        merged["prevision"] = merged["prevision"].fillna(merged["_fallback"]).fillna(0)

    values = merged["prevision"].astype(float)
    if clip_zero:
        values = values.clip(lower=0)
    return values.reset_index(drop=True), n_fallback


def backtest_split(
    train: pd.DataFrame, spec: CompetitionSpec, horizon: int, scored_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rejoue la configuration du vrai test sur les derniers jours connus :
    on coupe `horizon` jours avant la fin du train, et seuls les
    `scored_days` derniers sont notés (même écart non noté qu'au vrai test).
    Renvoie (historique d'entraînement, vérité terrain agrégée par jour)."""
    train_end = train[spec.date_col].max()
    cutoff = train_end - pd.Timedelta(days=horizon)
    fit = train[train[spec.date_col] <= cutoff]
    if fit.empty:
        raise ValueError("Historique trop court pour un backtest de cet horizon")
    window = train[train[spec.date_col] > train_end - pd.Timedelta(days=scored_days)]
    truth = (
        window.groupby([*spec.entity_cols, spec.date_col], observed=True)[spec.target_col].sum()
        .reset_index().rename(columns={spec.date_col: "date", spec.target_col: "actual"})
    )
    return fit, truth


def seasonal_naive(history: pd.DataFrame, targets: pd.DataFrame, spec: CompetitionSpec) -> pd.Series:
    """Référence « même jour de la semaine, dernière semaine connue » par
    entité — le score à battre, affiché à côté de celui du modèle."""
    daily = history.groupby([*spec.entity_cols, spec.date_col], observed=True)[spec.target_col].sum().reset_index()
    daily = daily.sort_values(spec.date_col)
    last_week = daily.groupby(spec.entity_cols, observed=True).tail(7).copy()
    last_week["dow"] = last_week[spec.date_col].dt.dayofweek
    last_week = last_week.drop_duplicates([*spec.entity_cols, "dow"], keep="last")
    t = targets[[*spec.entity_cols, "date"]].copy()
    t["dow"] = t["date"].dt.dayofweek
    merged = t.merge(last_week[[*spec.entity_cols, "dow", spec.target_col]], on=[*spec.entity_cols, "dow"], how="left")
    return merged[spec.target_col].fillna(0).astype(float).reset_index(drop=True)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = predicted - actual
    a_pos, p_pos = np.clip(actual, 0, None), np.clip(predicted, 0, None)
    denom = np.abs(actual) + np.abs(predicted)
    smape = np.where(denom == 0, 0.0, 2 * np.abs(err) / np.where(denom == 0, 1, denom))
    return {
        "rmsle": float(np.sqrt(np.mean((np.log1p(p_pos) - np.log1p(a_pos)) ** 2))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "smape": float(100 * np.mean(smape)),
    }


def submission_csv(keys: pd.DataFrame, values: pd.Series, spec: CompetitionSpec) -> str:
    out = pd.DataFrame({spec.id_col: keys[spec.id_col].to_numpy(), spec.submission_target: values.round(4).to_numpy()})
    return out.to_csv(index=False)
