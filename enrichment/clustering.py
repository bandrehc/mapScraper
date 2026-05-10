"""Semantic clustering pipeline for enriched lead data.

Two methods available:
  kmeans  — fast, deterministic, requires k upfront; good baseline
  hdbscan — density-based, finds variable-size clusters and flags noise;
             requires `pip install hdbscan`

Each cluster is interpreted via TF-IDF over its member texts, then given a
human-readable heuristic label.

Bonus: cluster membership can be used to adjust the existing lead score —
see `cluster_score_adjustments()`.
"""
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    logger.info("KMeans — k=%d …", n_clusters)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(embeddings)

    if len(embeddings) > n_clusters + 1:
        sample = min(len(embeddings), 5_000)
        score = silhouette_score(embeddings, labels, sample_size=sample, random_state=random_state)
        logger.info("KMeans silhouette score: %.4f", score)

    return labels


def _hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> np.ndarray:
    try:
        import hdbscan as _hdbscan_lib
    except ImportError:
        raise ImportError(
            "hdbscan is required for HDBSCAN clustering.\n"
            "Install it with:  pip install hdbscan"
        )

    logger.info("HDBSCAN — min_cluster_size=%d, min_samples=%d …", min_cluster_size, min_samples)
    clusterer = _hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
    )
    labels = clusterer.fit_predict(embeddings)

    n = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100
    logger.info("HDBSCAN — %d clusters found, %.1f%% noise", n, noise_pct)
    return labels


# ---------------------------------------------------------------------------
# Cluster interpretation
# ---------------------------------------------------------------------------

def _interpret(
    labels: np.ndarray,
    texts: List[str],
    company_names: List[str],
    top_n_keywords: int,
    n_representative: int,
) -> Dict[int, Dict]:
    """Return per-cluster summary: keywords, representative names, size."""
    # Group texts and indices by label
    buckets: Dict[int, List[int]] = {}
    for i, lbl in enumerate(labels):
        buckets.setdefault(int(lbl), []).append(i)

    # Build one "document" per cluster for TF-IDF
    docs = {lbl: ' '.join(texts[i] for i in idxs) for lbl, idxs in buckets.items()}

    tfidf = TfidfVectorizer(
        max_features=3_000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
    )
    try:
        matrix = tfidf.fit_transform(list(docs.values()))
        features = tfidf.get_feature_names_out()
    except ValueError:
        return {}

    summary: Dict[int, Dict] = {}
    for col_idx, lbl in enumerate(sorted(docs.keys())):
        row = matrix[col_idx].toarray().flatten()
        top_idx = row.argsort()[::-1][:top_n_keywords]
        keywords = [features[j] for j in top_idx if row[j] > 0]

        idxs = buckets[lbl]
        reps = [company_names[i] for i in idxs[:n_representative]]

        summary[lbl] = {
            'keywords': keywords,
            'representative_companies': reps,
            'size': len(idxs),
        }

    return summary


# ---------------------------------------------------------------------------
# Heuristic labelling
# ---------------------------------------------------------------------------

_LABEL_RULES: List[Tuple[List[str], str]] = [
    (['seo', 'organic', 'ranking', 'backlink', 'link building'],    'SEO Agencies'),
    (['ppc', 'google ads', 'paid search', 'paid media', 'adwords'], 'Paid Media / PPC'),
    (['social media', 'instagram', 'tiktok', 'facebook ads'],       'Social Media Marketing'),
    (['branding', 'brand identity', 'logo', 'visual identity'],     'Branding & Design'),
    (['web development', 'software', 'app development', 'frontend'], 'Web & App Development'),
    (['ecommerce', 'shopify', 'woocommerce', 'conversion rate'],    'E-commerce Specialists'),
    (['email marketing', 'automation', 'crm', 'hubspot', 'klaviyo'], 'Email & Marketing Automation'),
    (['content marketing', 'copywriting', 'content strategy'],      'Content Marketing'),
    (['pr', 'public relations', 'communications', 'reputation'],    'PR & Communications'),
    (['video', 'animation', 'motion graphics', 'production'],       'Video & Creative Production'),
    (['healthcare', 'medical', 'pharma', 'health'],                 'Healthcare Marketing'),
    (['fintech', 'finance', 'banking', 'investment'],               'Finance / Fintech Marketing'),
    (['saas', 'b2b', 'enterprise', 'software'],                     'B2B / SaaS Marketing'),
    (['strategy', 'consulting', 'growth strategy', 'go-to-market'], 'Strategy & Consulting'),
    (['influencer', 'creator', 'ugc', 'ambassador'],                'Influencer Marketing'),
    (['analytics', 'data', 'business intelligence', 'reporting'],   'Data & Analytics'),
    (['performance', 'roi', 'results', 'growth', 'revenue'],        'Performance Marketing'),
]


def _label(keywords: List[str]) -> str:
    kw_str = ' '.join(keywords[:8]).lower()
    for patterns, label in _LABEL_RULES:
        if any(p in kw_str for p in patterns):
            return label
    return (keywords[0].title() + ' Specialists') if keywords else 'General Agency'


# ---------------------------------------------------------------------------
# Score adjustment (bonus)
# ---------------------------------------------------------------------------

def cluster_score_adjustments(
    df: pd.DataFrame,
    summary: Dict[int, Dict],
) -> pd.Series:
    """Return a score delta Series based on cluster signals.

    Logic:
      - Small, niche clusters (size < 10% of total) → +5 pts (specialisation premium)
      - Large generic clusters (size > 30% of total) → -3 pts
      - Noise cluster (-1 from HDBSCAN) → -5 pts
    Capped to keep the final score in [0, 100].
    """
    total = len(df)
    cluster_sizes = {cid: info['size'] for cid, info in summary.items()}

    def delta(cluster_id: int) -> float:
        if cluster_id == -1:
            return -5.0
        size = cluster_sizes.get(cluster_id, 0)
        ratio = size / total if total else 0
        if ratio < 0.10:
            return +5.0
        if ratio > 0.30:
            return -3.0
        return 0.0

    return df['cluster_id'].apply(delta)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

ClusterResult = Tuple[pd.DataFrame, Dict[int, Dict]]


def run_clustering(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    texts: List[str],
    *,
    method: str = 'kmeans',
    n_clusters: int = 8,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    random_state: int = 42,
    top_n_keywords: int = 8,
    n_representative: int = 3,
    adjust_scores: bool = False,
) -> ClusterResult:
    """Run clustering and return (augmented_df, cluster_summary).

    New columns added to df:
      cluster_id     int   (-1 = HDBSCAN noise)
      cluster_label  str   (heuristic label)

    cluster_summary maps cluster_id → {keywords, representative_companies, size}.
    """
    method = method.lower()

    if method == 'kmeans':
        labels = _kmeans(embeddings, n_clusters, random_state)
    elif method == 'hdbscan':
        labels = _hdbscan(embeddings, min_cluster_size, min_samples)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: kmeans, hdbscan")

    names = df.get('title', pd.Series(range(len(df)))).fillna('Unknown').tolist()
    summary = _interpret(labels, texts, names, top_n_keywords, n_representative)

    label_map = {cid: _label(info['keywords']) for cid, info in summary.items()}
    label_map[-1] = 'Noise / Unclustered'

    df = df.copy()
    df['cluster_id'] = labels
    df['cluster_label'] = [label_map.get(int(lbl), f'Cluster {lbl}') for lbl in labels]

    if adjust_scores and 'score' in df.columns:
        delta = cluster_score_adjustments(df, summary)
        df['score'] = (df['score'].astype(float) + delta).clip(0, 100).round(1)
        logger.info("Cluster-based score adjustments applied.")

    return df, summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_summary(summary: Dict[int, Dict], method: str, n_total: int) -> str:
    """Return a human-readable cluster summary string."""
    lines = [
        f"=== Cluster Summary ({method.upper()}, {n_total} companies) ===",
        "",
    ]
    noise_info = summary.get(-1)

    for cid in sorted(k for k in summary if k != -1):
        info = summary[cid]
        label = _label(info['keywords'])
        kw = ', '.join(info['keywords'][:6])
        reps = ', '.join(info['representative_companies'])
        pct = info['size'] / n_total * 100
        lines += [
            f"Cluster {cid:>2} — {label}  [{info['size']} companies, {pct:.1f}%]",
            f"  Keywords:  {kw}",
            f"  Examples:  {reps}",
            "",
        ]

    if noise_info:
        pct = noise_info['size'] / n_total * 100
        lines += [
            f"Noise / Unclustered — [{noise_info['size']} companies, {pct:.1f}%]",
            "",
        ]

    return '\n'.join(lines)


def save_summary_json(summary: Dict[int, Dict], path: str) -> None:
    serialisable = {str(k): v for k, v in summary.items()}
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(serialisable, fh, indent=2, ensure_ascii=False)
    logger.info("Cluster summary saved to %s", path)
