"""Central pipeline orchestrator.

Exposes a single public function: run_pipeline(mode, ...)

Modes
-----
scrape   Run the Google Maps scraper only. Outputs a raw CSV.
enrich   Load an existing raw CSV, run enrichment, save an enriched CSV.
full     Scrape first, then enrich the result.
cluster  Load any CSV (raw or enriched), embed + cluster, append cluster columns.

The scraper and the enrichment step are never executed simultaneously —
the orchestrator runs them sequentially within a single process.
"""
import logging
import os
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must be present in any CSV handed to the enrichment step
_REQUIRED_COLUMNS = [
    'id', 'url_place', 'title', 'category', 'address',
    'phoneNumber', 'completePhoneNumber', 'domain', 'url',
    'coor', 'stars', 'reviews', 'source_query',
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _enriched_path(raw_path: str) -> str:
    base, ext = os.path.splitext(raw_path)
    return f"{base}_enriched{ext or '.csv'}"


def _clustered_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_clustered{ext or '.csv'}"


def _summary_path(input_path: str) -> str:
    base, _ = os.path.splitext(input_path)
    return f"{base}_cluster_summary.json"


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def _run_scraper(
    queries: List[str],
    lang: str,
    country: str,
    limit: Optional[int],
    output_file: str,
    max_concurrent: int,
) -> None:
    import mapScraper.placesCrawlerV2 as crawler

    logger.info("Scraping %d queries → %s", len(queries), output_file)
    _ensure_dir(output_file)

    results = crawler.search_multiple(queries, lang, country, limit, max_concurrent)
    if not results:
        logger.warning("Scraper returned 0 results.")
    crawler.save_to_csv(results, output_file)
    logger.info("Scrape done: %d places saved to %s", len(results), output_file)


def _run_enrichment(
    df: pd.DataFrame,
    *,
    scrape_websites: bool,
    web_concurrent: int,
    web_batch_size: int,
    web_timeout: int,
    web_max_pages: int,
) -> pd.DataFrame:
    from enrichment.features import engineer_features
    from enrichment.scoring import compute_scores

    logger.info("Engineering features …")
    df = engineer_features(df)

    if scrape_websites:
        from enrichment.web_scraper import enrich_websites
        urls = df['url'].fillna('').tolist()
        non_empty = sum(1 for u in urls if u.strip())
        logger.info("Deep scraping %d/%d websites (max %d pages each) …",
                    non_empty, len(urls), web_max_pages)
        web_rows = enrich_websites(
            urls,
            max_concurrent=web_concurrent,
            batch_size=web_batch_size,
            timeout=web_timeout,
            max_pages=web_max_pages,
        )
        web_df = pd.DataFrame(web_rows, index=df.index)
        df = pd.concat([df, web_df], axis=1)
    else:
        logger.info("Website scraping skipped (--no-web-scraping).")

    logger.info("Computing lead scores …")
    df = compute_scores(df)

    return df


# ---------------------------------------------------------------------------
# Clustering stage
# ---------------------------------------------------------------------------

def _run_clustering_stage(
    df: pd.DataFrame,
    *,
    method: str,
    n_clusters: int,
    min_cluster_size: int,
    min_samples: int,
    model_name: str,
    batch_size: int,
    include_clients: bool,
    use_cache: bool,
    adjust_scores: bool,
    embeddings_file: str,
    summary_out: str,
) -> pd.DataFrame:
    from enrichment.text_builder import build_texts
    from enrichment.embeddings import load_or_generate
    from enrichment.clustering import run_clustering, format_summary, save_summary_json

    logger.info("Building text representations …")
    texts = build_texts(df, include_clients=include_clients)

    logger.info("Generating / loading embeddings …")
    embeddings = load_or_generate(
        texts,
        cache_file=embeddings_file or None,
        model_name=model_name,
        batch_size=batch_size,
    )

    logger.info("Clustering (%s) …", method)
    df, summary = run_clustering(
        df,
        embeddings,
        texts,
        method=method,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        adjust_scores=adjust_scores,
    )

    print(format_summary(summary, method, len(df)))

    save_summary_json(summary, summary_out)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    mode: str,
    *,
    # --- scraping ---
    queries: Optional[List[str]] = None,
    lang: str = 'en',
    country: str = 'us',
    limit: Optional[int] = None,
    max_concurrent: int = 3,
    output_file: str = 'data/output.csv',
    # --- enrichment ---
    input_path: Optional[str] = None,
    scrape_websites: bool = True,
    web_concurrent: int = 5,
    web_batch_size: int = 50,
    web_timeout: int = 15,
    web_max_pages: int = 10,
    # --- clustering ---
    cluster_method: str = 'kmeans',
    cluster_k: int = 8,
    cluster_min_size: int = 5,
    cluster_min_samples: int = 3,
    cluster_model: str = 'all-MiniLM-L6-v2',
    cluster_batch_size: int = 64,
    cluster_include_clients: bool = True,
    cluster_no_cache: bool = False,
    cluster_adjust_scores: bool = False,
    cluster_save_embeddings: str = '',
) -> None:
    """Execute the pipeline in the requested mode.

    Modes: scrape | enrich | full | cluster
    """
    mode = mode.lower().strip()

    if mode == 'scrape':
        if not queries:
            raise ValueError("'scrape' mode requires at least one query.")
        _run_scraper(queries, lang, country, limit, output_file, max_concurrent)

    elif mode == 'enrich':
        if not input_path:
            raise ValueError("'enrich' mode requires --input <path-to-csv>.")
        df = _load_csv(input_path)
        df = _run_enrichment(
            df,
            scrape_websites=scrape_websites,
            web_concurrent=web_concurrent,
            web_batch_size=web_batch_size,
            web_timeout=web_timeout,
            web_max_pages=web_max_pages,
        )
        out = _enriched_path(input_path)
        _ensure_dir(out)
        df.to_csv(out, index=False)
        logger.info("Enriched CSV saved to %s", out)
        print(f"Enriched CSV saved to: {out}")

    elif mode == 'full':
        if not queries:
            raise ValueError("'full' mode requires at least one query.")
        _run_scraper(queries, lang, country, limit, output_file, max_concurrent)
        df = _load_csv(output_file)
        df = _run_enrichment(
            df,
            scrape_websites=scrape_websites,
            web_concurrent=web_concurrent,
            web_batch_size=web_batch_size,
            web_timeout=web_timeout,
            web_max_pages=web_max_pages,
        )
        out = _enriched_path(output_file)
        _ensure_dir(out)
        df.to_csv(out, index=False)
        logger.info("Enriched CSV saved to %s", out)
        print(f"Raw CSV:      {output_file}")
        print(f"Enriched CSV: {out}")

    elif mode == 'cluster':
        if not input_path:
            raise ValueError("'cluster' mode requires --input <path-to-csv>.")
        df = pd.read_csv(input_path, dtype=str)
        logger.info("Loaded %d rows from %s", len(df), input_path)

        out = _clustered_path(input_path)
        summary_out = _summary_path(input_path)

        df = _run_clustering_stage(
            df,
            method=cluster_method,
            n_clusters=cluster_k,
            min_cluster_size=cluster_min_size,
            min_samples=cluster_min_samples,
            model_name=cluster_model,
            batch_size=cluster_batch_size,
            include_clients=cluster_include_clients,
            use_cache=not cluster_no_cache,
            adjust_scores=cluster_adjust_scores,
            embeddings_file=cluster_save_embeddings,
            summary_out=summary_out,
        )

        _ensure_dir(out)
        df.to_csv(out, index=False)
        logger.info("Clustered CSV saved to %s", out)
        print(f"Clustered CSV:   {out}")
        print(f"Cluster summary: {summary_out}")

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Valid modes: scrape, enrich, full, cluster."
        )
