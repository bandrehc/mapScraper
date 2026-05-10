"""main.py — Pipeline entry point.

Usage
-----
  python main.py --mode scrape "restaurants in Miami" --limit 50
  python main.py --mode scrape --queries-file query_example.txt
  python main.py --mode enrich --input data/output.csv
  python main.py --mode full   --queries-file query_example.txt --output-file data/leads.csv
  python main.py --mode cluster --input data/leads_enriched.csv
  python main.py --mode cluster --input data/leads_enriched.csv --cluster-method hdbscan

The original CLI (mapScraperX.py) is unchanged and continues to work as before.
"""
import argparse
import logging
import sys


def _read_queries(file_path: str) -> list[str]:
    try:
        with open(file_path, encoding='utf-8') as fh:
            return [ln.strip() for ln in fh if ln.strip() and not ln.startswith('#')]
    except FileNotFoundError:
        print(f"Error: queries file not found: {file_path}")
        sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='mapScraper pipeline — Google Maps scraping + enrichment + clustering.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes
-----
  scrape   Run the Google Maps scraper only (same output as mapScraperX.py).
  enrich   Load an existing CSV and add deep website signals + lead scores.
  full     Scrape, then enrich the result.
  cluster  Load any CSV (raw or enriched), embed + cluster, append cluster columns.

Examples
--------
  python main.py --mode scrape "marketing agencies in New York" --limit 50
  python main.py --mode scrape --queries-file query_example.txt
  python main.py --mode enrich --input data/output.csv
  python main.py --mode enrich --input data/output.csv --no-web-scraping
  python main.py --mode full --queries-file query_example.txt --output-file data/leads.csv
  python main.py --mode cluster --input data/leads_enriched.csv
  python main.py --mode cluster --input data/leads_enriched.csv --cluster-method hdbscan --cluster-k 10
  python main.py --mode cluster --input data/leads_enriched.csv --cluster-adjust-scores
        """,
    )

    parser.add_argument(
        '--mode',
        choices=['scrape', 'enrich', 'full', 'cluster'],
        default='scrape',
        help='Pipeline mode (default: scrape)',
    )

    # --- scraping -------------------------------------------------------------
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument('query', nargs='?', type=str,
                             help='Single search query (scrape / full modes)')
    query_group.add_argument('--queries-file', type=str, metavar='FILE',
                             help='Text file with one search query per line')
    parser.add_argument('--lang', type=str, default='en', metavar='CODE',
                        help='Language code (default: en)')
    parser.add_argument('--country', type=str, default='us', metavar='CODE',
                        help='Country code (default: us)')
    parser.add_argument('--limit', type=int, default=None, metavar='N',
                        help='Max results per query (default: no limit)')
    parser.add_argument('--output-file', type=str, default='data/output.csv', metavar='PATH',
                        help='Raw scrape output CSV (default: data/output.csv)')
    parser.add_argument('--concurrent', type=int, default=3, metavar='N',
                        help='Concurrent scraper tasks (default: 3)')

    # --- enrichment -----------------------------------------------------------
    parser.add_argument('--input', type=str, metavar='PATH',
                        help='Input CSV for enrich / cluster mode')
    parser.add_argument('--no-web-scraping', action='store_true',
                        help='Skip website crawling (enrich mode)')
    parser.add_argument('--web-concurrent', type=int, default=5, metavar='N',
                        help='Concurrent domain crawls (default: 5)')
    parser.add_argument('--web-batch-size', type=int, default=50, metavar='N',
                        help='Domains per async batch (default: 50)')
    parser.add_argument('--web-timeout', type=int, default=15, metavar='SEC',
                        help='Per-page HTTP timeout in seconds (default: 15)')
    parser.add_argument('--web-max-pages', type=int, default=10, metavar='N',
                        help='Max pages crawled per domain (default: 10)')

    # --- clustering -----------------------------------------------------------
    parser.add_argument('--cluster-method', choices=['kmeans', 'hdbscan'],
                        default='kmeans', metavar='METHOD',
                        help='Clustering algorithm: kmeans or hdbscan (default: kmeans)')
    parser.add_argument('--cluster-k', type=int, default=8, metavar='N',
                        help='Number of clusters for KMeans (default: 8)')
    parser.add_argument('--cluster-min-size', type=int, default=5, metavar='N',
                        help='min_cluster_size for HDBSCAN (default: 5)')
    parser.add_argument('--cluster-min-samples', type=int, default=3, metavar='N',
                        help='min_samples for HDBSCAN (default: 3)')
    parser.add_argument('--cluster-model', type=str, default='all-MiniLM-L6-v2',
                        metavar='NAME',
                        help='sentence-transformers model (default: all-MiniLM-L6-v2)')
    parser.add_argument('--cluster-batch-size', type=int, default=64, metavar='N',
                        help='Embedding batch size (default: 64)')
    parser.add_argument('--cluster-no-clients', action='store_true',
                        help='Exclude client names from the text representation')
    parser.add_argument('--cluster-no-cache', action='store_true',
                        help='Do not use cached embeddings (force re-encode)')
    parser.add_argument('--cluster-adjust-scores', action='store_true',
                        help='Apply cluster-based adjustments to the lead score column')
    parser.add_argument('--cluster-save-embeddings', type=str, default='',
                        metavar='PATH',
                        help='Save embedding matrix to a .npy file at this path')

    # --- logging --------------------------------------------------------------
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging verbosity (default: INFO)')

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)-8s %(name)s — %(message)s',
        datefmt='%H:%M:%S',
    )

    from pipeline.orchestrator import run_pipeline

    queries: list[str] = []
    if args.mode in ('scrape', 'full'):
        if args.query:
            queries = [args.query]
        elif args.queries_file:
            queries = _read_queries(args.queries_file)
        else:
            parser.error(f"--mode {args.mode} requires a query or --queries-file.")

    run_pipeline(
        mode=args.mode,
        # scraping
        queries=queries,
        lang=args.lang,
        country=args.country,
        limit=args.limit,
        max_concurrent=args.concurrent,
        output_file=args.output_file,
        # enrichment
        input_path=args.input,
        scrape_websites=not args.no_web_scraping,
        web_concurrent=args.web_concurrent,
        web_batch_size=args.web_batch_size,
        web_timeout=args.web_timeout,
        web_max_pages=args.web_max_pages,
        # clustering
        cluster_method=args.cluster_method,
        cluster_k=args.cluster_k,
        cluster_min_size=args.cluster_min_size,
        cluster_min_samples=args.cluster_min_samples,
        cluster_model=args.cluster_model,
        cluster_batch_size=args.cluster_batch_size,
        cluster_include_clients=not args.cluster_no_clients,
        cluster_no_cache=args.cluster_no_cache,
        cluster_adjust_scores=args.cluster_adjust_scores,
        cluster_save_embeddings=args.cluster_save_embeddings,
    )


if __name__ == '__main__':
    main()
