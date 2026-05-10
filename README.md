![Header](https://github.com/christivn/mapScraper/blob/main/github-header-image.png?raw=true)

# Google Maps Scraper + Enrichment + Clustering Pipeline

A Python tool for scraping Google Maps local services data, enriching leads with deep website intelligence, and grouping them into semantic clusters for market segmentation.

Two entry points, fully independent:

| Script | Purpose |
|--------|---------|
| `mapScraperX.py` | Original scraper CLI — unchanged, backward-compatible |
| `main.py` | Pipeline CLI with `scrape / enrich / full / cluster` modes |

---

## Features

### Scraping (original)
- Place ID, Maps URL, business name, category, full address
- Phone (local + international format)
- Website domain + URL
- GPS coordinates
- Average star rating + review count
- Concurrent async processing, configurable language / country

### Deep Enrichment (new)
- **Multi-page crawling** — up to 10 pages per domain, prioritised by content type
- **Business description** extraction (short from meta + long from body)
- **Services taxonomy** — 18 categories (SEO, PPC, Branding, Web Dev, …)
- **Industry taxonomy** — 16 verticals (Healthcare, Fintech, SaaS, …)
- **Client intelligence** — logo alt-text extraction, proper noun NER, testimonial detection
- **Company signals** — team size, years of experience, certifications, office locations
- **Technical signals** — CMS detection (10 platforms), tracking tools (10 tools), modern framework detection
- **Lead scoring** — interpretable 0–100 score, segment label, client quality score
- **Feature engineering** — phone / website presence, domain validation, rating score, review density

### Semantic Clustering (new)
- **Text representation** — combines description, services, industries, keywords, testimonials per lead
- **Embeddings** — sentence-transformers (`all-MiniLM-L6-v2` by default, any model supported)
- **Disk cache** — embeddings cached by content hash, zero re-computation cost on reruns
- **KMeans** — fast, deterministic baseline (configurable k)
- **HDBSCAN** — density-based, finds variable-size clusters and flags outliers
- **Cluster interpretation** — TF-IDF keywords + representative companies per cluster
- **Heuristic labels** — automatic human-readable labels (SEO Agencies, Branding & Design, …)
- **Score adjustment** — optional cluster-based delta applied to existing lead scores
- **JSON summary** — machine-readable cluster report saved alongside output CSV

---

## Prerequisites

- Python 3.10+
- pip

## Installation

```bash
git clone https://github.com/christivn/mapScraper.git
cd mapScraper
pip install -r requirements.txt
```

Dependencies: `aiohttp`, `tqdm`, `pandas`, `beautifulsoup4`, `scikit-learn`, `sentence-transformers`, `hdbscan`

---

## Usage — original scraper (mapScraperX.py)

Everything here works exactly as before.

```bash
# Single query
python mapScraperX.py "restaurants in Miami" --limit 50

# Multiple queries from file
python mapScraperX.py --queries-file query_example.txt

# With concurrency and custom output
python mapScraperX.py --queries-file query_example.txt \
  --lang en --country us --limit 25 \
  --output-file data/custom.csv --concurrent 5
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `query` | — | Single search query |
| `--queries-file FILE` | — | File with one query per line |
| `--lang CODE` | `en` | Language code |
| `--country CODE` | `us` | Country code |
| `--limit N` | no limit | Max results (total / per query) |
| `--output-file PATH` | `data/output.csv` | Output CSV path |
| `--concurrent N` | `3` | Max concurrent queries (3–5 recommended) |

---

## Usage — pipeline (main.py)

### Modes

| Mode | What it does |
|------|-------------|
| `scrape` | Google Maps scraping only — identical output to `mapScraperX.py` |
| `enrich` | Load an existing CSV → deep enrichment → save enriched CSV |
| `full` | Scrape first, then enrich the result |
| `cluster` | Load any CSV → embed → cluster → append `cluster_id` + `cluster_label` |

### Examples

```bash
# Scrape (same as mapScraperX.py)
python main.py --mode scrape "marketing agencies in New York" --limit 50
python main.py --mode scrape --queries-file query_example.txt

# Enrich a previously generated CSV (full deep crawl)
python main.py --mode enrich --input data/output.csv

# Enrich without fetching websites (feature engineering + scoring only)
python main.py --mode enrich --input data/output.csv --no-web-scraping

# Full pipeline in one command
python main.py --mode full \
  --queries-file query_example.txt \
  --output-file data/leads.csv

# Cluster an enriched CSV (KMeans, k=8)
python main.py --mode cluster --input data/leads_enriched.csv

# Cluster with HDBSCAN (automatic k)
python main.py --mode cluster --input data/leads_enriched.csv \
  --cluster-method hdbscan --cluster-min-size 8

# Cluster with score adjustment and save embeddings
python main.py --mode cluster --input data/leads_enriched.csv \
  --cluster-k 10 --cluster-adjust-scores \
  --cluster-save-embeddings data/embeddings.npy

# Verbose debug output
python main.py --mode cluster --input data/leads_enriched.csv --log-level DEBUG
```

### All options

```
--mode {scrape,enrich,full,cluster}  Pipeline mode (default: scrape)
query                                 Single search query
--queries-file FILE                   File with one query per line
--lang CODE                           Language code (default: en)
--country CODE                        Country code (default: us)
--limit N                             Max results per query
--output-file PATH                    Raw scrape output (default: data/output.csv)
--concurrent N                        Concurrent scraper tasks (default: 3)
--input PATH                          Input CSV for enrich / cluster mode
--no-web-scraping                     Skip website crawling (enrich mode)
--web-concurrent N                    Concurrent domain crawls (default: 5)
--web-batch-size N                    Domains per async batch (default: 50)
--web-timeout SEC                     Per-page HTTP timeout (default: 15)
--web-max-pages N                     Max pages crawled per domain (default: 10)
--cluster-method {kmeans,hdbscan}     Clustering algorithm (default: kmeans)
--cluster-k N                         KMeans number of clusters (default: 8)
--cluster-min-size N                  HDBSCAN min_cluster_size (default: 5)
--cluster-min-samples N               HDBSCAN min_samples (default: 3)
--cluster-model NAME                  sentence-transformers model (default: all-MiniLM-L6-v2)
--cluster-batch-size N                Embedding batch size (default: 64)
--cluster-no-clients                  Exclude client names from text representation
--cluster-no-cache                    Force re-encode (ignore cached embeddings)
--cluster-adjust-scores               Apply cluster-based delta to lead score column
--cluster-save-embeddings PATH        Save embedding matrix to .npy file
--log-level {DEBUG,INFO,...}          Logging verbosity (default: INFO)
```

---

## Output format

### Raw scrape CSV (unchanged schema)

| Column | Description | Example |
|--------|-------------|---------|
| `id` | Google Place ID | `ChIJN1t_tDeuEmsRUsoyG83frY4` |
| `url_place` | Google Maps link | `https://www.google.com/maps/place/?q=place_id:...` |
| `title` | Business name | `Joe's Pizza` |
| `category` | Business category | `Pizza restaurant` |
| `address` | Full address | `123 Main St, New York, NY 10001` |
| `phoneNumber` | Local phone | `(555) 123-4567` |
| `completePhoneNumber` | International phone | `+1 555-123-4567` |
| `domain` | Website domain | `joespizza.com` |
| `url` | Full website URL | `https://www.joespizza.com` |
| `coor` | Coordinates | `40.7128,-74.0060` |
| `stars` | Average rating | `4.5` |
| `reviews` | Review count | `234` |
| `source_query` | Original query | `pizza in New York` |

### Enriched CSV (all original columns plus the following)

#### Feature engineering

| Column | Type | Description |
|--------|------|-------------|
| `has_phone` | bool | Phone number present |
| `has_website` | bool | Website domain or URL present |
| `domain_valid` | bool | Domain passes basic format validation |
| `rating_score` | float | `stars × log(reviews+1)` — penalises high ratings with few reviews |
| `review_density` | float | Normalised review count within the batch (0–1) |

#### Website description

| Column | Type | Description |
|--------|------|-------------|
| `description_short` | str | Meta description or first 2 sentences (≤300 chars) |
| `description_long` | str | First substantial paragraph from homepage (≤800 chars) |

#### Taxonomy (JSON lists)

| Column | Type | Description |
|--------|------|-------------|
| `services_list` | JSON | Matched service categories (e.g. `["SEO","PPC / Paid Ads","Branding"]`) |
| `industries_list` | JSON | Matched industry verticals (e.g. `["Finance / Fintech","Technology / SaaS"]`) |

**Service categories detected:** SEO, PPC / Paid Ads, Social Media Marketing, Content Marketing, Email Marketing, Web Development, App Development, Branding, UI/UX Design, E-commerce, Analytics & Data, Strategy & Consulting, PR & Communications, Video Production, Photography, Influencer Marketing, Marketing Automation, Affiliate Marketing

**Industry verticals detected:** Healthcare, Finance / Fintech, Retail / E-commerce, Technology / SaaS, Education / EdTech, Real Estate, Legal, Manufacturing, Hospitality / Travel, Non-profit / NGO, B2B, Automotive, Energy / Sustainability, Sports & Fitness, Media & Entertainment, Government / Public Sector

#### Client intelligence (JSON lists)

| Column | Type | Description |
|--------|------|-------------|
| `clients_list` | JSON | Extracted client/brand names (logo alt-text + proper noun NER) |
| `testimonials` | JSON | Extracted testimonial quotes (blockquotes + attribution patterns) |
| `client_count` | int | Number of unique client names found |
| `client_quality_score` | float 0–100 | Client signal quality (see breakdown below) |

#### Company signals

| Column | Type | Description |
|--------|------|-------------|
| `team_size_indicator` | str | e.g. `"Team of 45+"` or `"200+ employees"` |
| `years_experience` | str | e.g. `"12 years of experience"` or `"since 2008"` |
| `certifications` | JSON | e.g. `["Premier Google Partner","ISO 9001"]` |
| `locations` | JSON | e.g. `["New York","London"]` |

#### Technical signals

| Column | Type | Description |
|--------|------|-------------|
| `cms_detected` | str | Detected CMS platform (WordPress, Webflow, Shopify, Wix, Squarespace, Ghost, Drupal, Joomla, HubSpot CMS, Framer) |
| `tracking_tools` | JSON | Detected analytics / tracking tools |

**Tracking tools detected:** Google Analytics, Google Tag Manager, Meta Pixel, HotJar, Intercom, Mixpanel, HubSpot Tracking, Microsoft Clarity, LinkedIn Insight, Amplitude

#### Keywords & value propositions

| Column | Type | Description |
|--------|------|-------------|
| `keywords_detected` | JSON | Top 15 content keywords from all crawled pages |
| `value_prop_signals` | JSON | Value-proposition words detected (ROI, growth, conversion, …) |

#### Scores & metadata

| Column | Type | Description |
|--------|------|-------------|
| `site_complexity_score` | float 0–10 | Page depth × service breadth × content richness |
| `digital_maturity_score` | float 0–10 | CMS + tracking stack + modern framework + client social proof |
| `content_length` | int | Total character count across all crawled pages |
| `pages_crawled` | int | Number of pages successfully fetched |
| `score` | float 0–100 | Lead score (see breakdown below) |
| `segment` | str | `micro / small / medium / large` |

---

## Scoring breakdowns

### Lead score (0–100)

| Signal | Max pts | Thresholds |
|--------|---------|------------|
| Review count | 30 | ≥500→30, ≥200→24, ≥100→18, ≥50→12, ≥10→7, ≥1→3 |
| Star rating | 25 | ≥4.5→25, ≥4.0→20, ≥3.5→15, ≥3.0→10, >0→5 |
| Website presence | 30 | has_website+10, domain_valid+5, web_has_contact+5, web_has_services+5, web_is_modern+5 |
| Phone | 15 | has_phone→15 |

| Segment | Score |
|---------|-------|
| micro | 0–24 |
| small | 25–49 |
| medium | 50–74 |
| large | 75–100 |

### Client quality score (0–100)

| Signal | Max pts |
|--------|---------|
| Number of clients (×4 each) | 40 |
| Number of testimonials (×5 each) | 20 |
| Has case study / portfolio pages | 20 |
| Multi-word client names (precision signal) | 20 |

### Site complexity score (0–10)
Pages crawled, service category count, client section presence, tracking tools, content volume.

### Digital maturity score (0–10)
CMS presence, tracking stack depth, modern framework, contact section, client social proof, certifications.

---

## Architecture

```
mapScraper/
├── mapScraperX.py              original scraper CLI (unchanged)
├── main.py                     pipeline CLI
├── requirements.txt
├── mapScraper/
│   └── placesCrawlerV2.py      async Google Maps scraper (deduplicates by id)
├── enrichment/
│   ├── features.py             feature engineering from CSV columns
│   ├── web_scraper.py          deep async website crawl + signal extraction
│   └── scoring.py              lead score (0–100) + segmentation
├── pipeline/
│   └── orchestrator.py         run_pipeline() — wires all stages
└── data/
    └── output.csv              scrape output (example)
```

### web_scraper.py internals

```
enrich_websites(urls)
  └─ _run_batch(urls)  [asyncio.gather, semaphore-limited]
       └─ _enrich_one(url)
            └─ _crawl_domain(url, max_pages)  [sequential per domain]
                 ├─ _fetch(page_url)
                 ├─ _clean(html)  → text, headings, meta_desc
                 ├─ _extract_links(html) → _prioritise(links) → queue
                 └─ (repeat up to max_pages)
            └─ _build_signals(pages)
                 ├─ _short_description / _long_description
                 ├─ _match_taxonomy (services, industries)
                 ├─ _extract_clients_and_testimonials
                 │    ├─ _find_client_sections (heading + class/id heuristic)
                 │    ├─ img[alt] extraction (logo names — highest precision)
                 │    ├─ blockquote / testimonial pattern detection
                 │    └─ _extract_proper_nouns (NER fallback)
                 ├─ _detect_cms / _detect_tracking / _detect_modern
                 ├─ _detect_years / _detect_team_size / _detect_certifications
                 ├─ _detect_locations
                 └─ _site_complexity / _digital_maturity / _client_quality
```

### Design principles

- Scraper and enrichment are **fully independent** — no cross-imports.
- Scraper output schema is **frozen** — the 13-column CSV never changes.
- Enriched output is a **superset** — all original columns preserved.
- Website crawling is **fault-tolerant** — any failure returns empty signals silently.
- Crawling is **domain-sequential** — pages within a domain are fetched one at a time to avoid hammering target sites.
- Client names from **logo alt-text** are the highest-precision signal; proper noun extraction is the fallback.
- All list fields stored as **JSON strings** in the CSV — directly parseable with `json.loads()`.
- Duplicates are removed by `id` at scrape-save time — first occurrence wins.

---

## Performance guidance

| Scenario | Recommended settings |
|----------|---------------------|
| Quick signal pass (no web scraping) | `--no-web-scraping` |
| Fast web pass (homepage only) | `--web-max-pages 1 --web-concurrent 15` |
| Standard deep crawl | `--web-max-pages 10 --web-concurrent 5` (default) |
| Maximum depth | `--web-max-pages 15 --web-concurrent 3 --web-timeout 20` |

For large CSVs (50k+ rows), web scraping is the bottleneck. Use `--no-web-scraping` first to get feature engineering and scoring, then run a second enrichment pass on high-scoring leads only.

---

## Supported languages and countries

| Code | Language | Code | Country |
|------|----------|------|---------|
| `en` | English | `us` | United States |
| `es` | Spanish | `es` | Spain |
| `fr` | French | `fr` | France |
| `de` | German | `de` | Germany |
| `it` | Italian | `it` | Italy |
| `pt` | Portuguese | `br` | Brazil |
| `ja` | Japanese | `jp` | Japan |
| `ko` | Korean | `kr` | South Korea |
| `zh` | Chinese | `cn` | China |

---

## What changed (April 2026 fix)

Google shut down the `/localservices/prolist` endpoint (HTTP 410).

The scraper now uses a two-step approach:
1. `GET https://www.google.com/maps/search/{query}` — extracts a canonical `pb=` URL from the Maps SPA page.
2. `GET https://www.google.com/search?tbm=map&…&pb=…` — parses the `)]}'`-prefixed JSON at `data[64]`.

`requests-html` / pyppeteer removed; only `aiohttp` and `tqdm` needed for scraping.

---

## Troubleshooting

**Empty results / "Could not find pb= search URL"**
Google may be serving a consent wall. Try matching `--lang` and `--country` to your locale.

**"data[64] is missing"**
Google may have changed the response structure again. Run with `--log-level DEBUG` and open an issue.

**Enriched CSV has empty `web_*` columns**
Check `web_scraped` column — `False` means the domain was unreachable (expected, silent). Try `--web-timeout 20` for slow sites.

**`clients_list` contains non-company names**
Client extraction uses heuristic NER (no heavy NLP library). Logo alt-texts are the highest-precision signal; proper-noun extraction may include some false positives. Use `client_quality_score` as a relative ranking rather than trusting every name literally.

**Large CSVs are slow to enrich**
Web crawling is the bottleneck. Use `--no-web-scraping` for a fast first pass, then deep-crawl only the highest-scoring leads.

---

## License

Provided as-is for educational and research purposes. Please respect Google's Terms of Service.
