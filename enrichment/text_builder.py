"""Build clean text representations from enriched (or raw) lead rows.

Works with both the raw scrape CSV (13 columns) and the fully enriched CSV
(44+ columns).  Missing columns are silently skipped so the same builder
works at any pipeline stage.
"""
import json
import re
import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

_EMPTY_LIST_VALS = frozenset(['[]', '', 'nan', 'none', 'null'])
_NAN_VALS = frozenset(['nan', 'none', 'null', ''])


def _str(row: pd.Series, col: str) -> str:
    val = row.get(col, '')
    if pd.isna(val):
        return ''
    s = str(val).strip()
    return '' if s.lower() in _NAN_VALS else s


def _json_list(row: pd.Series, col: str) -> List[str]:
    raw = _str(row, col)
    if not raw or raw in _EMPTY_LIST_VALS:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _clean(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def build_text(row: pd.Series, *, include_clients: bool = True) -> str:
    """Return a single clean string summarising one lead for embedding.

    Field priority (richest signal first):
      1. title + category
      2. description (short preferred, fall back to long)
      3. services taxonomy
      4. industries taxonomy
      5. value proposition signals
      6. top keywords
      7. testimonials (first 2, truncated)
      8. clients (optional)
      9. source_query as background context
    """
    parts: List[str] = []

    # Identity
    title = _str(row, 'title')
    category = _str(row, 'category')
    if title:
        parts.append(title)
    if category and category.lower() != title.lower():
        parts.append(category)

    # Description — prefer short (it's the meta description: densest signal)
    desc = _str(row, 'description_short') or _str(row, 'description_long')
    if desc:
        parts.append(_clean(desc)[:400])

    # Services taxonomy
    services = _json_list(row, 'services_list')
    if services:
        parts.append('Services: ' + ', '.join(services))

    # Industries taxonomy
    industries = _json_list(row, 'industries_list')
    if industries:
        parts.append('Industries: ' + ', '.join(industries))

    # Value proposition keywords
    value_props = _json_list(row, 'value_prop_signals')
    if value_props:
        parts.append('Value: ' + ', '.join(value_props))

    # Top content keywords (skip generic web_keywords fallback field)
    keywords = _json_list(row, 'keywords_detected')
    if keywords:
        parts.append('Keywords: ' + ', '.join(keywords[:12]))
    elif _str(row, 'web_keywords'):
        parts.append('Keywords: ' + _str(row, 'web_keywords'))

    # Testimonials (short excerpts)
    testimonials = _json_list(row, 'testimonials')
    if testimonials:
        excerpts = [_clean(t)[:120] for t in testimonials[:2]]
        parts.append('Testimonials: ' + ' | '.join(excerpts))

    # Client names (optional — can add noise for pure topic clustering)
    if include_clients:
        clients = _json_list(row, 'clients_list')
        if clients:
            parts.append('Clients: ' + ', '.join(clients[:15]))

    # Source query: only add if we have very little other signal,
    # to avoid it dominating the embedding for companies with no web data.
    source_query = _str(row, 'source_query')
    if source_query and len(parts) <= 2:
        parts.append(source_query)

    return ' | '.join(parts)


def build_texts(
    df: pd.DataFrame,
    *,
    include_clients: bool = True,
) -> List[str]:
    """Build text representations for every row in df."""
    texts = [build_text(row, include_clients=include_clients) for _, row in df.iterrows()]
    empty = sum(1 for t in texts if not t.strip())
    if empty:
        logger.warning("%d/%d rows produced empty text — will embed as blank string", empty, len(df))
    return texts
