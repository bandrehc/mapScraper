"""Email enrichment module for mapScraper.

Discovers personal / owner contact emails for business domains using:
  1. Fast async crawl of high-value pages (contact, about, team) — aiohttp,
     the same HTTP stack already used by web_scraper.py.
  2. Optional DNS lookup (SOA / DMARC / SPF) via mailget.dns_utils when
     dnspython is installed.
  3. Personal-email scoring: provider type + username pattern + source quality.
  4. Name inference: username splitting → mailto anchor text → nearby-text
     heuristics → schema.org markup.

py-get-mail integration
-----------------------
The mailget package (py-get-mail repo, sibling directory) is imported at
runtime via a sys.path injection — no subprocess, no shell wrapper.
Specifically we reuse:
  • mailget.extractors  — compiled EMAIL_RE, normalize_email, deobfuscation
  • mailget.dns_utils   — async SOA/DMARC/SPF resolution (optional)
If either import fails the module falls back to inline implementations so
mapScraper still works even when py-get-mail is absent.

Public API
----------
    enrich_emails_batch(
        urls, domains, *,
        config: EmailEnrichConfig | None = None,
        ...keyword overrides...
    ) -> List[EmailEnrichResult]

Result helpers
--------------
    result.mails_column          → "a@b.com;c@d.com"
    result.destinataries_column  → "Name <a@b.com>; Unknown <c@d.com>"
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# py-get-mail injection — direct import, zero subprocess
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYGETMAIL_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', 'py-get-mail'))
if os.path.isdir(_PYGETMAIL_ROOT) and _PYGETMAIL_ROOT not in sys.path:
    sys.path.insert(0, _PYGETMAIL_ROOT)
    logger.debug("Injected py-get-mail path: %s", _PYGETMAIL_ROOT)

_MAILGET_EXTRACTORS = False
try:
    from mailget.extractors import extract_emails as _mg_extract_emails       # noqa: E402
    from mailget.extractors import normalize_email as _mg_normalize_email     # noqa: E402
    _MAILGET_EXTRACTORS = True
    logger.debug("mailget.extractors loaded")
except ImportError:
    logger.debug("mailget.extractors unavailable — using inline fallback")

_DNS_AVAILABLE = False
try:
    import dns.asyncresolver  # noqa: F401 — existence check only
    from mailget.dns_utils import resolve_all as _dns_resolve_all             # noqa: E402
    from mailget.config import AuditConfig as _MgAuditConfig                 # noqa: E402
    _DNS_AVAILABLE = True
    logger.debug("mailget.dns_utils + dnspython available")
except ImportError:
    logger.debug("dnspython / mailget.dns_utils unavailable — DNS sources disabled")

# ---------------------------------------------------------------------------
# Inline email extraction (fallback when mailget is absent)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"(?:mailto:)?([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})"
)

_DEOBFUSCATE: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'\s*\[\s*at\s*\]\s*', re.I), '@'),
    (re.compile(r'\s+at\s+', re.I), '@'),
    (re.compile(r'\s*\(\s*at\s*\)\s*', re.I), '@'),
    (re.compile(r'\s*\[\s*dot\s*\]\s*', re.I), '.'),
    (re.compile(r'\s+dot\s+', re.I), '.'),
    (re.compile(r'\s*\(\s*dot\s*\)\s*', re.I), '.'),
]


def _inline_normalize(raw: str) -> str:
    s = raw.strip()
    for pat, rep in _DEOBFUSCATE:
        s = pat.sub(rep, s)
    if s.lower().startswith('mailto:'):
        s = s[7:]
    return s.strip('<> ,;').lower()


def _inline_extract(text: str) -> List[str]:
    if not text:
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for raw in _EMAIL_RE.findall(text):
        e = _inline_normalize(raw)
        if '@' in e and not e.endswith('@') and e not in seen:
            seen.add(e)
            out.append(e)
    return out


def _extract_raw_emails(text: str) -> List[str]:
    """Extract + normalise emails; reuses mailget primitives when available."""
    if _MAILGET_EXTRACTORS:
        return _mg_extract_emails(text, ())   # no pre-filtering — we score later
    return _inline_extract(text)


# ---------------------------------------------------------------------------
# Provider & prefix registries
# ---------------------------------------------------------------------------

PERSONAL_PROVIDERS: frozenset = frozenset([
    # Google
    'gmail.com', 'googlemail.com',
    # Microsoft
    'hotmail.com', 'hotmail.es', 'hotmail.fr', 'hotmail.co.uk', 'hotmail.it',
    'hotmail.com.ar', 'hotmail.com.mx', 'hotmail.com.br',
    'outlook.com', 'outlook.es', 'outlook.fr', 'outlook.com.ar',
    'live.com', 'live.es', 'live.fr', 'live.co.uk', 'live.com.ar',
    'msn.com',
    # Yahoo
    'yahoo.com', 'yahoo.es', 'yahoo.co.uk', 'yahoo.fr',
    'yahoo.com.ar', 'yahoo.com.mx', 'yahoo.com.br', 'ymail.com',
    # Privacy-focused
    'protonmail.com', 'proton.me', 'tutanota.com', 'tutamail.com',
    # Apple
    'icloud.com', 'me.com', 'mac.com',
    # Other well-known free providers
    'aol.com',
    'zoho.com',
    'gmx.com', 'gmx.net', 'gmx.de', 'gmx.at', 'gmx.ch',
    'yandex.com', 'yandex.ru', 'yandex.es',
    'mail.com', 'mail.ru',
    'fastmail.com', 'fastmail.fm',
])

# Generic / non-human prefixes — score is forced to 0 (excluded from output)
GENERIC_PREFIXES: frozenset = frozenset([
    'info', 'information', 'contact', 'contacto', 'contacts', 'contactus',
    'support', 'soporte', 'help', 'ayuda', 'helpcenter', 'helpdesk',
    'admin', 'administrator', 'administrador', 'administration',
    'noreply', 'no-reply', 'noresponder', 'do-not-reply', 'donotreply',
    'sales', 'ventas', 'comercial', 'commerce',
    'office', 'oficina',
    'legal', 'privacy', 'privacidad', 'dpo',
    'jobs', 'careers', 'empleo', 'trabajo', 'hr', 'humanresources', 'recruiting',
    'billing', 'facturas', 'payments', 'invoices', 'accounts',
    'notifications', 'notificaciones', 'alerts', 'alertas', 'noti',
    'webmaster', 'hostmaster', 'postmaster', 'abuse',
    'hello', 'hola', 'hi', 'hey',
    'team', 'equipo',
    'service', 'services', 'servicio', 'servicios',
    'customer', 'customers', 'clients',
    'marketing', 'newsletter', 'news',
    'press', 'media', 'mediarelations', 'pr',
    'security', 'devsecurity',
    'general', 'enquiries', 'enquiry', 'inquiry', 'inquiries',
    'mail', 'email', 'correo',
    'api', 'bot', 'system', 'daemon', 'robot',
    'reservations', 'reservas', 'booking', 'bookings',
    'reception', 'recepcion',
    'feedback', 'suggestions', 'partners',
    'data', 'gdpr', 'compliance',
    'registration', 'register', 'signup', 'unsubscribe',
    'order', 'orders', 'shop', 'store',
    'office1', 'office2', 'info1', 'info2', 'contact1', 'contact2',
])

# ---------------------------------------------------------------------------
# HTTP constants
# ---------------------------------------------------------------------------

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
}

_MAX_HTML_BYTES = 80_000   # cap raw HTML per page; keeps memory bounded

_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)

_SKIP_RE = re.compile(
    r'/\d{4}/\d{2}/'
    r'|/page/\d+'
    r'|/tag/|/category/|/author/'
    r'|/feed/|/rss'
    r'|/wp-admin/|/wp-login'
    r'|/admin/|/login|/register|/signup'
    r'|/checkout|/cart|/account'
    r'|\.(pdf|jpg|jpeg|png|gif|svg|webp|css|js|xml|json|zip)$',
    re.I,
)

# Pages most likely to expose personal emails — pushed to front of crawl queue
_PRIORITY_PATHS: Tuple[str, ...] = (
    'contact', 'contacto', 'about', 'sobre', 'about-us', 'sobre-nosotros',
    'team', 'equipo', 'staff', 'people', 'our-team', 'the-team',
    'us', 'nosotros', 'who-we-are', 'quienes-somos',
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EmailEnrichConfig:
    """All tuneable parameters for one email enrichment pass.

    Defaults are calibrated for speed (≈3-7 s per domain) while maintaining
    good coverage of personal / owner contact emails.
    """

    max_pages: int = 5               # pages crawled per domain
    timeout: int = 10                # per-request HTTP timeout (s)
    max_concurrent: int = 5          # concurrent domain crawls
    min_score: float = 25.0          # minimum score to include an email
    max_emails_per_domain: int = 5   # max emails returned per domain
    include_dns: bool = False         # enable DNS sources (requires dnspython)
    dns_timeout: float = 4.0         # per-DNS-query timeout (s)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class EmailCandidate:
    """A single scored email address with optional name association."""

    email: str
    score: float = 0.0
    provider_type: str = 'unknown'    # 'personal' | 'corporate' | 'generic'
    name_guess: str = ''
    name_confidence: float = 0.0      # 0–1
    sources: List[str] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        if self.name_guess and self.name_confidence >= 0.3:
            return self.name_guess
        return 'Unknown'

    @property
    def destinatary(self) -> str:
        """RFC 5322-style 'Name <email>' string."""
        return f"{self.display_name} <{self.email}>"


@dataclass
class EmailEnrichResult:
    """Enrichment outcome for one domain."""

    domain: str
    candidates: List[EmailCandidate] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    def top_candidates(self, max_n: int = 5) -> List[EmailCandidate]:
        return sorted(self.candidates, key=lambda c: c.score, reverse=True)[:max_n]

    @property
    def mails_column(self) -> str:
        """Semicolon-separated email addresses, highest score first."""
        return ";".join(c.email for c in self.top_candidates())

    @property
    def destinataries_column(self) -> str:
        """Semicolon-separated 'Name <email>' strings, highest score first."""
        return "; ".join(c.destinatary for c in self.top_candidates())


# ---------------------------------------------------------------------------
# Email classification & scoring
# ---------------------------------------------------------------------------


def _is_generic_prefix(local: str) -> bool:
    """True if the local part is a known generic/non-human address prefix."""
    if local in GENERIC_PREFIXES:
        return True
    # Numbered variants: info1, info2, contact1
    if re.sub(r'\d+$', '', local) in GENERIC_PREFIXES:
        return True
    # Hyphen/underscore/dot-collapsed variants: no-reply → noreply
    collapsed = local.replace('-', '').replace('_', '').replace('.', '')
    return collapsed in GENERIC_PREFIXES


def classify_provider(email: str) -> str:
    """Classify an email as 'personal', 'corporate', or 'generic'."""
    try:
        local, domain = email.lower().split('@', 1)
    except ValueError:
        return 'generic'
    if _is_generic_prefix(local):
        return 'generic'
    return 'personal' if domain in PERSONAL_PROVIDERS else 'corporate'


def _score_username(local: str) -> float:
    """Score the local part of an email address (0–35 pts).

    Signal: does the username look like a human name?
    """
    clean = re.sub(r'\d+$', '', local.lower())   # strip trailing digits
    parts = re.split(r'[._\-+]', clean)
    alpha_parts = [p for p in parts if p.isalpha() and len(p) >= 2]

    if len(alpha_parts) >= 2:
        return 35.0   # john.smith, maria_lopez, juan-perez
    if len(alpha_parts) == 1 and len(parts) >= 2:
        return 22.0   # j.smith, m.rodriguez (initial + surname)
    if len(alpha_parts) == 1 and len(alpha_parts[0]) >= 5:
        return 15.0   # carla, carlos — single long name
    if len(alpha_parts) == 1 and len(alpha_parts[0]) >= 3:
        return 10.0   # single shorter name
    # Mixed alphanumeric (john123) — strip digits and re-evaluate length
    alpha_only = re.sub(r'[^a-z]', '', clean)
    if len(alpha_only) >= 4:
        return 8.0
    return 3.0


def score_email(email: str, sources: Optional[List[str]] = None) -> float:
    """Return a 0–100 personal-contact score for *email*.

    Score of 0 means the email is excluded (generic prefix or malformed).
    Higher scores indicate a more likely personal / owner contact.

    Breakdown (max 100):
      Provider type   0–40  (personal provider >> corporate >> unknown)
      Username        0–35  (name pattern quality)
      Source quality  0–20  (contact/about/team page >> homepage)
      Recurrence      0– 5  (appears on multiple pages)
    """
    try:
        local, domain_part = email.lower().split('@', 1)
    except ValueError:
        return 0.0

    if _is_generic_prefix(local):
        return 0.0   # hard filter

    score = 0.0

    # Provider type
    if domain_part in PERSONAL_PROVIDERS:
        score += 40.0
    else:
        score += 15.0   # corporate domain — still a valuable lead if name-like

    # Username quality
    score += _score_username(local)

    # Source quality
    if sources:
        src = ' '.join(s.lower() for s in sources)
        if any(p in src for p in ('contact', 'contacto')):
            score += 15.0
        elif any(p in src for p in ('about', 'team', 'staff', 'people', 'sobre', 'equipo')):
            score += 12.0
        else:
            score += 5.0
        if len(sources) > 1:
            score += 5.0   # appeared in multiple pages → higher confidence

    return min(round(score, 1), 100.0)


# ---------------------------------------------------------------------------
# Name inference
# ---------------------------------------------------------------------------

# Compiled once at import — used across all domains
_MAILTO_ANCHOR_RE = re.compile(
    r'<a[^>]+href=["\']mailto:([^"\'?]+)[^"\']*["\'][^>]*>\s*([^<]{2,80}?)\s*</a>',
    re.IGNORECASE,
)
_NAME_BEFORE_EMAIL_RE_TEMPLATE = (
    r'([A-ZÁÉÍÓÚÜÑÀ][a-záéíóúüñà]{1,25}'
    r'(?:\s+[A-ZÁÉÍÓÚÜÑÀ][a-záéíóúüñà]{1,25}){1,3})'
    r'\s{{0,10}}[<(]?\s*(?:mailto:)?{email}'
)
_SCHEMA_NAME_RE = re.compile(
    r'(?:itemprop|property)=["\'](?:name|author)["\'][^>]*>\s*([^<]{2,80})',
    re.IGNORECASE,
)
_GENERIC_LINK_TEXT = frozenset([
    'email', 'mail', 'contact', 'email us', 'send email', 'click here',
    'here', 'send', 'write', 'get in touch', 'reach us', 'reach out',
    'contact us', 'write to us',
])


def _name_from_username(local: str) -> Tuple[str, float]:
    """Derive a human name from the email local part. Returns (name, confidence)."""
    clean = re.sub(r'\d+$', '', local.lower()).strip()
    parts = re.split(r'[._\-+]', clean)
    alpha_parts = [p for p in parts if p.isalpha() and len(p) >= 2]

    if len(alpha_parts) >= 2:
        name = ' '.join(p.capitalize() for p in alpha_parts[:2])
        return name, 0.70

    # Single initial + surname: j.smith → "J Smith"
    if len(parts) >= 2 and len(parts[0]) == 1:
        rest = [p for p in parts[1:] if p.isalpha() and len(p) >= 2]
        if rest:
            return f"{parts[0].upper()} {rest[0].capitalize()}", 0.40

    if len(alpha_parts) == 1 and len(alpha_parts[0]) >= 5:
        return alpha_parts[0].capitalize(), 0.25

    return '', 0.0


def _name_from_context(email: str, html: str) -> Tuple[str, float]:
    """Scan raw HTML for a name near *email*. Returns (name, confidence)."""
    if not html or not email:
        return '', 0.0

    email_lower = email.lower()

    # Strategy 1: <a href="mailto:email">Name</a>
    for m in _MAILTO_ANCHOR_RE.finditer(html):
        href_email = m.group(1).strip().lower()
        text = m.group(2).strip()
        if href_email != email_lower:
            continue
        if text.lower() in _GENERIC_LINK_TEXT or len(text) < 3:
            continue
        # Must start with an uppercase letter (real name, not a URL or code)
        if re.match(r'^[A-ZÁÉÍÓÚÜÑ]', text, re.UNICODE):
            return text[:60], 0.85

    # Strategy 2: "Name <email>" or "Name (email)" pattern nearby in text
    try:
        pattern = re.compile(
            _NAME_BEFORE_EMAIL_RE_TEMPLATE.format(email=re.escape(email_lower)),
            re.IGNORECASE | re.UNICODE,
        )
        m = pattern.search(html)
        if m:
            candidate = m.group(1).strip()
            confidence = 0.80 if len(candidate.split()) >= 2 else 0.50
            return candidate[:60], confidence
    except re.error:
        pass

    # Strategy 3: schema.org itemprop="name" or "author" near the email
    email_pos = html.lower().find(email_lower)
    if email_pos >= 0:
        window = html[max(0, email_pos - 600): email_pos + 200]
        for m in _SCHEMA_NAME_RE.finditer(window):
            text = m.group(1).strip()
            if text and re.match(r'^[A-ZÁÉÍÓÚÜÑ][a-z]', text, re.UNICODE):
                return text[:60], 0.75

    return '', 0.0


def infer_name(
    email: str,
    pages: Optional[List[Dict]] = None,
) -> Tuple[str, float]:
    """Return *(name, confidence)* for *email*.

    Confidence scale: 0 = unknown, 0.3–0.5 = plausible, 0.7+ = high confidence.
    Sources tried in priority order:
      1. Username parsing (fast, no page context)
      2. mailto anchor text in page HTML
      3. "Name <email>" proximity pattern
      4. schema.org itemprop="name" / "author"
    """
    name, conf = _name_from_username(email.split('@')[0])

    if pages and conf < 0.85:
        for page in pages:
            n, c = _name_from_context(email, page.get('html_raw', ''))
            if c > conf:
                name, conf = n, c
            if conf >= 0.85:
                break

    return name, conf


# ---------------------------------------------------------------------------
# Async HTTP helpers
# ---------------------------------------------------------------------------


async def _fetch_page(
    session: aiohttp.ClientSession,
    url: str,
    timeout: int,
) -> Optional[str]:
    """GET *url*; return raw HTML (capped at _MAX_HTML_BYTES) or None."""
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    try:
        async with session.get(
            url,
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=timeout),
            ssl=False,
            allow_redirects=True,
            max_redirects=5,
        ) as resp:
            if resp.status == 200:
                raw = await resp.read()
                return raw[:_MAX_HTML_BYTES].decode('utf-8', errors='replace')
            logger.debug("[email] HTTP %d for %s", resp.status, url)
    except Exception as exc:
        logger.debug("[email] Fetch error for %s: %s", url, exc)
    return None


def _extract_internal_links(
    html: str, base_url: str, base_netloc: str,
) -> Tuple[List[str], List[str]]:
    """Split page links into priority (contact/about/team) and normal buckets."""
    priority: List[str] = []
    normal: List[str] = []
    seen: Set[str] = set()

    for href in _HREF_RE.findall(html):
        href = href.strip()
        if not href or href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
            continue
        full = urljoin(base_url, href).split('#')[0].rstrip('/')
        parsed = urlparse(full)
        if (
            parsed.netloc.lower() == base_netloc
            and parsed.scheme in ('http', 'https')
            and full not in seen
            and not _SKIP_RE.search(parsed.path)
        ):
            seen.add(full)
            path_lower = parsed.path.lower()
            if any(kw in path_lower for kw in _PRIORITY_PATHS):
                priority.append(full)
            else:
                normal.append(full)

    return priority, normal


async def _crawl_domain_for_emails(
    session: aiohttp.ClientSession,
    start_url: str,
    semaphore: asyncio.Semaphore,
    config: EmailEnrichConfig,
) -> List[Dict]:
    """Crawl up to *config.max_pages* pages from *start_url*, priority-first.

    Returns a list of {url, html_raw, text} dicts.
    Priority queue drains before the normal queue, ensuring contact/about/team
    pages are fetched even when max_pages is small.
    """
    url = start_url.strip()
    if not url:
        return []
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    base_netloc = urlparse(url).netloc.lower()
    visited: Set[str] = set()
    priority_q: List[str] = [url]
    normal_q: List[str] = []
    pages: List[Dict] = []

    async with semaphore:
        while (priority_q or normal_q) and len(pages) < config.max_pages:
            current = priority_q.pop(0) if priority_q else normal_q.pop(0)
            if current in visited:
                continue
            visited.add(current)

            html = await _fetch_page(session, current, config.timeout)
            if not html:
                continue

            # Lightweight text strip for name-context search
            text = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', html)).strip()

            pages.append({'url': current, 'html_raw': html, 'text': text})

            # Discover more links from homepage and whenever queue is exhausted
            if len(pages) == 1 or (not priority_q and not normal_q):
                pri, nrm = _extract_internal_links(html, current, base_netloc)
                for link in pri:
                    if link not in visited and link not in priority_q:
                        priority_q.append(link)
                for link in nrm:
                    if link not in visited and link not in normal_q:
                        normal_q.append(link)

    return pages


# ---------------------------------------------------------------------------
# Email candidate extraction from crawled pages
# ---------------------------------------------------------------------------


def _page_label(url: str) -> str:
    """Short label for a page URL — used as the 'source' signal in scoring."""
    path = urlparse(url).path.lower().rstrip('/')
    if not path or path == '/':
        return 'homepage'
    for kw in ('contact', 'contacto', 'about', 'sobre', 'team', 'equipo', 'staff', 'people'):
        if kw in path:
            return kw
    return 'page'


def _build_candidates(
    pages: List[Dict],
    config: EmailEnrichConfig,
) -> List[EmailCandidate]:
    """Extract, score, and name-infer all email candidates from *pages*."""
    email_sources: Dict[str, List[str]] = {}

    for page in pages:
        raw_emails = _extract_raw_emails(page['html_raw'] + ' ' + page.get('text', ''))
        label = _page_label(page['url'])
        for email in raw_emails:
            email = email.lower()
            if email not in email_sources:
                email_sources[email] = []
            if label not in email_sources[email]:
                email_sources[email].append(label)

    candidates: List[EmailCandidate] = []
    for email, sources in email_sources.items():
        if classify_provider(email) == 'generic':
            continue   # hard filter — no point scoring

        s = score_email(email, sources)
        if s < config.min_score:
            continue

        name, conf = infer_name(email, pages)
        provider_type = classify_provider(email)

        candidates.append(EmailCandidate(
            email=email,
            score=s,
            provider_type=provider_type,
            name_guess=name,
            name_confidence=conf,
            sources=sources,
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    # Keep extra headroom so DNS merge below can deduplicate
    return candidates[: config.max_emails_per_domain * 3]


# ---------------------------------------------------------------------------
# Optional DNS sources (SOA / DMARC / SPF)
# ---------------------------------------------------------------------------


async def _dns_candidates(
    domain: str,
    config: EmailEnrichConfig,
) -> List[EmailCandidate]:
    """Resolve SOA/DMARC/SPF for *domain*; return scored candidates.

    Silently skipped when DNS is disabled or dnspython is unavailable.
    """
    if not _DNS_AVAILABLE or not config.include_dns:
        return []
    try:
        mg_cfg = _MgAuditConfig(
            dns_timeout=config.dns_timeout,
            dns_lifetime=config.dns_timeout * 2.5,
            exclude_email_keywords=(),   # we score ourselves
        )
        soa, dmarc_list, spf_list = await _dns_resolve_all(domain, mg_cfg)
        dns_emails: List[str] = []
        if soa:
            dns_emails.append(soa)
        dns_emails.extend(dmarc_list)
        dns_emails.extend(spf_list)

        out: List[EmailCandidate] = []
        for email in dns_emails:
            if classify_provider(email) == 'generic':
                continue
            s = score_email(email, ['dns'])
            if s >= config.min_score:
                name, conf = infer_name(email)
                out.append(EmailCandidate(
                    email=email,
                    score=s,
                    provider_type=classify_provider(email),
                    name_guess=name,
                    name_confidence=conf,
                    sources=['dns'],
                ))
        return out
    except Exception as exc:
        logger.debug("[email] DNS error for %s: %s", domain, exc)
        return []


# ---------------------------------------------------------------------------
# Per-domain async orchestration
# ---------------------------------------------------------------------------


async def _enrich_one_domain(
    session: aiohttp.ClientSession,
    url: str,
    domain: str,
    semaphore: asyncio.Semaphore,
    config: EmailEnrichConfig,
) -> EmailEnrichResult:
    t0 = time.monotonic()
    result = EmailEnrichResult(domain=domain)

    try:
        # 1. Web crawl (priority: contact/about/team pages)
        start = url or ('https://' + domain if domain else '')
        pages = await _crawl_domain_for_emails(session, start, semaphore, config)
        web_candidates = _build_candidates(pages, config) if pages else []

        # 2. DNS sources (optional, async)
        dns_cands = await _dns_candidates(domain, config)

        # 3. Merge & deduplicate by email address (web results win on conflict)
        seen: Set[str] = set()
        merged: List[EmailCandidate] = []
        for c in web_candidates + dns_cands:
            key = c.email.lower()
            if key not in seen:
                seen.add(key)
                merged.append(c)

        merged.sort(key=lambda c: c.score, reverse=True)
        result.candidates = merged[: config.max_emails_per_domain]

    except Exception as exc:
        result.error = str(exc)
        logger.debug("[email] Unexpected error for %s: %s", domain, exc)

    result.elapsed_seconds = round(time.monotonic() - t0, 2)
    logger.debug(
        "[email] %s — %d candidates in %.1fs",
        domain or url, len(result.candidates), result.elapsed_seconds,
    )
    return result


async def _run_email_batch_async(
    rows: List[Tuple[str, str]],     # (url, domain) pairs
    config: EmailEnrichConfig,
) -> List[EmailEnrichResult]:
    semaphore = asyncio.Semaphore(config.max_concurrent)
    connector = aiohttp.TCPConnector(ssl=False, limit=config.max_concurrent + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _enrich_one_domain(session, url, domain, semaphore, config)
            for url, domain in rows
        ]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enrich_emails_batch(
    urls: List[str],
    domains: Optional[List[str]] = None,
    *,
    config: Optional[EmailEnrichConfig] = None,
    # Keyword overrides (used when config is None)
    max_concurrent: int = 5,
    timeout: int = 10,
    max_pages: int = 5,
    include_dns: bool = False,
    batch_size: int = 50,
) -> List[EmailEnrichResult]:
    """Enrich a list of URLs/domains with personal contact email discovery.

    Parameters
    ----------
    urls:
        Website URLs — from mapScraper's ``url`` column.  Empty strings allowed.
    domains:
        Domain names — from mapScraper's ``domain`` column.  Used as fallback
        when the URL is empty.  Must be the same length as ``urls`` if provided.
    config:
        Explicit ``EmailEnrichConfig``.  When ``None`` the keyword overrides below
        are used to build one.
    max_concurrent, timeout, max_pages, include_dns:
        Quick overrides (ignored when *config* is supplied).
    batch_size:
        Domains per ``asyncio.run()`` call (prevents runaway memory on huge CSVs).

    Returns
    -------
    List[EmailEnrichResult] — one entry per input row, in the same order.
    Rows with no URL and no domain return an empty result (no candidates).
    """
    if config is None:
        config = EmailEnrichConfig(
            max_concurrent=max_concurrent,
            timeout=timeout,
            max_pages=max_pages,
            include_dns=include_dns,
        )

    if domains is None:
        domains = [''] * len(urls)

    def _normalise(url: str, domain: str) -> Tuple[str, str]:
        u = (url or '').strip()
        d = (domain or '').strip()
        if not d and u:
            try:
                parsed = urlparse(u if '://' in u else 'https://' + u)
                d = re.sub(r'^www\.', '', parsed.netloc)
            except Exception:
                pass
        if not u and d:
            u = 'https://' + d
        return u, d

    rows = [_normalise(u, d) for u, d in zip(urls, domains)]

    results: List[EmailEnrichResult] = []
    total = len(rows)
    n_batches = (total + batch_size - 1) // batch_size

    for i in range(0, total, batch_size):
        batch = rows[i: i + batch_size]
        batch_num = i // batch_size + 1
        active = sum(1 for u, d in batch if u or d)
        logger.info(
            "Email enrichment batch %d/%d — %d/%d rows with domains",
            batch_num, n_batches, active, len(batch),
        )
        batch_results = asyncio.run(_run_email_batch_async(batch, config))
        results.extend(batch_results)

    found = sum(1 for r in results if r.candidates)
    logger.info(
        "Email enrichment complete: %d/%d domains yielded ≥1 candidate",
        found, total,
    )
    return results
