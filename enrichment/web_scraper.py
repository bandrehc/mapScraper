"""Deep semantic website scraper for the enrichment pipeline.

For each lead URL, crawls up to `max_pages` pages per domain and extracts
structured business intelligence:

  - Business description (short + long)
  - Services taxonomy match
  - Industry taxonomy match
  - Client / testimonial extraction (primary value signal)
  - Company signals (team size, years of experience, certifications, locations)
  - Technical signals (CMS, tracking tools, modern framework)
  - Computed scores: site_complexity, digital_maturity, client_quality

All failures are silently caught. A failed domain returns _EMPTY (all False / 0 / []).
Backward compatible: every field from the original lightweight scraper is preserved.
"""
import asyncio
import json
import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False
    BeautifulSoup = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP
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

# Cap raw HTML stored per page to keep memory bounded
_MAX_HTML_BYTES = 150_000

# ---------------------------------------------------------------------------
# Crawl priorities & skip patterns
# ---------------------------------------------------------------------------

_PRIORITY_PATHS = [
    'clients', 'testimonials', 'case', 'portfolio', 'work',
    'services', 'solutions', 'about', 'team', 'industries',
    'projects', 'expertise', 'capabilities', 'contact', 'products',
]

_SKIP_RE = re.compile(
    r'/\d{4}/\d{2}/'           # date-based blog URLs
    r'|/page/\d+'              # pagination
    r'|/tag/|/category/|/author/'
    r'|/feed/|/rss'
    r'|/wp-admin/|/wp-login'
    r'|/admin/|/login|/register|/signup'
    r'|/checkout|/cart|/account'
    r'|\.(pdf|jpg|jpeg|png|gif|svg|webp|css|js|xml|json|zip)$',
    re.I,
)

# ---------------------------------------------------------------------------
# NLP constants
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r'\b[a-z]{4,20}\b')
# Max 3 words — limits sentence-fragment false positives
_PROPER_RE = re.compile(r'\b[A-Z][a-zA-Z]{1,30}(?:\s+[A-Z][a-zA-Z]{1,30}){0,2}\b')
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

_STOP_WORDS = frozenset([
    'that', 'this', 'with', 'have', 'from', 'they', 'will', 'been',
    'your', 'more', 'were', 'when', 'there', 'their', 'what', 'about',
    'which', 'also', 'into', 'than', 'then', 'some', 'make', 'just',
    'like', 'time', 'know', 'take', 'year', 'good', 'people', 'come',
    'home', 'page', 'menu', 'search', 'cookie', 'privacy', 'terms',
    'policy', 'copyright', 'rights', 'reserved', 'read', 'more',
    'click', 'here', 'learn', 'view', 'get', 'start', 'free',
])

_GENERIC_PROPER = frozenset([
    'The', 'Our', 'We', 'You', 'They', 'It', 'Its', 'Their', 'Your',
    'Client', 'Clients', 'Customer', 'Customers', 'Partner', 'Partners',
    'Brand', 'Brands', 'Company', 'Companies', 'Business', 'Team',
    'Work', 'Project', 'Service', 'Product', 'Solution', 'Case', 'Study',
    'Testimonial', 'Review', 'Award', 'Expert', 'Professional', 'Digital',
    'About', 'Contact', 'Home', 'Blog', 'News', 'Portfolio', 'Page',
    'Read', 'Learn', 'More', 'View', 'See', 'Get', 'Start', 'Join',
    'Click', 'Here', 'Please', 'Thank', 'Today', 'Free', 'Best', 'Top',
    'New', 'First', 'Second', 'Third', 'Next', 'Last', 'Monday', 'Tuesday',
    'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
    'Industries', 'Services', 'Solutions', 'Projects', 'Results', 'Success',
    'Subscribe', 'Newsletter', 'Follow', 'Share', 'Like', 'Comment',
    'North', 'South', 'East', 'West', 'United', 'States', 'America',
    # Job titles / roles — not company names
    'CEO', 'CMO', 'CTO', 'CFO', 'COO', 'CPO', 'CRO', 'VP', 'SVP', 'EVP',
    'Director', 'Manager', 'Head', 'Lead', 'Senior', 'Junior', 'Associate',
    'President', 'Founder', 'Owner', 'Principal', 'Officer', 'Analyst',
    # Common abbreviations that look like proper nouns
    'ROI', 'SEO', 'PPC', 'SEM', 'KPI', 'CRM', 'ERP', 'API', 'SaaS',
    'B2B', 'B2C', 'GTM', 'CTA', 'UX', 'UI', 'PR', 'HR',
    # Navigation / generic page labels
    'Ads', 'Paid', 'Search', 'Engine', 'Optimization', 'Identity',
    'Working', 'Best', 'Investment', 'Trusted', 'Leading',
])

# ---------------------------------------------------------------------------
# Taxonomies
# ---------------------------------------------------------------------------

_SERVICE_TAXONOMY: Dict[str, List[str]] = {
    'SEO': ['seo', 'search engine optimization', 'organic search', 'keyword ranking', 'backlinks', 'link building'],
    'PPC / Paid Ads': ['ppc', 'pay per click', 'google ads', 'paid search', 'sem', 'facebook ads', 'meta ads', 'paid media', 'programmatic'],
    'Social Media Marketing': ['social media', 'instagram marketing', 'facebook marketing', 'tiktok', 'linkedin marketing', 'community management', 'social strategy'],
    'Content Marketing': ['content marketing', 'content strategy', 'copywriting', 'content creation', 'editorial', 'blogging', 'thought leadership'],
    'Email Marketing': ['email marketing', 'newsletter', 'email campaigns', 'drip campaigns', 'mailchimp', 'klaviyo', 'email automation'],
    'Web Development': ['web development', 'web design', 'website development', 'frontend', 'backend', 'full stack', 'wordpress development', 'custom development'],
    'App Development': ['app development', 'mobile app', 'ios app', 'android app', 'flutter', 'react native', 'mobile development'],
    'Branding': ['branding', 'brand identity', 'logo design', 'visual identity', 'brand strategy', 'rebrand', 'brand guidelines'],
    'UI/UX Design': ['ui/ux', 'user experience', 'user interface', 'ux design', 'product design', 'wireframe', 'prototyping'],
    'E-commerce': ['e-commerce', 'ecommerce', 'shopify', 'woocommerce', 'online store', 'conversion rate optimization', 'cro'],
    'Analytics & Data': ['analytics', 'data analysis', 'business intelligence', 'reporting', 'dashboards', 'data-driven', 'data strategy'],
    'Strategy & Consulting': ['consulting', 'digital strategy', 'management consulting', 'growth strategy', 'go-to-market', 'marketing strategy'],
    'PR & Communications': ['public relations', ' pr ', 'media relations', 'press release', 'communications', 'reputation management'],
    'Video Production': ['video production', 'video marketing', 'videography', 'animation', 'motion graphics', 'explainer video'],
    'Photography': ['photography', 'photo shoot', 'photographer', 'visual content', 'product photography'],
    'Influencer Marketing': ['influencer', 'influencer marketing', 'creator', 'ambassador program'],
    'Marketing Automation': ['marketing automation', 'hubspot', 'salesforce', 'crm', 'lead nurturing', 'automation'],
    'Affiliate Marketing': ['affiliate', 'affiliate marketing', 'performance marketing', 'referral program'],
}

_INDUSTRY_TAXONOMY: Dict[str, List[str]] = {
    'Healthcare': ['healthcare', 'medical', 'hospital', 'clinic', 'pharma', 'health tech', 'telemedicine', 'wellness'],
    'Finance / Fintech': ['finance', 'fintech', 'banking', 'insurance', 'investment', 'financial services', 'wealth management', 'crypto'],
    'Retail / E-commerce': ['retail', 'e-commerce', 'ecommerce', 'consumer goods', 'fashion', 'apparel', 'fmcg', 'd2c', 'direct to consumer'],
    'Technology / SaaS': ['technology', 'tech', 'software', 'saas', 'startup', 'it services', 'cloud', 'platform', 'enterprise software'],
    'Education / EdTech': ['education', 'edtech', 'e-learning', 'training', 'university', 'school', 'online courses', 'certification'],
    'Real Estate': ['real estate', 'property', 'housing', 'construction', 'architecture', 'proptech', 'commercial real estate'],
    'Legal': ['legal', 'law firm', 'attorney', 'lawyer', 'legal services', 'compliance', 'legaltech'],
    'Manufacturing': ['manufacturing', 'industrial', 'production', 'factory', 'supply chain', 'logistics', 'distribution'],
    'Hospitality / Travel': ['hospitality', 'hotel', 'restaurant', 'travel', 'tourism', 'food & beverage', 'airlines', 'booking'],
    'Non-profit / NGO': ['non-profit', 'nonprofit', 'charity', 'ngo', 'social impact', 'foundation', 'cause'],
    'B2B': ['b2b', 'business to business', 'enterprise', 'corporate', 'wholesale'],
    'Automotive': ['automotive', 'car dealership', 'vehicle', 'transportation', 'fleet management', 'mobility'],
    'Energy / Sustainability': ['energy', 'renewable', 'solar', 'sustainability', 'green', 'clean tech', 'esg', 'climate'],
    'Sports & Fitness': ['sports', 'fitness', 'wellness', 'gym', 'athletic', 'nutrition', 'performance'],
    'Media & Entertainment': ['media', 'entertainment', 'publishing', 'streaming', 'gaming', 'content', 'creative'],
    'Government / Public Sector': ['government', 'public sector', 'municipality', 'federal', 'city', 'nonprofit organization'],
}

_VALUE_PROP_KW = frozenset([
    'growth', 'roi', 'revenue', 'performance', 'conversion', 'scaling',
    'results', 'traffic', 'leads', 'sales', 'engagement', 'visibility',
    'efficiency', 'automation', 'data-driven', 'proven', 'trusted',
    'award', 'certified', 'expert', 'innovation', 'transparent',
    'measurable', 'accountable', 'guaranteed', 'optimization',
    'increase', 'improve', 'boost', 'accelerate', 'transform',
])

# ---------------------------------------------------------------------------
# Technical detection
# ---------------------------------------------------------------------------

_CMS_PATTERNS: Dict[str, List[str]] = {
    'WordPress': ['wp-content/', 'wp-includes/', '/wp-json/', 'wordpress'],
    'Webflow': ['webflow.com/', 'data-wf-', '.webflow.io'],
    'Shopify': ['cdn.shopify.com', 'shopify.com/s/'],
    'Wix': ['.wixstatic.com', 'wix.com/'],
    'Squarespace': ['squarespace.com', 'sqsp.net'],
    'Ghost': ['ghost.io/', '/ghost/api'],
    'Drupal': ['drupal', 'sites/default/files'],
    'Joomla': ['/components/com_', 'joomla'],
    'HubSpot CMS': ['hs-scripts.com', 'hubspotusercontent'],
    'Framer': ['framer.com', 'framerusercontent'],
}

_TRACKING_PATTERNS: Dict[str, List[str]] = {
    'Google Analytics': ['google-analytics.com', 'gtag(', 'ga.js', 'analytics.js'],
    'Google Tag Manager': ['googletagmanager.com', 'gtm.js'],
    'Meta Pixel': ['connect.facebook.net', 'fbevents.js', 'fbq('],
    'HotJar': ['hotjar.com', 'hjSetting'],
    'Intercom': ['intercom.io', 'intercomSettings'],
    'Mixpanel': ['mixpanel.com', 'mixpanel.init('],
    'HubSpot Tracking': ['hs-analytics.net', 'hsq.push'],
    'Microsoft Clarity': ['clarity.ms', 'clarity('],
    'LinkedIn Insight': ['snap.licdn.com', 'linkedin.com/insight'],
    'Amplitude': ['amplitude.com', 'amplitude.init('],
}

_MODERN_SIGNALS = ['__next', '__nuxt', 'react', 'vue', 'angular', 'gatsby', 'svelte', '_app.js', 'chunk.js']

# ---------------------------------------------------------------------------
# Company signal patterns
# ---------------------------------------------------------------------------

_CERT_KW = [
    'google partner', 'google certified', 'meta business partner', 'facebook partner',
    'hubspot certified', 'salesforce partner', 'aws certified', 'iso 9001', 'iso 27001',
    'shopify partner', 'wordpress vip', 'bing ads accredited', 'premier google partner',
    'clutch', 'agency of the year', 'inc 500', 'inc500', 'fortune 500',
]

_YEAR_RE = re.compile(
    r'(\d{1,2})\+?\s+years?\s+(?:of\s+)?(?:experience|expertise|in\s+(?:the\s+)?(?:business|industry|marketing|digital))'
    r'|since\s+(19\d{2}|20[012]\d)'
    r'|founded\s+in\s+(19\d{2}|20[012]\d)'
    r'|established\s+in\s+(19\d{2}|20[012]\d)'
    r'|over\s+(\d{1,2})\s+years?',
    re.I,
)

_TEAM_RE = re.compile(
    r'team\s+of\s+(\d+[+]?)'
    r'|(\d+[+]?)\s+(?:team\s+members?|employees?|professionals?|experts?|specialists?|people\s+strong)'
    r'|(\d+[+]?)-?person\s+team',
    re.I,
)

_LOCATION_RE = re.compile(
    r'offices?\s+in\s+([A-Z][a-zA-Z\s,]{3,50}?)(?:[.;(]|$)'
    r'|(?:headquartered|based|located)\s+in\s+([A-Z][a-zA-Z\s]{3,40}?)(?:[,.(]|$)',
    re.I,
)

# ---------------------------------------------------------------------------
# Client section detection
# ---------------------------------------------------------------------------

_CLIENT_SECTION_KW = frozenset([
    'case stud', 'testimonial', 'our clients', 'our customers', 'client',
    'customer', 'trusted by', 'brands we', 'worked with', 'portfolio',
    'our work', 'success stor', 'who we', 'affiliates', 'partners',
])

_CLIENT_ATTR_RE = re.compile(
    r'client|testimonial|partner|customer|logo|brand|trust|case.stud|portfolio|review',
    re.I,
)

_NOISE_TAGS = frozenset(['script', 'style', 'noscript', 'iframe', 'svg',
                          'head', 'form', 'button', 'input', 'select'])

# ---------------------------------------------------------------------------
# Empty result (all fields, backward-compat + new)
# ---------------------------------------------------------------------------

def _empty() -> Dict:
    return {
        # --- backward compat ---
        'web_has_contact': False,
        'web_has_services': False,
        'web_keywords': '',
        'web_is_modern': False,
        'web_scraped': False,
        # --- descriptions ---
        'description_short': '',
        'description_long': '',
        # --- taxonomy ---
        'services_list': '[]',
        'industries_list': '[]',
        # --- client intelligence ---
        'clients_list': '[]',
        'testimonials': '[]',
        'client_count': 0,
        'client_quality_score': 0.0,
        # --- keywords & value props ---
        'keywords_detected': '[]',
        'value_prop_signals': '[]',
        # --- company signals ---
        'team_size_indicator': '',
        'years_experience': '',
        'certifications': '[]',
        'locations': '[]',
        # --- technical ---
        'cms_detected': '',
        'tracking_tools': '[]',
        # --- scores ---
        'site_complexity_score': 0.0,
        'digital_maturity_score': 0.0,
        'content_length': 0,
        'pages_crawled': 0,
    }


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def _fetch(session: aiohttp.ClientSession, url: str, timeout: int) -> Optional[str]:
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
            logger.debug("[%s] HTTP %d", url, resp.status)
    except Exception as exc:
        logger.debug("[%s] fetch error: %s", url, exc)
    return None


# ---------------------------------------------------------------------------
# Link extraction & prioritisation
# ---------------------------------------------------------------------------

def _extract_links(html: str, base_url: str) -> List[str]:
    if not _BS4:
        return []
    base = urlparse(base_url)
    base_domain = base.netloc.lower()
    soup = BeautifulSoup(html, 'html.parser')
    seen: Set[str] = set()
    links: List[str] = []
    for tag in soup.find_all('a', href=True):
        href = str(tag['href']).strip()
        if not href or href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
            continue
        full = urljoin(base_url, href).split('#')[0].rstrip('/')
        parsed = urlparse(full)
        if (parsed.netloc.lower() == base_domain
                and parsed.scheme in ('http', 'https')
                and full not in seen
                and not _SKIP_RE.search(parsed.path)):
            seen.add(full)
            links.append(full)
    return links


def _prioritise(links: List[str]) -> List[str]:
    def rank(url: str) -> int:
        path = urlparse(url).path.lower()
        for i, kw in enumerate(_PRIORITY_PATHS):
            if kw in path:
                return i
        return len(_PRIORITY_PATHS)
    return sorted(links, key=rank)


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

def _clean(html: str) -> Tuple[str, List[str], Optional[str]]:
    """Returns (visible_text, headings, meta_description)."""
    if not _BS4:
        text = re.sub(r'<[^>]+>', ' ', html)
        return re.sub(r'\s+', ' ', text).strip(), [], None

    soup = BeautifulSoup(html, 'html.parser')

    meta_desc: Optional[str] = None
    meta_tag = soup.find('meta', attrs={'name': re.compile(r'^description$', re.I)})
    if meta_tag and meta_tag.get('content'):  # type: ignore[union-attr]
        meta_desc = str(meta_tag['content']).strip()[:400]  # type: ignore[index]

    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    headings: List[str] = []
    for h in soup.find_all(['h1', 'h2', 'h3']):
        t = h.get_text(' ', strip=True)
        if 3 < len(t) < 200:
            headings.append(t)

    text = soup.get_text(' ', strip=True)
    text = re.sub(r'\s+', ' ', text).strip()
    return text, headings, meta_desc


# ---------------------------------------------------------------------------
# Taxonomy matching
# ---------------------------------------------------------------------------

def _match_taxonomy(text: str, headings: List[str], taxonomy: Dict[str, List[str]]) -> List[str]:
    combined = text + ' ' + ' '.join(headings)
    combined_lower = combined.lower()
    return [cat for cat, kws in taxonomy.items() if any(kw in combined_lower for kw in kws)]


# ---------------------------------------------------------------------------
# Client / testimonial extraction
# ---------------------------------------------------------------------------

def _find_client_sections(soup) -> list:
    """Return soup tags that are likely client / testimonial containers."""
    found = []
    # Strategy 1: headings containing client keywords → grab parent container
    for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5']):
        h_text = h.get_text(' ', strip=True).lower()
        if any(kw in h_text for kw in _CLIENT_SECTION_KW):
            parent = h.parent
            if parent and parent not in found:
                found.append(parent)

    # Strategy 2: elements with client-related class / id attributes
    for tag in soup.find_all(True):
        cls = ' '.join(tag.get('class', []))
        eid = tag.get('id', '')
        if _CLIENT_ATTR_RE.search(cls) or _CLIENT_ATTR_RE.search(eid):
            if tag not in found:
                found.append(tag)

    return found


def _is_valid_client_name(name: str) -> bool:
    if not name or len(name) < 3 or len(name) > 55:
        return False
    words = name.split()
    if not words or len(words) > 3:
        return False
    if all(w in _GENERIC_PROPER for w in words):
        return False
    # Reject single-word all-caps abbreviations (ROI, SEO, etc.)
    if len(words) == 1 and words[0].isupper():
        return False
    valid_words = [w for w in words if w not in _GENERIC_PROPER and len(w) > 1]
    return bool(valid_words)


def _extract_proper_nouns(text: str, out: List[str], seen: Set[str]) -> None:
    """Extract capitalised multi-word phrases from non-sentence-start positions."""
    sentences = _SENTENCE_SPLIT.split(text)
    for sent in sentences:
        words = sent.strip().split()
        if len(words) < 2:
            continue
        # Skip first word (sentence start capitalisation)
        remainder = ' '.join(words[1:])
        for m in _PROPER_RE.finditer(remainder):
            name = m.group().strip()
            key = name.lower()
            if key not in seen and _is_valid_client_name(name):
                seen.add(key)
                out.append(name)


def _extract_clients_and_testimonials(
    pages: List[Dict],
) -> Tuple[List[str], List[str]]:
    clients: List[str] = []
    testimonials: List[str] = []
    seen_c: Set[str] = set()
    seen_t: Set[str] = set()

    for page in pages:
        html = page.get('html_raw', '')
        if not html or not _BS4:
            _extract_proper_nouns(page.get('text', ''), clients, seen_c)
            continue

        soup = BeautifulSoup(html, 'html.parser')
        sections = _find_client_sections(soup)

        if not sections:
            # Fall back: scan entire page for proper nouns
            _extract_proper_nouns(page.get('text', ''), clients, seen_c)
            continue

        for section in sections:
            # 1. Alt text from images (most reliable: logos with company names)
            for img in section.find_all('img', alt=True):
                alt = str(img['alt']).strip()
                if alt and 2 <= len(alt) <= 60 and _is_valid_client_name(alt):
                    key = alt.lower()
                    if key not in seen_c:
                        seen_c.add(key)
                        clients.append(alt)

            # 2. Proper noun extraction from section text
            section_text = section.get_text(' ', strip=True)
            _extract_proper_nouns(section_text, clients, seen_c)

            # 3. Explicit quote elements
            for q in section.find_all(['blockquote', 'q']):
                q_text = q.get_text(' ', strip=True)
                if 30 <= len(q_text) <= 600:
                    key = q_text[:80].lower()
                    if key not in seen_t:
                        seen_t.add(key)
                        testimonials.append(q_text[:400])

            # 4. Paragraphs that look like testimonials
            for p in section.find_all(['p', 'div', 'span']):
                p_text = p.get_text(' ', strip=True)
                if _looks_like_testimonial(p_text):
                    key = p_text[:80].lower()
                    if key not in seen_t:
                        seen_t.add(key)
                        testimonials.append(p_text[:400])

    return clients[:60], testimonials[:20]


def _looks_like_testimonial(text: str) -> bool:
    if not 40 <= len(text) <= 600:
        return False
    if text[0] in ('"', '“', '«', "'"):
        return True
    if re.search(r'[–—-]\s+[A-Z]', text):
        return True
    return False


# ---------------------------------------------------------------------------
# Company signals
# ---------------------------------------------------------------------------

def _detect_years(text: str) -> str:
    m = _YEAR_RE.search(text)
    return m.group(0).strip() if m else ''


def _detect_team_size(text: str) -> str:
    m = _TEAM_RE.search(text)
    return m.group(0).strip() if m else ''


def _detect_certifications(text: str) -> List[str]:
    tl = text.lower()
    return [c.title() for c in _CERT_KW if c in tl]


def _detect_locations(text: str) -> List[str]:
    found: List[str] = []
    seen: Set[str] = set()
    for m in _LOCATION_RE.finditer(text):
        loc = (m.group(1) or m.group(2) or '').strip()
        if loc and loc not in seen and len(loc) > 2:
            seen.add(loc)
            found.append(loc[:60])
    return found[:10]


# ---------------------------------------------------------------------------
# Technical signals
# ---------------------------------------------------------------------------

def _detect_cms(html_lower: str) -> str:
    for name, patterns in _CMS_PATTERNS.items():
        if any(p in html_lower for p in patterns):
            return name
    return ''


def _detect_tracking(html_lower: str) -> List[str]:
    return [name for name, pats in _TRACKING_PATTERNS.items()
            if any(p in html_lower for p in pats)]


def _detect_modern(html_lower: str) -> bool:
    return any(s in html_lower for s in _MODERN_SIGNALS)


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def _top_keywords(text: str, n: int = 15) -> List[str]:
    words = _WORD_RE.findall(text.lower())
    freq = Counter(w for w in words if w not in _STOP_WORDS)
    return [w for w, _ in freq.most_common(n)]


# ---------------------------------------------------------------------------
# Description extraction
# ---------------------------------------------------------------------------

def _short_description(meta: Optional[str], text: str) -> str:
    if meta and len(meta) > 20:
        return meta[:300]
    sentences = _SENTENCE_SPLIT.split(text.strip())
    meaningful = [s.strip() for s in sentences if len(s.strip()) > 25]
    return ' '.join(meaningful[:2])[:300]


def _long_description(text: str) -> str:
    chunks = re.split(r'\s{3,}', text.strip())
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) >= 120:
            return chunk[:800]
    return text[:800]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _site_complexity(
    pages_crawled: int,
    services: List[str],
    clients: List[str],
    tracking: List[str],
    cms: str,
    content_len: int,
) -> float:
    score = 0.0
    score += min(pages_crawled / 2.0, 3.0)      # up to 3: page depth
    score += min(len(services) / 2.0, 2.5)      # up to 2.5: service breadth
    score += 1.5 if clients else 0.0             # client section present
    score += min(len(tracking) * 0.4, 1.0)      # up to 1: analytics stack
    score += 0.5 if cms else 0.0                 # CMS detected
    score += 1.0 if content_len > 8_000 else 0.5 if content_len > 3_000 else 0.0
    return round(min(score, 10.0), 1)


def _digital_maturity(
    cms: str,
    tracking: List[str],
    is_modern: bool,
    has_contact: bool,
    has_clients: bool,
    has_testimonials: bool,
    certifications: List[str],
) -> float:
    score = 0.0
    score += 1.0 if cms else 0.0
    score += min(len(tracking) * 0.7, 2.1)      # up to 2.1
    score += 2.0 if is_modern else 0.0
    score += 0.8 if has_contact else 0.0
    score += 1.2 if has_clients else 0.0
    score += 1.2 if has_testimonials else 0.0
    score += min(len(certifications) * 0.5, 1.5)  # up to 1.5
    return round(min(score, 10.0), 1)


def _client_quality(
    clients: List[str],
    testimonials: List[str],
    pages: List[Dict],
) -> float:
    score = 0.0
    # Volume: up to 40 pts
    score += min(len(clients) * 4.0, 40.0)
    # Testimonials: up to 20 pts
    score += min(len(testimonials) * 5.0, 20.0)
    # Case study or portfolio pages found: 20 pts
    has_case_page = any(
        any(kw in p.get('url', '').lower() for kw in ('case', 'portfolio', 'work', 'project'))
        for p in pages
    )
    score += 20.0 if has_case_page else 0.0
    # Name quality: multi-word names suggest real companies (up to 20 pts)
    if clients:
        multi = sum(1 for c in clients if len(c.split()) >= 2)
        score += (multi / len(clients)) * 20.0
    return round(min(score, 100.0), 1)


# ---------------------------------------------------------------------------
# Domain-level orchestration
# ---------------------------------------------------------------------------

async def _crawl_domain(
    session: aiohttp.ClientSession,
    start_url: str,
    semaphore: asyncio.Semaphore,
    max_pages: int,
    timeout: int,
) -> List[Dict]:
    """Fetch up to max_pages pages from start_url's domain. Sequential per domain."""
    url = start_url.strip()
    if not url:
        return []
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    visited: Set[str] = set()
    queue: List[str] = [url]
    pages: List[Dict] = []

    async with semaphore:
        while queue and len(pages) < max_pages:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            html = await _fetch(session, current, timeout)
            if not html:
                continue

            text, headings, meta_desc = _clean(html)
            pages.append({
                'url': current,
                'text': text,
                'headings': headings,
                'meta_desc': meta_desc,
                'html_raw': html,
            })

            # Expand queue from the homepage and from any page that hasn't
            # yet seeded the queue (handles small sites with few links on hp)
            if len(pages) == 1 or (len(queue) == 0 and len(pages) < max_pages):
                for link in _prioritise(_extract_links(html, current)):
                    if link not in visited and link not in queue:
                        queue.append(link)

    return pages


def _build_signals(pages: List[Dict]) -> Dict:
    """Combine all page data into a single structured signal dict."""
    if not pages:
        return _empty()

    all_text = ' '.join(p['text'] for p in pages)
    all_text_lower = all_text.lower()
    all_headings = [h for p in pages for h in p['headings']]
    all_html_lower = ' '.join(p.get('html_raw', '') for p in pages).lower()

    homepage = pages[0]
    hp_text = homepage['text']

    # --- descriptions ---
    desc_short = _short_description(homepage.get('meta_desc'), hp_text)
    desc_long = _long_description(hp_text)

    # --- taxonomy ---
    services = _match_taxonomy(all_text_lower, all_headings, _SERVICE_TAXONOMY)
    industries = _match_taxonomy(all_text_lower, all_headings, _INDUSTRY_TAXONOMY)

    # --- value props ---
    value_props = [kw for kw in _VALUE_PROP_KW if kw in all_text_lower]

    # --- client intelligence ---
    clients, testimonials = _extract_clients_and_testimonials(pages)

    # --- company signals ---
    years_exp = _detect_years(all_text_lower)
    team_size = _detect_team_size(all_text)
    certs = _detect_certifications(all_text_lower)
    locations = _detect_locations(all_text)

    # --- technical ---
    cms = _detect_cms(all_html_lower)
    tracking = _detect_tracking(all_html_lower)
    is_modern = _detect_modern(all_html_lower)

    # --- legacy backward-compat fields ---
    has_contact = any(kw in all_text_lower for kw in ('contact', 'contacto', 'reach us', 'get in touch'))
    has_services = bool(services)

    # --- keywords ---
    top_kw = _top_keywords(all_text)

    # --- metrics ---
    content_len = len(all_text)
    pages_crawled = len(pages)

    # --- scores ---
    complexity = _site_complexity(pages_crawled, services, clients, tracking, cms, content_len)
    maturity = _digital_maturity(cms, tracking, is_modern, has_contact, bool(clients), bool(testimonials), certs)
    client_q = _client_quality(clients, testimonials, pages)

    return {
        # backward compat
        'web_has_contact': has_contact,
        'web_has_services': has_services,
        'web_keywords': ', '.join(top_kw[:10]),
        'web_is_modern': is_modern,
        'web_scraped': True,
        # descriptions
        'description_short': desc_short,
        'description_long': desc_long,
        # taxonomy
        'services_list': json.dumps(services, ensure_ascii=False),
        'industries_list': json.dumps(industries, ensure_ascii=False),
        # client intelligence
        'clients_list': json.dumps(clients, ensure_ascii=False),
        'testimonials': json.dumps(testimonials, ensure_ascii=False),
        'client_count': len(clients),
        'client_quality_score': client_q,
        # signals
        'keywords_detected': json.dumps(top_kw, ensure_ascii=False),
        'value_prop_signals': json.dumps(value_props, ensure_ascii=False),
        # company
        'team_size_indicator': team_size,
        'years_experience': years_exp,
        'certifications': json.dumps(certs, ensure_ascii=False),
        'locations': json.dumps(locations, ensure_ascii=False),
        # technical
        'cms_detected': cms,
        'tracking_tools': json.dumps(tracking, ensure_ascii=False),
        # scores
        'site_complexity_score': complexity,
        'digital_maturity_score': maturity,
        'content_length': content_len,
        'pages_crawled': pages_crawled,
    }


async def _enrich_one(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    max_pages: int,
    timeout: int,
) -> Dict:
    if not url or not url.strip():
        return _empty()
    try:
        pages = await _crawl_domain(session, url, semaphore, max_pages, timeout)
        if pages:
            return _build_signals(pages)
    except Exception as exc:
        logger.debug("[%s] unhandled error: %s", url, exc)
    return _empty()


async def _run_batch(
    urls: List[str],
    max_concurrent: int,
    timeout: int,
    max_pages: int,
) -> List[Dict]:
    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(ssl=False, limit=max_concurrent + 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_enrich_one(session, url, semaphore, max_pages, timeout) for url in urls]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Public API  (signature backward-compatible; max_pages is new keyword arg)
# ---------------------------------------------------------------------------

def enrich_websites(
    urls: List[str],
    *,
    max_concurrent: int = 5,
    batch_size: int = 50,
    timeout: int = 15,
    max_pages: int = 10,
) -> List[Dict]:
    """Crawl and extract deep signals for each URL.

    Parameters
    ----------
    urls:           List of website URLs (from the 'url' column of the scrape CSV).
    max_concurrent: Max concurrent domain crawls (default 5 — each crawls N pages).
    batch_size:     Domains per asyncio.run() call (default 50).
    timeout:        Per-page HTTP timeout in seconds (default 15).
    max_pages:      Max pages crawled per domain (default 10).

    Returns a list of signal dicts aligned 1-to-1 with `urls`.
    """
    if not _BS4:
        logger.warning(
            "beautifulsoup4 not installed — HTML parsing will be degraded. "
            "Run: pip install beautifulsoup4"
        )

    results: List[Dict] = []
    total = len(urls)
    total_batches = (total + batch_size - 1) // batch_size

    for i in range(0, total, batch_size):
        batch = urls[i: i + batch_size]
        batch_num = i // batch_size + 1
        non_empty = sum(1 for u in batch if u and u.strip())
        logger.info(
            "Deep web scraping batch %d/%d — %d/%d URLs with domains",
            batch_num, total_batches, non_empty, len(batch),
        )
        batch_results = asyncio.run(_run_batch(batch, max_concurrent, timeout, max_pages))
        results.extend(batch_results)

    return results
