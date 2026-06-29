"""
text_extractor.py
-----------------
Fetches the Wikipedia article on Tarot, extracts paragraph text using
BeautifulSoup, and saves the cleaned content to Selected_Document.txt.

Note: Wikipedia returns HTTP 403 when requests are made without a browser
User-Agent header. This script sets a realistic User-Agent to work around
that restriction. If the request still fails, a helpful error is printed.
"""

import requests
from bs4 import BeautifulSoup

# Hardcoded URL – Wikipedia article on Tarot
URL = "https://en.wikipedia.org/wiki/Tarot"

# Mimic a real browser to avoid Wikipedia's bot-blocking (403)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def scrape_wikipedia(url: str = URL) -> str:
    """
    Fetches the page at `url`, parses all <p> tags inside the main content
    div, joins them with blank lines, and writes the result to
    Selected_Document.txt (UTF-8).

    Returns the extracted text on success, or an empty string on failure.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
    except requests.exceptions.RequestException as e:
        print(f"[FAILURE] Network error: {e}")
        return ""

    if response.status_code != 200:
        print(f"[FAILURE] HTTP {response.status_code} — could not retrieve {url}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    # Target only the main article body to avoid nav/footer noise
    content_div = soup.find("div", {"id": "mw-content-text"})
    if content_div is None:
        content_div = soup  # fallback: search entire page

    paragraphs = content_div.find_all("p")
    text_blocks = [
        p.get_text(separator=" ", strip=True)
        for p in paragraphs
        if p.get_text(strip=True)
    ]
    full_text = "\n\n".join(text_blocks)

    output_path = "Selected_Document.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"[SUCCESS] Extracted {len(text_blocks)} paragraphs ({len(full_text)} chars) → {output_path}")
    return full_text


def main():
    scrape_wikipedia(URL)


if __name__ == "__main__":
    main()
