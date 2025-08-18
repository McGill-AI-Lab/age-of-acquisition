import re

import requests


class Gutenberg:
    @classmethod
    def get_raw_text_from_gutenberg_url(cls, url: str) -> str:
        response = requests.get(url)
        raw_text = response.text
        text = cls._extract_gutenberg_text(raw_text)

        return text

    @classmethod
    def _extract_gutenberg_text(cls, raw_text: str) -> str:
        pattern = re.compile(
            r"(?is)"  # ignore case, dot matches newlines
            r"\*\*\*\s*START[^*]+?\*\*\*"  # match *** START ... ***
            r"(.*?)"  # capture everything in between
            r"\*\*\*\s*END[^*]+?\*\*\*",  # match *** END ... ***
        )
        match = pattern.search(raw_text)
        if match:
            return match.group(1).strip()
        else:
            raise ValueError("Could not find Gutenberg book body markers.")

