import re
import pandas as pd
from urllib.parse import urlparse

class URLFeatureExtractor:
    def __init__(self):
        self.suspicious_keywords = [
            "login", "verify", "update", "secure", "account",
            "bank", "paypal", "signin", "confirm", "password"
        ]

    def extract_features(self, url: str) -> list:
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path

        features = {}

        features["url_length"] = len(url)
        features["num_dots"] = url.count(".")
        features["num_hyphens"] = url.count("-")
        features["num_underscores"] = url.count("_")
        features["num_slashes"] = url.count("/")
        features["num_question_marks"] = url.count("?")
        features["num_equals"] = url.count("=")
        features["num_ampersands"] = url.count("&")
        features["num_digits"] = sum(c.isdigit() for c in url)
        features["num_alphabets"] = sum(c.isalpha() for c in url)

        features["domain_length"] = len(domain)
        features["subdomain_length"] = max(0, len(domain.split(".")) - 2)
        features["num_subdomains"] = max(0, len(domain.split(".")) - 2)

        features["has_https"] = int(url.startswith("https"))
        features["has_ip"] = int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", domain)))
        features["has_at_symbol"] = int("@" in url)
        features["has_double_slash"] = int("//" in path)

        features["suspicious_keywords_count"] = sum(
            kw in url.lower() for kw in self.suspicious_keywords
        )

        special_chars = re.findall(r"[^\w\s]", url)
        features["special_chars_ratio"] = len(special_chars) / max(1, len(url))

        features["path_length"] = len(path)
        features["num_path_segments"] = len(path.split("/"))
        features["query_length"] = len(parsed.query)
        features["num_query_params"] = len(parsed.query.split("&")) if parsed.query else 0

        return list(features.values())

    def feature_names(self):
        return [
            "url_length", "num_dots", "num_hyphens", "num_underscores",
            "num_slashes", "num_question_marks", "num_equals", "num_ampersands",
            "num_digits", "num_alphabets", "domain_length", "subdomain_length",
            "num_subdomains", "has_https", "has_ip", "has_at_symbol",
            "has_double_slash", "suspicious_keywords_count",
            "special_chars_ratio", "path_length", "num_path_segments",
            "query_length", "num_query_params"
        ]
