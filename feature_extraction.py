import re
from urllib.parse import urlparse


def extract_features(url):
    """
    Extract numerical features from a URL.
    """
    features = {}
    url = str(url)
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    domain = domain.split(":")[0]
    features["url_length"] = len(url)
    features["domain_length"] = len(domain)
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    special_characters = "@?=&%"
    features["num_special_chars"] = sum(
        url.count(char) for char in special_characters
    )
    features["has_at"] = 1 if "@" in url else 0
    features["has_https"] = 1 if parsed_url.scheme.lower() == "https" else 0
    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    features["has_ip"] = 1 if re.match(ip_pattern, domain) else 0
    domain_parts = domain.split(".")

    if len(domain_parts) > 2:
        features["num_subdomains"] = len(domain_parts) - 2
    else:
        features["num_subdomains"] = 0
    suspicious_words = [
        "login",
        "verify",
        "account",
        "update",
        "secure",
        "password",
        "signin",
        "confirm"
    ]

    url_lower = url.lower()

    features["suspicious_keyword"] = (
        1 if any(word in url_lower for word in suspicious_words) else 0
    )

    return features