import re
from urllib.parse import urlparse


def extract_features(url):
    """
    Extract numerical features from a URL.
    """

    features = {}

    # Make sure URL is a string
    url = str(url)

    # Parse URL
    parsed_url = urlparse(url)

    domain = parsed_url.netloc

    # Remove port number if present
    domain = domain.split(":")[0]

    # 1. URL length
    features["url_length"] = len(url)

    # 2. Domain length
    features["domain_length"] = len(domain)

    # 3. Number of dots
    features["num_dots"] = url.count(".")

    # 4. Number of hyphens
    features["num_hyphens"] = url.count("-")

    # 5. Number of special characters
    special_characters = "@?=&%"
    features["num_special_chars"] = sum(
        url.count(char) for char in special_characters
    )

    # 6. Presence of @ symbol
    features["has_at"] = 1 if "@" in url else 0

    # 7. HTTPS usage
    features["has_https"] = 1 if parsed_url.scheme.lower() == "https" else 0

    # 8. Check whether domain contains an IP address
    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    features["has_ip"] = 1 if re.match(ip_pattern, domain) else 0

    # 9. Number of subdomains
    domain_parts = domain.split(".")

    if len(domain_parts) > 2:
        features["num_subdomains"] = len(domain_parts) - 2
    else:
        features["num_subdomains"] = 0

    # 10. Suspicious keywords
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