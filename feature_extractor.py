import re
import tldextract
from urllib.parse import urlparse
import pandas as pd

class URLFeatureExtractor:
    def __init__(self):
        self.suspicious_keywords = [
            'login', 'verify', 'account', 'secure', 'update', 'password',
            'bank', 'paypal', 'ebay', 'amazon', 'google', 'microsoft',
            'apple', 'facebook', 'twitter', 'instagram', 'linkedin'
        ]

    def extract_features(self, url):
        """
        Extract features from a single URL
        """
        features = {}

        # Basic URL features
        features['url_length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_question_marks'] = url.count('?')
        features['num_equals'] = url.count('=')
        features['num_ampersands'] = url.count('&')
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_alphabets'] = sum(c.isalpha() for c in url)

        # Parse URL
        parsed = urlparse(url)
        features['scheme'] = parsed.scheme
        features['netloc'] = parsed.netloc
        features['path'] = parsed.path
        features['query'] = parsed.query
        features['fragment'] = parsed.fragment

        # Domain features
        domain_info = tldextract.extract(url)
        features['domain'] = domain_info.domain
        features['subdomain'] = domain_info.subdomain
        features['suffix'] = domain_info.suffix

        features['domain_length'] = len(domain_info.domain)
        features['subdomain_length'] = len(domain_info.subdomain)
        features['num_subdomains'] = len(domain_info.subdomain.split('.')) if domain_info.subdomain else 0

        # Security features
        features['has_https'] = 1 if parsed.scheme == 'https' else 0
        features['has_ip'] = 1 if self._is_ip_address(domain_info.domain) else 0
        features['has_at_symbol'] = 1 if '@' in url else 0
        features['has_double_slash'] = 1 if '//' in url else 0

        # Suspicious keywords
        features['suspicious_keywords_count'] = sum(1 for keyword in self.suspicious_keywords if keyword.lower() in url.lower())

        # Special characters
        features['special_chars_ratio'] = sum(1 for c in url if not c.isalnum() and c not in ['.', '-', '_', '/', '?', '=', '&', ':']) / len(url) if len(url) > 0 else 0

        # Path features
        path_parts = parsed.path.split('/') if parsed.path else []
        features['path_length'] = len(parsed.path)
        features['num_path_segments'] = len([p for p in path_parts if p])

        # Query features
        features['query_length'] = len(parsed.query)
        features['num_query_params'] = len(parsed.query.split('&')) if parsed.query else 0

        return features

    def _is_ip_address(self, domain):
        """
        Check if domain is an IP address
        """
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        return bool(re.match(ip_pattern, domain))

    def extract_features_from_df(self, df, url_column='url'):
        """
        Extract features from a DataFrame containing URLs
        """
        feature_list = []
        for url in df[url_column]:
            features = self.extract_features(url)
            feature_list.append(features)

        features_df = pd.DataFrame(feature_list)
        return pd.concat([df.reset_index(drop=True), features_df], axis=1)

# Example usage
if __name__ == "__main__":
    extractor = URLFeatureExtractor()
    sample_url = "https://login.secure-bank-update.com/verify?user=123&pass=abc"
    features = extractor.extract_features(sample_url)
    print("Extracted features:", features)
