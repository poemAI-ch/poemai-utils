from urllib.parse import quote, unquote


def ensure_encoded(url):

    if url is None:
        return None
    # First, decode the URL to handle cases where it is already encoded
    decoded_url = unquote(url)

    # Now, re-encode the URL, ensuring it is quoted exactly once
    encoded_url = quote(decoded_url, safe=":/")

    return encoded_url
