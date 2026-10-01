"""Validate HTTP authority, browser origin and optional API credentials."""

import hmac
from urllib.parse import urlsplit

if __package__:
    from .errors import APIError
else:
    from errors import APIError


def validate_api_key(value):
    if not value or any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise ValueError("API key must contain only visible ASCII characters")
    return value


def authenticate(headers, key):
    if key is None:
        return
    authorization = headers.get_all("Authorization", [])
    api_keys = headers.get_all("x-api-key", [])
    # Anthropic's SDK sends both headers when it has an API key and an auth
    # token: every credential given must be the key, each header only once.
    supplied = list(api_keys)
    for header in authorization:
        scheme, separator, value = header.partition(" ")
        supplied.append(value if separator and scheme.lower() == "bearer" else None)
    if (
        not supplied
        or len(authorization) > 1
        or len(api_keys) > 1
        or not all(
            value is not None and hmac.compare_digest(value.encode(), key.encode())
            for value in supplied
        )
    ):
        raise APIError(401, "invalid or missing API key", "authentication_error")


def _authority(value):
    if not value or any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise ValueError("invalid authority")
    parsed = urlsplit("//" + value)
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid authority")
    return parsed.hostname.lower().rstrip("."), parsed.port


def _forbidden(message):
    return APIError(403, message, "forbidden")


# Every origin: in --allowed-origin, and in Access-Control-Allow-Origin.
ANY_ORIGIN = "*"
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin(value):
    # An origin as a browser serializes it: a scheme and an authority and
    # nothing after them. A port it does not name is its scheme's default.
    if any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise ValueError("invalid origin")
    parsed = urlsplit(value)
    if not parsed.scheme or parsed.path or parsed.query or parsed.fragment:
        raise ValueError("invalid origin")
    host, port = _authority(parsed.netloc)
    return (
        parsed.scheme,
        host,
        _DEFAULT_PORTS.get(parsed.scheme) if port is None else port,
    )


def parse_allowed_origins(values):
    """The origins --allowed-origin names, as validate_headers compares them."""
    origins = set()
    for value in values:
        try:
            origins.add(value if value == ANY_ORIGIN else _origin(value))
        except ValueError:
            raise ValueError(
                f"{value} is not an origin: expected a scheme and a host, as in "
                "tauri://localhost or http://localhost:3000, or * for every origin"
            ) from None
    return frozenset(origins)


def validate_headers(headers, allowed_hosts, allowed_origins=frozenset()):
    """Refuses a request whose Host or Origin the server does not serve.
    Returns what its response owes a browser in Access-Control-Allow-Origin:
    the origin of a page elsewhere that --allowed-origin admits, or None."""
    hosts = headers.get_all("Host", [])
    origins = headers.get_all("Origin", [])
    if len(hosts) != 1 or len(origins) > 1:
        raise _forbidden("expected one Host header and at most one Origin header")
    try:
        host, port = _authority(hosts[0])
    except ValueError:
        raise _forbidden("invalid Host header") from None
    if host not in allowed_hosts:
        # Only the bind address, loopback and --allowed-host names are served,
        # which keeps DNS-rebound pages out. Name the fix for the operator.
        raise _forbidden(
            f"Host {host} is not allowed; restart the server with "
            f"--allowed-host {host} to accept it"
        )
    if not origins:
        return None
    if ANY_ORIGIN in allowed_origins:
        return ANY_ORIGIN
    try:
        origin = _origin(origins[0])
    except ValueError:
        raise _forbidden("invalid Origin header") from None
    default_port = _DEFAULT_PORTS.get(origin[0])
    if default_port is not None and origin[1:] == (
        host,
        default_port if port is None else port,
    ):
        # The server's own pages.
        return None
    if origin not in allowed_origins:
        # A browser sends Origin with what a page asks of another origin. Only
        # the origins --allowed-origin names may, which keeps the pages of
        # other sites out. Name the fix for the operator.
        raise _forbidden(
            f"Origin {origins[0]} is not allowed; restart the server with "
            f"--allowed-origin {origins[0]} to accept it"
        )
    return origins[0]
