"""URL safety: normalization and the internal-target (SSRF) check.

Lives in ``core`` — not ``web`` — because the POLICY GATE needs it and core
must never import outward. The web tools import it from here, so check time
(gate) and use time (fetcher) share one implementation, mirroring how
``core/paths.py`` is shared by the loader and the filesystem tools.

Rule 13's internal-target floor: ``web_fetch`` may never reach loopback,
private/LAN, link-local (including the 169.254.169.254 cloud-metadata
address), reserved, multicast or unspecified addresses, and may only speak
http/https. The check resolves the host and inspects the resolved IPs — a
hostname that merely LOOKS public is not enough.

KNOWN LIMITATION (documented, not hidden): resolving here and connecting
later leaves a DNS-rebinding race — a name that resolves public at check
time could resolve internal at connect time. Mitigated by re-checking
immediately before every connect and on every redirect hop; closing it
fully needs socket-level IP pinning, the flagged upgrade path.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse, urlunparse

ALLOWED_SCHEMES = ("http", "https")
_DEFAULT_PORTS = {"http": 80, "https": 443}


def parse_host(url: str) -> str:
    """Lowercased hostname (no port, no userinfo); '' if unparseable."""
    try:
        return (urlparse(url.strip()).hostname or "").lower()
    except ValueError:
        return ""


def url_domain(url: str) -> str:
    return parse_host(url)


def normalize_url(url: str) -> str:
    """Canonical form for provenance comparison.

    Lowercases scheme/host, drops the default port, userinfo and fragment,
    and treats an empty path as '/'. The QUERY IS PRESERVED EXACTLY — it is
    where exfiltrated data would ride, so '/p' and '/p?data=SECRET' must
    never compare equal.
    """
    try:
        p = urlparse(url.strip())
    except ValueError:
        return url.strip()
    scheme = p.scheme.lower()
    host = (p.hostname or "").lower()
    netloc = f"[{host}]" if ":" in host else host
    try:
        port = p.port
    except ValueError:
        port = None
    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        netloc = f"{netloc}:{port}"
    return urlunparse((scheme, netloc, p.path or "/", p.params, p.query, ""))


def same_origin(a: str, b: str) -> bool:
    pa, pb = urlparse(normalize_url(a)), urlparse(normalize_url(b))
    return (pa.scheme, pa.netloc) == (pb.scheme, pb.netloc)


def resolve_ips(host: str) -> tuple[str, ...]:
    """Resolve a host to IP strings. An IP literal resolves to itself.

    A module-level function on purpose: it is the seam tests monkeypatch to
    exercise the guard without real DNS.
    """
    try:
        ipaddress.ip_address(host)
        return (host,)
    except ValueError:
        pass
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return tuple(dict.fromkeys(info[4][0] for info in infos))


def _ip_problem(raw: str) -> str | None:
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return f"'{raw}' is not a valid IP address"
    # Unwrap IPv4-mapped/6to4 IPv6 so ::ffff:127.0.0.1 cannot sneak through.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    sixtofour = getattr(ip, "sixtofour", None)
    if sixtofour is not None:
        ip = sixtofour
    if ip.is_loopback:
        return "loopback address (internal target)"
    if ip.is_link_local:
        return "link-local address (internal/cloud-metadata target)"
    if ip.is_private:
        return "private/LAN address (internal target)"
    if ip.is_reserved:
        return "reserved address"
    if ip.is_multicast:
        return "multicast address"
    if ip.is_unspecified:
        return "unspecified address"
    if not ip.is_global:
        return "non-public address"
    return None


def check_public_target(url: str) -> str | None:
    """Return a denial reason, or None if the URL targets a public host.

    Fail-closed: an unparseable URL, an unsupported scheme, a missing host
    and a DNS failure are all denials.
    """
    try:
        p = urlparse(url.strip())
    except ValueError as e:
        return f"URL could not be parsed ({type(e).__name__}) — refusing (fail-closed)"
    scheme = p.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        return (
            f"scheme '{scheme or '(none)'}' is not allowed; only "
            f"{'/'.join(ALLOWED_SCHEMES)} may be fetched"
        )
    host = parse_host(url)
    if not host:
        return "URL has no host"
    try:
        ips = resolve_ips(host)
    except OSError as e:
        return (
            f"could not resolve host '{host}' ({type(e).__name__}) — "
            f"refusing (fail-closed)"
        )
    if not ips:
        return f"host '{host}' resolved to no addresses — refusing (fail-closed)"
    for ip in ips:
        problem = _ip_problem(ip)
        if problem is not None:
            return f"host '{host}' resolves to {ip}: {problem}"
    return None


def domain_allowed(domain: str, allowlist: list[str] | tuple[str, ...]) -> bool:
    """Exact host match or a subdomain of an allowlisted host."""
    domain = domain.lower()
    for entry in allowlist:
        entry = entry.strip().lower().lstrip(".")
        if not entry:
            continue
        if domain == entry or domain.endswith("." + entry):
            return True
    return False
