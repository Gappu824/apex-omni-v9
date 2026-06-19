"""
APEX OMNI v9 — MORNING TOKEN HELPER
===================================
Kite access tokens die nightly (SEBI daily logout), so this is the one
manual step of every trading morning:

    python get_token.py

It prints your login URL; you log in in the browser, copy the
`request_token=` value from the redirect URL, paste it here, and the fresh
access token is written into .env automatically. ~20 seconds.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import config

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

ENV = Path(__file__).resolve().parent / ".env"


def write_env(key: str, value: str):
    lines = ENV.read_text().splitlines() if ENV.exists() else []
    pat = re.compile(rf"^\s*{key}\s*=")
    out, done = [], False
    for ln in lines:
        if pat.match(ln):
            out.append(f"{key}={value}")
            done = True
        else:
            out.append(ln)
    if not done:
        out.append(f"{key}={value}")
    ENV.write_text("\n".join(out) + "\n")


def main():
    if KiteConnect is None:
        sys.exit("pip install kiteconnect first")
    if not config.KITE_API_KEY or config.KITE_API_KEY.startswith("..."):
        sys.exit("Fill KITE_API_KEY / KITE_API_SECRET in .env first.")
    kite = KiteConnect(api_key=config.KITE_API_KEY)
    print("\n1) Open this URL and log in:\n\n   " + kite.login_url())
    print("\n2) After login you land on your redirect URL — copy the value "
          "of `request_token=` from it.\n")
    request_token = input("   Paste request_token: ").strip()
    data = kite.generate_session(request_token,
                                 api_secret=config.KITE_API_SECRET)
    token = data["access_token"]
    write_env("KITE_ACCESS_TOKEN", token)
    print(f"\n✓ access token written to .env for user "
          f"{data.get('user_id', '?')} — valid until tonight's logout.")
    print("  Every process you start from now on picks it up automatically.")


if __name__ == "__main__":
    main()
