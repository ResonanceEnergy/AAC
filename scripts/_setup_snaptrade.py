#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
SnapTrade Setup — Link Wealthsimple to AAC
═══════════════════════════════════════════════════════════════════════════
Step-by-step setup for SnapTrade → Wealthsimple connection.

Prerequisites:
    1. Sign up at https://dashboard.snaptrade.com
    2. Get your clientId + consumerKey
    3. Add to .env:
         SNAPTRADE_CLIENT_ID=your_client_id
         SNAPTRADE_CONSUMER_KEY=your_consumer_key

Usage:
    python _setup_snaptrade.py --register     # Create SnapTrade user (one-time)
    python _setup_snaptrade.py --connect      # Generate OAuth link for Wealthsimple
    python _setup_snaptrade.py --status       # Check connection status
    python _setup_snaptrade.py --accounts     # List linked accounts
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv, set_key

# Resolve project-root .env (script lives in scripts/, .env lives in repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = str(PROJECT_ROOT / ".env")
load_dotenv(ENV_PATH, override=True)


def _extract_response_body(response):
    """Return API payload from v11 ApiResponse or plain dict/list."""
    if hasattr(response, "body"):
        return response.body
    return response


def _get_user_credentials() -> tuple[str, str]:
    """Return configured SnapTrade user id and secret from environment."""
    return os.getenv("SNAPTRADE_USER_ID", ""), os.getenv("SNAPTRADE_USER_SECRET", "")


def _print_missing_user_secret(user_id: str) -> None:
    """Explain the recovery path when a personal-key user exists without its secret."""
    print(f"SnapTrade user exists but SNAPTRADE_USER_SECRET is missing for: {user_id}")
    print("This blocks --connect, --status, and --accounts.")
    print()
    print("Recovery options:")
    print("  1. In SnapTrade dashboard, find/reset this user's secret and save it to .env as SNAPTRADE_USER_SECRET")
    print("  2. Or create a fresh personal API key pair, replace SNAPTRADE_CLIENT_ID / SNAPTRADE_CONSUMER_KEY, then run --register again")


def get_client():
    """Initialize SnapTrade client."""
    try:
        from snaptrade_client.client import SnapTrade
    except ImportError:
        print("ERROR: snaptrade-python-sdk not installed.")
        print("  Run: .venv\\Scripts\\pip install snaptrade-python-sdk")
        sys.exit(1)

    client_id = os.getenv("SNAPTRADE_CLIENT_ID", "")
    consumer_key = os.getenv("SNAPTRADE_CONSUMER_KEY", "")

    if not client_id or not consumer_key:
        print("ERROR: SNAPTRADE_CLIENT_ID and SNAPTRADE_CONSUMER_KEY must be set in .env")
        print()
        print("Steps:")
        print("  1. Go to https://dashboard.snaptrade.com")
        print("  2. Sign up / log in")
        print("  3. Copy your clientId and consumerKey")
        print("  4. Add to .env:")
        print("       SNAPTRADE_CLIENT_ID=your_client_id")
        print("       SNAPTRADE_CONSUMER_KEY=your_consumer_key")
        sys.exit(1)

    from snaptrade_client import Configuration
    cfg = Configuration(consumer_key=consumer_key, client_id=client_id)
    return SnapTrade(configuration=cfg), client_id


def cmd_register(args):
    """Register a new SnapTrade user (one-time)."""
    snap, client_id = get_client()

    existing_user_id, existing_user_secret = _get_user_credentials()
    if existing_user_id:
        print(f"User already registered: SNAPTRADE_USER_ID={existing_user_id}")
        if not existing_user_secret:
            _print_missing_user_secret(existing_user_id)
        print("To re-register, remove SNAPTRADE_USER_ID and SNAPTRADE_USER_SECRET from .env")
        return

    # Generate a unique user ID
    user_id = f"aac-{uuid.uuid4().hex[:12]}"
    print(f"Registering SnapTrade user: {user_id}")

    try:
        response = snap.authentication.register_snap_trade_user(
            user_id=user_id,
        )
        payload = _extract_response_body(response)
        user_secret = ""
        if isinstance(payload, dict):
            user_secret = payload.get("userSecret") or payload.get("user_secret") or ""
        else:
            user_secret = getattr(payload, "userSecret", "") or getattr(payload, "user_secret", "")

        if not user_secret:
            print(f"ERROR: Registration response missing userSecret: {response}")
            return

        # Save to .env
        set_key(ENV_PATH, "SNAPTRADE_USER_ID", user_id)
        set_key(ENV_PATH, "SNAPTRADE_USER_SECRET", user_secret)

        print(f"  User ID:     {user_id}")
        print(f"  User Secret: {user_secret[:20]}...")
        print(f"  Saved to .env")
        print()
        print("Next step: python _setup_snaptrade.py --connect")

    except Exception as e:
        msg = str(e)
        if "Personal keys can only register one user" in msg:
            print("This personal key already has a registered user.")
            try:
                users_resp = snap.authentication.list_snap_trade_users()
                users = _extract_response_body(users_resp) or []
                if users:
                    existing = str(users[0])
                    set_key(ENV_PATH, "SNAPTRADE_USER_ID", existing)
                    print(f"Saved existing SNAPTRADE_USER_ID={existing} to .env")
                    print("Next: obtain/reset SNAPTRADE_USER_SECRET in SnapTrade dashboard, then run --connect")
                else:
                    print("No users returned by list_snap_trade_users().")
            except Exception as inner:
                print(f"Could not list existing users: {inner}")
            return
        print(f"ERROR: Registration failed: {e}")


def cmd_connect(args):
    """Generate OAuth connection link for Wealthsimple."""
    snap, client_id = get_client()

    user_id, user_secret = _get_user_credentials()

    if not user_id or not user_secret:
        if user_id and not user_secret:
            _print_missing_user_secret(user_id)
        else:
            print("ERROR: Run --register first to create a SnapTrade user")
        return

    print(f"Generating connection portal for user: {user_id}")

    try:
        response = snap.authentication.login_snap_trade_user(
            user_id=user_id,
            user_secret=user_secret,
        )
        payload = _extract_response_body(response)

        redirect_url = None
        if isinstance(payload, dict):
            redirect_url = payload.get("redirectURI") or payload.get("redirect_uri")
        else:
            redirect_url = getattr(payload, "redirectURI", None) or getattr(payload, "redirect_uri", None)

        if redirect_url:
            print()
            print("═" * 60)
            print("  OPEN THIS URL IN YOUR BROWSER:")
            print(f"  {redirect_url}")
            print("═" * 60)
            print()
            print("  1. Click the link above (or copy/paste into browser)")
            print("  2. Select 'Wealthsimple' as your brokerage")
            print("  3. Log in with your Wealthsimple credentials")
            print("  4. Authorize the connection")
            print("  5. Come back here and run: python _setup_snaptrade.py --status")
        else:
            print(f"Response: {response}")
            print("Could not extract redirect URL from response")

    except Exception as e:
        print(f"ERROR: {e}")


def cmd_status(args):
    """Check connection status."""
    snap, client_id = get_client()

    user_id, user_secret = _get_user_credentials()

    if not user_id or not user_secret:
        if user_id and not user_secret:
            _print_missing_user_secret(user_id)
        else:
            print("Not registered. Run: python _setup_snaptrade.py --register")
        return

    print(f"User: {user_id}")
    print()

    try:
        # List brokerage connections
        connections_resp = snap.connections.list_brokerage_authorizations(
            user_id=user_id,
            user_secret=user_secret,
        )
        connections = _extract_response_body(connections_resp) or []

        if not connections:
            print("No brokerage connections found.")
            print("Run: python _setup_snaptrade.py --connect")
            return

        print(f"Brokerage Connections ({len(connections)}):")
        for conn in connections:
            if isinstance(conn, dict):
                brokerage = conn.get("brokerage", {})
                status = conn.get("disabled", False)
                status = "disabled" if status else "active"
            else:
                brokerage = getattr(conn, "brokerage", {})
                status = getattr(conn, "status", "unknown")
            if isinstance(brokerage, dict):
                name = brokerage.get("name") or brokerage.get("display_name") or "Unknown"
            else:
                name = str(brokerage)
            print(f"  - {name}: {status}")

    except Exception as e:
        print(f"ERROR: {e}")


def cmd_accounts(args):
    """List linked accounts and balances."""
    snap, client_id = get_client()

    user_id, user_secret = _get_user_credentials()

    if not user_id or not user_secret:
        if user_id and not user_secret:
            _print_missing_user_secret(user_id)
        else:
            print("Not registered. Run: python _setup_snaptrade.py --register")
        return

    try:
        accounts = snap.account_information.list_user_accounts(
            user_id=user_id,
            user_secret=user_secret,
        )

        if not accounts.body:
            print("No accounts found. Connect a brokerage first.")
            print("Run: python _setup_snaptrade.py --connect")
            return

        print(f"Linked Accounts ({len(accounts.body)}):")
        for acct in accounts.body:
            if isinstance(acct, dict):
                acct_id = acct.get('id', '?')
                name = acct.get('name') or acct.get('institution_name') or str(acct_id)
            else:
                acct_id = getattr(acct, 'id', '?')
                name = getattr(acct, 'name', None) or getattr(acct, 'institution_name', None) or str(acct_id)
            # Get balance for each account
            try:
                bal = snap.account_information.get_user_account_balance(
                    user_id=user_id, user_secret=user_secret, account_id=str(acct_id)
                )
                items = bal.body if hasattr(bal, 'body') else bal
                for b in (items or []):
                    if isinstance(b, dict):
                        cur = b.get('currency', {})
                        currency = cur.get('code') if isinstance(cur, dict) else (cur or 'CAD')
                        cash = b.get('cash', 0) or 0
                        buying_power = b.get('buying_power', 0) or 0
                        print(f"  {name} [{acct_id}]: {currency} cash={cash:,.2f} bp={buying_power:,.2f}")
                    else:
                        currency = getattr(b, 'currency', 'CAD')
                        total = getattr(b, 'total', 0) or getattr(b, 'cash', 0) or 0
                        print(f"  {name} [{acct_id}]: {currency} {total:,.2f}")
            except Exception as e:
                print(f"  {name} [{acct_id}]: (balance fetch failed: {e})")

    except Exception as e:
        print(f"ERROR: {e}")


def cmd_auto(args):
    """End-to-end automation: register (if needed) → open portal → poll → list accounts → refresh live balances.

    Idempotent. Safe to re-run. The only manual step left is the in-browser
    Wealthsimple login when the portal page opens.
    """
    import time
    import webbrowser

    snap, _client_id = get_client()
    user_id, user_secret = _get_user_credentials()

    # Step 1 — register if needed
    if not user_id or not user_secret:
        print("[1/4] Registering SnapTrade user...")
        cmd_register(args)
        load_dotenv(ENV_PATH, override=True)
        user_id, user_secret = _get_user_credentials()
        if not user_id or not user_secret:
            print("ERROR: registration did not produce SNAPTRADE_USER_ID / SNAPTRADE_USER_SECRET")
            return
    else:
        print(f"[1/4] User already registered: {user_id}")

    # Step 2 — check existing connections; skip portal if Wealthsimple already linked
    print("[2/4] Checking existing brokerage connections...")
    has_ws = False
    try:
        conns_resp = snap.connections.list_brokerage_authorizations(
            user_id=user_id, user_secret=user_secret,
        )
        conns = _extract_response_body(conns_resp) or []
        for c in conns:
            brokerage = c.get("brokerage", {}) if isinstance(c, dict) else getattr(c, "brokerage", {})
            name = brokerage.get("name", "") if isinstance(brokerage, dict) else str(brokerage)
            if "wealthsimple" in str(name).lower():
                has_ws = True
                print(f"  Wealthsimple already connected ({name})")
                break
    except Exception as e:
        print(f"  Could not list connections (will attempt portal anyway): {e}")

    # Step 3 — generate portal URL, open browser, poll for new connection
    if not has_ws:
        print("[3/4] Generating connection portal...")
        try:
            login_resp = snap.authentication.login_snap_trade_user(
                user_id=user_id, user_secret=user_secret,
            )
            payload = _extract_response_body(login_resp)
            redirect_url = None
            if isinstance(payload, dict):
                redirect_url = payload.get("redirectURI") or payload.get("redirect_uri")
            else:
                redirect_url = getattr(payload, "redirectURI", None) or getattr(payload, "redirect_uri", None)
            if not redirect_url:
                print(f"ERROR: no redirect URL in response: {payload}")
                return
            print(f"  Opening portal in browser: {redirect_url}")
            try:
                webbrowser.open(redirect_url)
            except Exception:
                pass
            print("  >> Log into Wealthsimple in the browser, then return here.")
            print("  Polling every 5s for up to 5 minutes...")

            deadline = time.time() + 300
            while time.time() < deadline:
                time.sleep(5)
                try:
                    conns_resp = snap.connections.list_brokerage_authorizations(
                        user_id=user_id, user_secret=user_secret,
                    )
                    conns = _extract_response_body(conns_resp) or []
                    for c in conns:
                        brokerage = c.get("brokerage", {}) if isinstance(c, dict) else getattr(c, "brokerage", {})
                        name = brokerage.get("name", "") if isinstance(brokerage, dict) else str(brokerage)
                        if "wealthsimple" in str(name).lower():
                            has_ws = True
                            print(f"  Wealthsimple connected: {name}")
                            break
                    if has_ws:
                        break
                    sys.stdout.write("."); sys.stdout.flush()
                except Exception as e:
                    print(f"  poll error: {e}")
            print()
            if not has_ws:
                print("  Timeout waiting for Wealthsimple connection. Re-run --auto after completing the portal login.")
                return
        except Exception as e:
            print(f"ERROR: portal generation failed: {e}")
            return
    else:
        print("[3/4] Skipping portal (already connected)")

    # Step 4 — list accounts and trigger live refresh
    print("[4/4] Fetching accounts + triggering live balance refresh...")
    cmd_accounts(args)

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from monitoring.live_portfolio_refresh import refresh_portfolio_live
        result = refresh_portfolio_live()
        ws_meta = (result.get("_meta", {}).get("live_refresh", {}) or {}).get("wealthsimple", {})
        print(f"  refresh_portfolio_live → wealthsimple: {ws_meta.get('status', '?')}")
        if ws_meta.get("note"):
            print(f"    note: {ws_meta['note']}")
    except Exception as e:
        print(f"  live refresh skipped: {e}")

    print()
    print("Done. Dashboard should now show Wealthsimple positions/balance.")


def main():
    parser = argparse.ArgumentParser(description="SnapTrade Setup for Wealthsimple")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--auto", action="store_true",
                       help="Fully automated: register (if needed) -> open portal -> poll -> refresh live")
    group.add_argument("--register", action="store_true", help="Register SnapTrade user (one-time)")
    group.add_argument("--connect", action="store_true", help="Generate OAuth link for Wealthsimple")
    group.add_argument("--status", action="store_true", help="Check connection status")
    group.add_argument("--accounts", action="store_true", help="List linked accounts")
    args = parser.parse_args()

    if args.auto:
        cmd_auto(args)
    elif args.register:
        cmd_register(args)
    elif args.connect:
        cmd_connect(args)
    elif args.status:
        cmd_status(args)
    elif args.accounts:
        cmd_accounts(args)


if __name__ == "__main__":
    main()
