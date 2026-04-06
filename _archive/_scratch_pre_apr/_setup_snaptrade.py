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
import os
import sys
import argparse
import uuid
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv, set_key
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")


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
            brokerage = getattr(conn, "brokerage", {})
            name = brokerage.get("name", "Unknown") if isinstance(brokerage, dict) else str(brokerage)
            status = getattr(conn, "status", "unknown")
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
            acct_id = getattr(acct, 'id', '?')
            name = getattr(acct, 'name', str(acct_id))
            # Get balance for each account
            try:
                bal = snap.account_information.get_user_account_balance(
                    user_id=user_id, user_secret=user_secret, account_id=str(acct_id)
                )
                for b in (bal.body or []):
                    currency = getattr(b, 'currency', 'CAD')
                    total = getattr(b, 'total', 0)
                    print(f"  {name} [{acct_id}]: {currency} {total:,.2f}")
            except Exception:
                print(f"  {name} [{acct_id}]: (could not fetch balance)")

    except Exception as e:
        print(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="SnapTrade Setup for Wealthsimple")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--register", action="store_true", help="Register SnapTrade user (one-time)")
    group.add_argument("--connect", action="store_true", help="Generate OAuth link for Wealthsimple")
    group.add_argument("--status", action="store_true", help="Check connection status")
    group.add_argument("--accounts", action="store_true", help="List linked accounts")
    args = parser.parse_args()

    if args.register:
        cmd_register(args)
    elif args.connect:
        cmd_connect(args)
    elif args.status:
        cmd_status(args)
    elif args.accounts:
        cmd_accounts(args)


if __name__ == "__main__":
    main()
