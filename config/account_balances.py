"""
AAC Account Balances — Single source of truth for all account values.

All modules that need account balances should import from here instead of
hardcoding dollar amounts.  The canonical data lives in
``data/account_balances.json`` and is loaded lazily on first access.

Usage (read):
    from config.account_balances import Balances

    Balances.ibkr_total()        # 30148.51  (total portfolio USD)
    Balances.ibkr_cash()         # 185.56    (available cash USD)
    Balances.moomoo()            # 2609.26   (USD)
    Balances.wealthsimple()      # 18637.76  (CAD)
    Balances.ndax()              # 4492.04   (CAD)
    Balances.polymarket()        # 535.73    (USD)
    Balances.eq_bank()           # 100.0     (CAD)
    Balances.cad_usd()           # 0.70
    Balances.injection_total()   # 19800.0
    Balances.get("ibkr")         # full dict for ibkr account
    Balances.all_accounts()      # full accounts dict
    Balances.total_portfolio_usd()  # everything converted to USD

Usage (update):
    from config.account_balances import Balances

    Balances.set("ibkr", balance=200.0, total_assets=31000.0)
    Balances.set("moomoo", balance=12000.0, note="Post-injection")
    Balances.set_fx(cad_usd=0.71)
    Balances.save()  # writes back to JSON

CLI:
    python -m config.account_balances show
    python -m config.account_balances set ibkr --balance 200 --total-assets 31000
    python -m config.account_balances set moomoo --balance 12000
    python -m config.account_balances set-fx 0.71
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_BALANCES_FILE = Path(__file__).resolve().parent.parent / "data" / "account_balances.json"


class _BalanceStore:
    """Lazy-loading singleton for account balance data."""

    def __init__(self) -> None:
        self._data: Optional[Dict[str, Any]] = None

    def _load(self) -> Dict[str, Any]:
        if self._data is None:
            if _BALANCES_FILE.exists():
                with open(_BALANCES_FILE, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            else:
                self._data = {"_meta": {}, "accounts": {}, "injection": {}, "fx": {}}
        return self._data

    def _acct(self, key: str) -> Dict[str, Any]:
        return self._load().get("accounts", {}).get(key, {})

    # ── Read helpers ──────────────────────────────────────────────────

    def ibkr_cash(self) -> float:
        return float(self._acct("ibkr").get("balance", 0))

    def ibkr_total(self) -> float:
        return float(self._acct("ibkr").get("total_assets", 0))

    def moomoo(self) -> float:
        return float(self._acct("moomoo").get("balance", 0))

    def wealthsimple(self) -> float:
        return float(self._acct("wealthsimple").get("balance", 0))

    def ndax(self) -> float:
        return float(self._acct("ndax").get("balance", 0))

    def polymarket(self) -> float:
        return float(self._acct("polymarket").get("balance", 0))

    def eq_bank(self) -> float:
        return float(self._acct("eq_bank").get("balance", 0))

    def cad_usd(self) -> float:
        return float(self._load().get("fx", {}).get("cad_usd", 0.70))

    def injection_total(self) -> float:
        return float(self._load().get("injection", {}).get("total", 0))

    def injection_ws(self) -> float:
        return float(self._load().get("injection", {}).get("wealthsimple_cad", 0))

    def injection_moo(self) -> float:
        return float(self._load().get("injection", {}).get("moomoo_usd", 0))

    def get(self, key: str) -> Dict[str, Any]:
        return dict(self._acct(key))

    def all_accounts(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._load().get("accounts", {}))

    def total_portfolio_usd(self) -> float:
        """Sum all accounts converted to USD."""
        rate = self.cad_usd()
        total = 0.0
        for acct in self._load().get("accounts", {}).values():
            val = float(acct.get("total_assets", 0) or acct.get("balance", 0))
            if acct.get("currency") == "CAD":
                val *= rate
            total += val
        return round(total, 2)

    # ── Write helpers ─────────────────────────────────────────────────

    def set(self, key: str, **kwargs: Any) -> None:
        """Update fields on an account.  Only provided kwargs are changed."""
        data = self._load()
        acct = data.setdefault("accounts", {}).setdefault(key, {})
        for k, v in kwargs.items():
            acct[k] = v
        acct["verified"] = datetime.now().strftime("%Y-%m-%d")
        data["_meta"]["updated"] = datetime.now().isoformat(timespec="seconds")

    def set_fx(self, cad_usd: float) -> None:
        data = self._load()
        data.setdefault("fx", {})["cad_usd"] = round(cad_usd, 4)
        data["fx"]["updated"] = datetime.now().strftime("%Y-%m-%d")
        data["_meta"]["updated"] = datetime.now().isoformat(timespec="seconds")

    def set_injection(self, **kwargs: Any) -> None:
        data = self._load()
        inj = data.setdefault("injection", {})
        for k, v in kwargs.items():
            inj[k] = v
        data["_meta"]["updated"] = datetime.now().isoformat(timespec="seconds")

    def save(self) -> None:
        """Persist current state back to JSON."""
        if self._data is None:
            return
        _BALANCES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_BALANCES_FILE, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def reload(self) -> None:
        """Force reload from disk."""
        self._data = None

    # ── Auto-update from live scan ────────────────────────────────────

    def sync_from_scan(self, scan_results: Dict[str, Any]) -> Dict[str, str]:
        """Ingest results from _check_all_balances.scan_all() into central config.

        Maps scanner output fields → account_balances.json fields and saves.
        Returns a dict of {account: status_message} showing what was updated.
        """
        # Scanner key → config key mapping
        KEY_MAP = {
            "ibkr": "ibkr",
            "moomoo": "moomoo",
            "ndax": "ndax",
            "polymarket": "polymarket",
            "eqbank": "eq_bank",
            "wealthsimple": "wealthsimple",
        }

        report: Dict[str, str] = {}
        for scan_key, config_key in KEY_MAP.items():
            r = scan_results.get(scan_key)
            if not r or r.get("status") == "error":
                report[config_key] = f"SKIPPED ({r.get('error', 'no data') if r else 'missing'})"
                continue

            nl = r.get("net_liquidation", 0)
            if nl <= 0 and config_key not in ("eq_bank",):
                # Don't overwrite good data with zero from a failed connection
                report[config_key] = f"SKIPPED (net_liq={nl}, keeping existing)"
                continue

            currency = r.get("currency", "USD")
            balances = r.get("balances", {})

            updates: Dict[str, Any] = {
                "balance": nl,
                "total_assets": nl,
                "currency": currency,
            }

            # IBKR: separate cash from total if available
            if config_key == "ibkr":
                cash = balances.get(
                    "AVAILABLE_FUNDS",
                    balances.get("TOTAL_CASH", balances.get("CashBalance", balances.get("TotalCashBalance", nl))),
                )
                updates["balance"] = float(cash) if cash else nl
                updates["total_assets"] = nl

            # Track positions value
            positions = r.get("positions", [])
            if positions:
                pos_val = sum(
                    abs(float(p.get("market_val", p.get("marketValue", 0))))
                    for p in positions
                )
                updates["in_positions"] = round(pos_val, 2)
                updates["positions"] = positions

            if r.get("platform"):
                updates["platform"] = r["platform"]
            if r.get("account_id"):
                updates["account_id"] = r["account_id"]

            # Add note with scan timestamp
            scan_time = scan_results.get("_summary", {}).get("scan_time", "")
            updates["note"] = f"Auto-scan {scan_time[:19]} — {r.get('status', 'ok')}"

            self.set(config_key, **updates)
            report[config_key] = f"UPDATED ${nl:,.2f} {currency}"

        # Update FX rate from scanner if available
        summary = scan_results.get("_summary", {})
        if summary.get("cad_usd_rate"):
            self.set_fx(float(summary["cad_usd_rate"]))

        self.save()
        return report

    def auto_update(self, quiet: bool = False) -> Dict[str, str]:
        """Run live balance scan and persist results.

        This is the one-call auto-update: scans all connected exchanges
        and writes results to data/account_balances.json.

        Returns dict of {account: status_message}.
        """
        import asyncio

        try:
            from _check_all_balances import scan_all
        except ImportError:
            return {"error": "Cannot import _check_all_balances — run from project root"}

        if not quiet:
            print("  Scanning all accounts...")

        # Handle existing event loops (e.g. called from async context)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                results = pool.submit(lambda: asyncio.run(scan_all())).result()
        else:
            results = asyncio.run(scan_all())

        report = self.sync_from_scan(results)

        if not quiet:
            for acct, status in report.items():
                print(f"    {acct}: {status}")
            print(f"  Total portfolio: ${self.total_portfolio_usd():,.2f} USD")

        return report


# Module-level singleton
Balances = _BalanceStore()


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def _cli() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m config.account_balances",
        description="AAC Account Balance Manager — single source of truth",
    )
    sub = parser.add_subparsers(dest="cmd")

    # show
    show_p = sub.add_parser("show", help="Display all account balances")
    show_p.add_argument("--json", action="store_true", help="Raw JSON output")

    # set
    set_p = sub.add_parser("set", help="Update an account balance")
    set_p.add_argument("account", help="Account key (ibkr, moomoo, wealthsimple, ndax, polymarket, eq_bank)")
    set_p.add_argument("--balance", type=float, help="Available cash")
    set_p.add_argument("--total-assets", type=float, dest="total_assets", help="Total portfolio value")
    set_p.add_argument("--in-positions", type=float, dest="in_positions", help="Value in positions")
    set_p.add_argument("--note", type=str, help="Update note")

    # set-fx
    fx_p = sub.add_parser("set-fx", help="Update CAD/USD exchange rate")
    fx_p.add_argument("rate", type=float, help="CAD to USD rate (e.g. 0.70)")

    # set-injection
    inj_p = sub.add_parser("set-injection", help="Update injection plan")
    inj_p.add_argument("--total", type=float)
    inj_p.add_argument("--ws-cad", type=float, dest="wealthsimple_cad")
    inj_p.add_argument("--moo-usd", type=float, dest="moomoo_usd")
    inj_p.add_argument("--status", type=str)

    # scan — auto-update from live connections
    scan_p = sub.add_parser("scan", help="Scan all exchanges and auto-update balances")
    scan_p.add_argument("--quiet", action="store_true", help="Suppress scan output")

    args = parser.parse_args()

    if args.cmd == "show":
        data = Balances._load()
        if args.json:
            print(json.dumps(data, indent=2))
            return

        rate = Balances.cad_usd()
        print("=" * 68)
        print("  AAC ACCOUNT BALANCES")
        print(f"  Last updated: {data.get('_meta', {}).get('updated', 'unknown')}")
        print(f"  FX rate: 1 CAD = {rate} USD")
        print("=" * 68)

        for key, acct in data.get("accounts", {}).items():
            cur = acct.get("currency", "USD")
            bal = acct.get("balance", 0)
            total = acct.get("total_assets", bal)
            note = acct.get("note", "")
            verified = acct.get("verified", "?")
            usd_eq = total * rate if cur == "CAD" else total
            print(f"\n  {key.upper()}")
            print(f"    Cash:     ${bal:,.2f} {cur}")
            print(f"    Total:    ${total:,.2f} {cur}" + (f"  (~${usd_eq:,.2f} USD)" if cur == "CAD" else ""))
            print(f"    Verified: {verified}")
            if note:
                print(f"    Note:     {note}")

        print(f"\n  {'─' * 60}")
        print(f"  TOTAL PORTFOLIO: ${Balances.total_portfolio_usd():,.2f} USD")

        inj = data.get("injection", {})
        if inj:
            print(f"\n  INJECTION PLAN: ${inj.get('total', 0):,.0f} "
                  f"(WS ${inj.get('wealthsimple_cad', 0):,.0f} CAD + "
                  f"Moo ${inj.get('moomoo_usd', 0):,.0f} USD) "
                  f"[{inj.get('status', '?')}]")
        print()

    elif args.cmd == "set":
        updates = {}
        if args.balance is not None:
            updates["balance"] = args.balance
        if args.total_assets is not None:
            updates["total_assets"] = args.total_assets
        if args.in_positions is not None:
            updates["in_positions"] = args.in_positions
        if args.note is not None:
            updates["note"] = args.note
        if not updates:
            print("Nothing to update. Use --balance, --total-assets, --in-positions, or --note")
            sys.exit(1)
        Balances.set(args.account, **updates)
        Balances.save()
        print(f"Updated {args.account}: {updates}")

    elif args.cmd == "set-fx":
        Balances.set_fx(args.rate)
        Balances.save()
        print(f"FX rate updated: CAD/USD = {args.rate}")

    elif args.cmd == "set-injection":
        updates = {}
        for k in ("total", "wealthsimple_cad", "moomoo_usd", "status"):
            v = getattr(args, k, None)
            if v is not None:
                updates[k] = v
        if updates:
            Balances.set_injection(**updates)
            Balances.save()
            print(f"Injection updated: {updates}")
        else:
            print("Nothing to update.")

    elif args.cmd == "scan":
        Balances.auto_update(quiet=getattr(args, "quiet", False))

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
