#!/usr/bin/env python3
"""
Trade Reporting & Regulatory Compliance System
==============================================
Automated trade logging, regulatory reporting, and trade reconstruction for compliance.
"""

import asyncio
import logging
import json
import csv
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from decimal import Decimal
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.audit_logger import get_audit_logger


@dataclass
class TradeRecord:
    """Complete trade record for regulatory reporting"""
    trade_id: str
    timestamp: datetime
    strategy_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: Decimal
    price: Decimal
    total_value: Decimal
    commission: Decimal
    exchange: str
    order_type: str
    execution_time: datetime
    account_id: str
    counterparty: Optional[str] = None
    regulatory_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegulatoryReport:
    """Regulatory report for submission"""
    report_id: str
    report_type: str  # "FINRA_TRF", "SEC_BD", "CFTC_CPR"
    reporting_period: str
    generated_at: datetime
    total_trades: int
    total_volume: Decimal
    report_data: List[Dict[str, Any]]
    submission_status: str = "pending"


class TradeReportingSystem:
    """Automated trade reporting and regulatory compliance"""

    def __init__(self):
        self.logger = logging.getLogger("TradeReporting")
        self.audit_logger = get_audit_logger()

        # Trade storage
        self.trade_records: List[TradeRecord] = []
        self.pending_reports: List[RegulatoryReport] = []

        # Reporting schedules
        self.reporting_schedules = {
            "FINRA_TRF": {"frequency": "daily", "deadline": "08:00"},  # Daily by 8 AM
            "SEC_BD": {"frequency": "daily", "deadline": "17:00"},     # Daily by 5 PM
            "CFTC_CPR": {"frequency": "weekly", "deadline": "monday_09:00"}  # Weekly Monday 9 AM
        }

        # Report storage paths
        self.reports_dir = PROJECT_ROOT / "reports" / "regulatory"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Load existing trade records
        self._load_trade_records()

    def _load_trade_records(self):
        """Load existing trade records from storage"""
        trades_file = self.reports_dir / "trade_records.json"

        if trades_file.exists():
            try:
                with open(trades_file, 'r') as f:
                    data = json.load(f)

                for record_data in data:
                    record = TradeRecord(
                        trade_id=record_data["trade_id"],
                        timestamp=datetime.fromisoformat(record_data["timestamp"]),
                        strategy_id=record_data["strategy_id"],
                        symbol=record_data["symbol"],
                        side=record_data["side"],
                        quantity=Decimal(str(record_data["quantity"])),
                        price=Decimal(str(record_data["price"])),
                        total_value=Decimal(str(record_data["total_value"])),
                        commission=Decimal(str(record_data["commission"])),
                        exchange=record_data["exchange"],
                        order_type=record_data["order_type"],
                        execution_time=datetime.fromisoformat(record_data["execution_time"]),
                        account_id=record_data["account_id"],
                        counterparty=record_data.get("counterparty"),
                        regulatory_flags=record_data.get("regulatory_flags", []),
                        metadata=record_data.get("metadata", {})
                    )
                    self.trade_records.append(record)

                self.logger.info(f"Loaded {len(self.trade_records)} existing trade records")

            except Exception as e:
                self.logger.error(f"Error loading trade records: {e}")

    async def record_trade(self,
                          strategy_id: str,
                          symbol: str,
                          side: str,
                          quantity: Decimal,
                          price: Decimal,
                          exchange: str,
                          order_type: str = "market",
                          account_id: str = "default",
                          counterparty: Optional[str] = None,
                          commission: Decimal = Decimal("0"),
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Record a completed trade for regulatory compliance"""

        trade_id = str(uuid.uuid4())
        timestamp = datetime.now()
        total_value = quantity * price

        # Determine regulatory flags
        regulatory_flags = []
        if total_value >= Decimal("10000"):  # Large trade reporting
            regulatory_flags.append("large_trade")
        if exchange in ["NYSE", "NASDAQ"]:
            regulatory_flags.append("equity_trade")
        if "crypto" in exchange.lower():
            regulatory_flags.append("crypto_trade")

        trade_record = TradeRecord(
            trade_id=trade_id,
            timestamp=timestamp,
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            total_value=total_value,
            commission=commission,
            exchange=exchange,
            order_type=order_type,
            execution_time=timestamp,
            account_id=account_id,
            counterparty=counterparty,
            regulatory_flags=regulatory_flags,
            metadata=metadata or {}
        )

        self.trade_records.append(trade_record)

        # Save to persistent storage
        await self._save_trade_records()

        # Audit the trade
        await self.audit_logger.log_event(
            category="trading",
            action="trade_recorded",
            details={
                "trade_id": trade_id,
                "strategy_id": strategy_id,
                "symbol": symbol,
                "side": side,
                "quantity": float(quantity),
                "price": float(price),
                "total_value": float(total_value),
                "exchange": exchange,
                "regulatory_flags": regulatory_flags
            }
        )

        self.logger.info(f"Trade recorded: {trade_id} - {side} {quantity} {symbol} @ ${price} on {exchange}")

        return trade_id

    async def _save_trade_records(self):
        """Save trade records to persistent storage"""
        trades_file = self.reports_dir / "trade_records.json"

        # Convert to serializable format
        records_data = []
        for record in self.trade_records[-1000:]:  # Keep last 1000 trades
            record_data = {
                "trade_id": record.trade_id,
                "timestamp": record.timestamp.isoformat(),
                "strategy_id": record.strategy_id,
                "symbol": record.symbol,
                "side": record.side,
                "quantity": float(record.quantity),
                "price": float(record.price),
                "total_value": float(record.total_value),
                "commission": float(record.commission),
                "exchange": record.exchange,
                "order_type": record.order_type,
                "execution_time": record.execution_time.isoformat(),
                "account_id": record.account_id,
                "counterparty": record.counterparty,
                "regulatory_flags": record.regulatory_flags,
                "metadata": record.metadata
            }
            records_data.append(record_data)

        try:
            with open(trades_file, 'w') as f:
                json.dump(records_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving trade records: {e}")

    async def generate_finra_trf_report(self, report_date: Optional[datetime] = None) -> RegulatoryReport:
        """Generate FINRA TRF (Trade Reporting Facility) report"""
        if report_date is None:
            report_date = datetime.now() - timedelta(days=1)  # Previous day

        # Get trades for the reporting period
        start_date = report_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)

        period_trades = [
            trade for trade in self.trade_records
            if start_date <= trade.timestamp < end_date
        ]

        # Format for FINRA TRF
        report_data = []
        total_volume = Decimal("0")

        for trade in period_trades:
            if "equity_trade" in trade.regulatory_flags or trade.total_value >= Decimal("10000"):
                trade_entry = {
                    "trade_id": trade.trade_id,
                    "timestamp": trade.timestamp.strftime("%Y%m%d%H%M%S"),
                    "symbol": trade.symbol,
                    "side": trade.side.upper(),
                    "quantity": int(trade.quantity),
                    "price": float(trade.price),
                    "total_value": float(trade.total_value),
                    "commission": float(trade.commission),
                    "exchange": trade.exchange,
                    "account_id": trade.account_id,
                    "strategy_id": trade.strategy_id
                }
                report_data.append(trade_entry)
                total_volume += trade.total_value

        report = RegulatoryReport(
            report_id=f"FINRA_TRF_{report_date.strftime('%Y%m%d')}",
            report_type="FINRA_TRF",
            reporting_period=report_date.strftime("%Y-%m-%d"),
            generated_at=datetime.now(),
            total_trades=len(report_data),
            total_volume=total_volume,
            report_data=report_data
        )

        self.pending_reports.append(report)

        # Save report to file
        await self._save_regulatory_report(report)

        self.logger.info(f"Generated FINRA TRF report: {report.report_id} with {len(report_data)} trades")

        return report

    async def generate_sec_bd_report(self, report_date: Optional[datetime] = None) -> RegulatoryReport:
        """Generate SEC Broker-Dealer report"""
        if report_date is None:
            report_date = datetime.now() - timedelta(days=1)

        start_date = report_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)

        period_trades = [
            trade for trade in self.trade_records
            if start_date <= trade.timestamp < end_date
        ]

        # SEC BD format
        report_data = []
        total_volume = Decimal("0")

        for trade in period_trades:
            trade_entry = {
                "date": trade.timestamp.strftime("%Y%m%d"),
                "time": trade.timestamp.strftime("%H%M%S"),
                "symbol": trade.symbol,
                "buy_sell": "B" if trade.side == "buy" else "S",
                "quantity": int(trade.quantity),
                "price": f"{trade.price:.4f}",
                "principal": f"{trade.total_value:.2f}",
                "commission": f"{trade.commission:.2f}",
                "exchange": trade.exchange,
                "account": trade.account_id
            }
            report_data.append(trade_entry)
            total_volume += trade.total_value

        report = RegulatoryReport(
            report_id=f"SEC_BD_{report_date.strftime('%Y%m%d')}",
            report_type="SEC_BD",
            reporting_period=report_date.strftime("%Y-%m-%d"),
            generated_at=datetime.now(),
            total_trades=len(report_data),
            total_volume=total_volume,
            report_data=report_data
        )

        self.pending_reports.append(report)
        await self._save_regulatory_report(report)

        self.logger.info(f"Generated SEC BD report: {report.report_id} with {len(report_data)} trades")

        return report

    async def _save_regulatory_report(self, report: RegulatoryReport):
        """Save regulatory report to file"""
        report_file = self.reports_dir / f"{report.report_id}.json"

        report_data = {
            "report_id": report.report_id,
            "report_type": report.report_type,
            "reporting_period": report.reporting_period,
            "generated_at": report.generated_at.isoformat(),
            "total_trades": report.total_trades,
            "total_volume": float(report.total_volume),
            "submission_status": report.submission_status,
            "report_data": report.report_data
        }

        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving regulatory report: {e}")

    async def submit_regulatory_report(self, report_id: str) -> bool:
        """Submit regulatory report (simulated)"""
        report = next((r for r in self.pending_reports if r.report_id == report_id), None)

        if not report:
            self.logger.error(f"Report not found: {report_id}")
            return False

        # Simulate submission
        self.logger.info(f"Submitting {report.report_type} report: {report_id}")

        # In real implementation, this would submit to regulatory APIs
        report.submission_status = "submitted"

        # Audit the submission
        await self.audit_logger.log_event(
            category="compliance",
            action="regulatory_report_submitted",
            details={
                "report_id": report_id,
                "report_type": report.report_type,
                "total_trades": report.total_trades,
                "total_volume": float(report.total_volume)
            }
        )

        return True

    def get_trade_reconstruction(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get complete trade reconstruction data for audit purposes"""
        trade = next((t for t in self.trade_records if t.trade_id == trade_id), None)

        if not trade:
            return None

        return {
            "trade_id": trade.trade_id,
            "timestamp": trade.timestamp.isoformat(),
            "strategy_id": trade.strategy_id,
            "symbol": trade.symbol,
            "side": trade.side,
            "quantity": float(trade.quantity),
            "price": float(trade.price),
            "total_value": float(trade.total_value),
            "commission": float(trade.commission),
            "exchange": trade.exchange,
            "order_type": trade.order_type,
            "execution_time": trade.execution_time.isoformat(),
            "account_id": trade.account_id,
            "counterparty": trade.counterparty,
            "regulatory_flags": trade.regulatory_flags,
            "metadata": trade.metadata,
            "audit_trail": self._get_trade_audit_trail(trade_id)
        }

    def _get_trade_audit_trail(self, trade_id: str) -> List[Dict[str, Any]]:
        """Get audit trail for a specific trade"""
        # In a real implementation, this would query the audit log
        # For now, return basic reconstruction data
        return [
            {
                "event": "trade_initiated",
                "timestamp": datetime.now().isoformat(),
                "details": f"Trade {trade_id} initiated by strategy"
            },
            {
                "event": "order_submitted",
                "timestamp": datetime.now().isoformat(),
                "details": f"Order for trade {trade_id} submitted to exchange"
            },
            {
                "event": "trade_executed",
                "timestamp": datetime.now().isoformat(),
                "details": f"Trade {trade_id} executed successfully"
            }
        ]

    async def run_daily_reporting(self):
        """Run daily regulatory reporting"""
        self.logger.info("Starting daily regulatory reporting...")

        report_date = datetime.now() - timedelta(days=1)

        # Generate FINRA TRF report
        finra_report = await self.generate_finra_trf_report(report_date)

        # Generate SEC BD report
        sec_report = await self.generate_sec_bd_report(report_date)

        # Submit reports
        await self.submit_regulatory_report(finra_report.report_id)
        await self.submit_regulatory_report(sec_report.report_id)

        self.logger.info("Daily regulatory reporting completed")

    def get_reporting_status(self) -> Dict[str, Any]:
        """Get current reporting status"""
        today = datetime.now().date()

        # Check if today's reports have been generated
        finra_today = any(r.report_type == "FINRA_TRF" and r.reporting_period == today.isoformat()
                         for r in self.pending_reports)
        sec_today = any(r.report_type == "SEC_BD" and r.reporting_period == today.isoformat()
                       for r in self.pending_reports)

        return {
            "total_trades_recorded": len(self.trade_records),
            "pending_reports": len(self.pending_reports),
            "today_reports_generated": {
                "FINRA_TRF": finra_today,
                "SEC_BD": sec_today
            },
            "last_report_date": max((r.generated_at.date() for r in self.pending_reports), default=None)
        }


# Global trade reporting system instance
trade_reporting_system = TradeReportingSystem()


async def initialize_trade_reporting():
    """Initialize the trade reporting system"""
    print("[TRADING] Initializing Trade Reporting System...")

    # Run daily reporting check
    await trade_reporting_system.run_daily_reporting()

    status = trade_reporting_system.get_reporting_status()

    print("[OK] Trade reporting system initialized")
    print(f"  Total Trades Recorded: {status['total_trades_recorded']}")
    print(f"  Pending Reports: {status['pending_reports']}")
    print(f"  Today's Reports: FINRA={status['today_reports_generated']['FINRA_TRF']}, SEC={status['today_reports_generated']['SEC_BD']}")

    return True


if __name__ == "__main__":
    asyncio.run(initialize_trade_reporting())