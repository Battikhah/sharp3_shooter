"""
live/execute.py — Order submission with safeguards.

Safeguards enforced on every submit_orders call:
  1. Signal flip:    |signal − prev_signal| > FLIP_THRESHOLD → block
  2. Position cap:   |target_$| > MAX_POSITION_PCT × NAV     → block
  3. Turnover cap:   cumulative |Δ$| > MAX_TURNOVER_PCT × NAV → block remaining

All decisions (approved + blocked) are appended to results/orders.log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ib_insync import IB, Stock
from ib_insync import Order as IBKROrder

LOG_PATH = Path("results/orders.log")
MAX_POSITION_PCT: float = 0.20
MAX_TURNOVER_PCT: float = 2.00
FLIP_THRESHOLD: float = 1.50


@dataclass(frozen=True)
class Order:
    ticker: str
    delta_shares: float
    current_shares: float
    signal: float
    prev_signal: float | None
    price: float


def _setup_logger() -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    root = logging.getLogger()
    already_attached = any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(LOG_PATH.resolve())
        for h in root.handlers
    )
    if not already_attached:
        fh = logging.FileHandler(LOG_PATH)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        root.addHandler(fh)


def _approved(orders: list[Order], nav: float) -> list[tuple[Order, str]]:
    result: list[tuple[Order, str]] = []
    cumulative_turnover = 0.0

    for o in orders:
        delta_dollars = abs(o.delta_shares) * o.price
        target_dollars = (o.current_shares + o.delta_shares) * o.price

        if o.prev_signal is not None:
            flip = abs(o.signal - o.prev_signal)
            if flip > FLIP_THRESHOLD:
                logging.warning(
                    "BLOCK %-6s signal flip %.3f → %.3f (Δ=%.3f > threshold=%.2f)",
                    o.ticker, o.prev_signal, o.signal, flip, FLIP_THRESHOLD,
                )
                continue

        if abs(target_dollars) > MAX_POSITION_PCT * nav:
            logging.warning(
                "BLOCK %-6s target $%,.0f exceeds %.0f%% NAV ($%,.0f)",
                o.ticker, abs(target_dollars), MAX_POSITION_PCT * 100, MAX_POSITION_PCT * nav,
            )
            continue

        if cumulative_turnover + delta_dollars > MAX_TURNOVER_PCT * nav:
            logging.warning(
                "BLOCK %-6s would push turnover to $%,.0f > %.0f%% NAV ($%,.0f)",
                o.ticker, cumulative_turnover + delta_dollars,
                MAX_TURNOVER_PCT * 100, MAX_TURNOVER_PCT * nav,
            )
            continue

        cumulative_turnover += delta_dollars
        reason = f"signal={o.signal:+.4f} delta={o.delta_shares:+.0f}sh ${delta_dollars:,.0f}"
        result.append((o, reason))

    return result


def _moo_order(action: str, qty: int) -> IBKROrder:
    o = IBKROrder()
    o.action = action
    o.totalQuantity = qty
    o.orderType = "MKT"
    o.tif = "OPG"
    return o


def submit_orders(
    ib: IB | None,
    orders: list[Order],
    nav: float,
    dry_run: bool = False,
) -> None:
    """Filter orders through safeguards then submit (or dry-print). ib may be None when dry_run=True."""
    _setup_logger()

    if not orders:
        logging.info("submit_orders: no orders provided")
        return

    approved = _approved(orders, nav)
    logging.info(
        "submit_orders: %d/%d approved  (NAV=$%,.0f)",
        len(approved), len(orders), nav,
    )

    for o, reason in approved:
        qty = abs(round(o.delta_shares))
        if qty < 1:
            logging.info("SKIP   %-6s delta rounds to 0 shares", o.ticker)
            continue

        action = "BUY" if o.delta_shares > 0 else "SELL"

        if dry_run:
            logging.info("DRY    %-6s %s %d sh  [%s]", o.ticker, action, qty, reason)
        else:
            assert ib is not None, "ib handle required for live submission"
            contract = Stock(o.ticker, "SMART", "USD")
            ibkr_order = _moo_order(action, qty)
            trade = ib.placeOrder(contract, ibkr_order)
            logging.info(
                "ORDER  %-6s %s %d sh  orderId=%s  [%s]",
                o.ticker, action, qty, trade.order.orderId, reason,
            )
