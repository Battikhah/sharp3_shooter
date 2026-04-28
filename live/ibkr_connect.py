"""
live/ibkr_connect.py - IBKR TWS/Gateway connection helpers.

Pre-requisites before running:
  1. TWS or IB Gateway is open and logged into your paper account
  2. API access enabled: File → Global Configuration → API → Settings
       ☑ Enable ActiveX and Socket Clients
       ☑ Allow connections from localhost (127.0.0.1)
  3. Port matches config:
       7497 — TWS paper trading (default in config.py)
       4002 — IB Gateway paper trading
"""
import sys
import pandas as pd
from ib_insync import IB

from config import IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID


def connect(
    host: str = IBKR_HOST,
    port: int = IBKR_PORT,
    client_id: int = IBKR_CLIENT_ID,
    timeout: int = 10,
    readonly: bool = True,
) -> IB:
    """Connect to TWS / IB Gateway and return the live IB handle."""
    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=timeout, readonly=readonly)
    return ib


_ACCOUNT_TAGS = frozenset({
    "NetLiquidation",
    "TotalCashValue",
    "BuyingPower",
    "UnrealizedPnL",
    "RealizedPnL",
    "GrossPositionValue",
})


def get_account_summary(ib: IB) -> dict[str, float]:
    """Return key account metrics keyed by IBKR account tag."""
    result: dict[str, float] = {}
    for item in ib.accountSummary():
        if item.tag in _ACCOUNT_TAGS:
            try:
                result[item.tag] = float(item.value)
            except ValueError:
                pass
    return result


def get_positions(ib: IB) -> pd.DataFrame:
    """
    Return current open positions.
    Columns: ticker, sec_type, currency, position, avg_cost
    """
    rows = [
        {
            "ticker": pos.contract.symbol,
            "sec_type": pos.contract.secType,
            "currency": pos.contract.currency,
            "position": float(pos.position),
            "avg_cost": float(pos.avgCost),
        }
        for pos in ib.positions()
    ]
    cols = ["ticker", "sec_type", "currency", "position", "avg_cost"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


if __name__ == "__main__":
    print(f"Connecting to {IBKR_HOST}:{IBKR_PORT}  client_id={IBKR_CLIENT_ID} ...")

    try:
        ib = connect()
    except ConnectionRefusedError:
        print(
            "\nConnection refused. Checklist:\n"
            "  • TWS / IB Gateway is running and you are logged in\n"
            "  • API access is enabled (Global Config → API → Settings)\n"
            "  • Port: 7497 for TWS paper, 4002 for Gateway paper\n"
            "  • 127.0.0.1 is in the trusted IP list (or left blank)\n"
        )
        sys.exit(1)
    except TimeoutError:
        print("Timed out. Check host/port and that TWS is not in 'offline' mode.")
        sys.exit(1)

    print(f"Connected  client_id={ib.client.clientId}\n")

    summary = get_account_summary(ib)
    print("Account Summary:")
    for tag, value in sorted(summary.items()):
        print(f"  {tag:<25} {value:>15,.2f}")

    positions = get_positions(ib)
    print(f"\nOpen Positions ({len(positions)} rows):")
    if positions.empty:
        print("  (none — clean paper account)")
    else:
        print(positions.to_string(index=False))

    # --- Assertions ---
    assert "NetLiquidation" in summary, "NetLiquidation missing from account summary"
    assert summary["NetLiquidation"] > 0, "NetLiquidation must be positive"
    assert set(positions.columns) == {"ticker", "sec_type", "currency", "position", "avg_cost"}, (
        f"Unexpected columns: {positions.columns.tolist()}"
    )

    print("\nAll assertions  PASS")
    ib.disconnect()
    print("Disconnected.")
