"""XRPL semantics contract tests on tiny synthetic tables."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import XRP_ASSET_ID, XRP_SENTINEL
from src.normalize import (
    asset_id,
    drops_to_xrp,
    economic_edges,
    load_rate_index,
    normalize,
    parse_amount,
    price_usd,
    quantity_from_source_value,
    rate_hour,
)

ISSUER_A = "rIssuerAAAAAAAAAAAAAAAAAAAAAAAAAA"
ISSUER_B = "rIssuerBBBBBBBBBBBBBBBBBBBBBBBBBB"
T0 = 1_788_220_800  # an exact UTC hour boundary


def _rates(rows):
    return pd.DataFrame(rows, columns=["TIMESTAMP", "ASSET", "PRICE_MICRO_USD"])


def _tx(h, ts, account, dest, delivered, fee=12, result="tesSUCCESS", success=True, extra=None, ledger=1, idx=0):
    extra = dict(extra or {})
    extra.setdefault("Destination", dest)
    extra.setdefault("Amount", delivered)
    extra.setdefault("DeliverMax", delivered)
    return dict(
        TRANSACTION_HASH=h, TIMESTAMP=ts, LEDGER_INDEX=ledger, TRANSACTION_INDEX=idx, ACCOUNT=account,
        TRANSACTION_TYPE="Payment", SOURCE_FEE=fee, FEE=fee / 1e6, TRANSACTION_RESULT=result, SUCCESS=success,
        DELIVERED_AMOUNT=json.dumps(delivered), EXTRA_TRANSACTION_FIELDS=json.dumps(extra),
    )


def _leg(h, i, ts, frm, to, token, sym, source_value, ttype="direct_payment", ledger=1, idx=0):
    value = source_value / 1e6 if sym == "XRP" else source_value
    return dict(
        TRANSACTION_HASH=h, TRANSFER_INDEX=i, TIMESTAMP=ts, LEDGER_INDEX=ledger, TRANSACTION_INDEX=idx,
        FROM_ADDRESS=frm, TO_ADDRESS=to, TOKEN_ADDRESS=token, SYMBOL=sym, SOURCE_VALUE=float(source_value),
        VALUE=float(value), TRANSFER_TYPE=ttype,
    )


def test_drops_string_is_250_xrp():
    assert drops_to_xrp("250000000") == 250.0
    assert parse_amount('"250000000"') == (XRP_ASSET_ID, 250.0)
    assert quantity_from_source_value(250_000_000, XRP_SENTINEL, "XRP") == 250.0


def test_tokens_are_not_divided_by_a_million():
    assert quantity_from_source_value(12.34, ISSUER_A, "USD") == 12.34
    asset, units = parse_amount(json.dumps({"currency": "USD", "issuer": ISSUER_A, "value": "12.34"}))
    assert asset == f"{ISSUER_A}:USD" and units == 12.34


def test_two_usd_issuers_are_distinct_assets():
    assert asset_id(ISSUER_A, "USD") != asset_id(ISSUER_B, "USD")
    rates = load_rate_index(_rates([(T0, f"{ISSUER_A}:USD", 1_000_000), (T0, f"{ISSUER_B}:USD", 900_000)]))
    assert price_usd(rates, T0 + 5, f"{ISSUER_A}:USD") == 1.0
    assert price_usd(rates, T0 + 5, f"{ISSUER_B}:USD") == 0.9


def test_hour_boundary_and_missing_rate():
    rates = load_rate_index(_rates([(T0, "XRP", 500_000), (T0 + 3600, "XRP", 600_000)]))
    assert rate_hour(T0 + 3599) == T0
    assert price_usd(rates, T0 + 3599, XRP_ASSET_ID) == 0.5
    assert price_usd(rates, T0 + 3600, XRP_ASSET_ID) == 0.6
    # before the first print: no backward/forward fill, stays NaN (never zero, never future)
    assert np.isnan(price_usd(rates, T0 - 1, XRP_ASSET_ID))
    assert np.isnan(price_usd(rates, T0 + 2 * 3600, XRP_ASSET_ID))


def test_fee_is_burned_not_a_recipient_and_three_legs_make_one_edge():
    tx = pd.DataFrame([
        _tx("H1", T0 + 10, "rAlice", "rBob", {"currency": "EUR", "issuer": ISSUER_B, "value": "90"}, fee=15,
            extra={"SendMax": "120000000", "Paths": []}),
    ])
    legs = pd.DataFrame([
        _leg("H1", 0, T0 + 10, "rAlice", "rLP1", XRP_SENTINEL, "XRP", 100_000_000, "path_payment"),
        _leg("H1", 1, T0 + 10, "rLP1", "rLP2", ISSUER_A, "USD", 50.0, "path_payment"),
        _leg("H1", 2, T0 + 10, "rLP2", "rBob", ISSUER_B, "EUR", 90.0, "path_payment"),
    ])
    rates = _rates([(T0, "XRP", 500_000), (T0, f"{ISSUER_A}:USD", 1_000_000), (T0, f"{ISSUER_B}:EUR", 550_000)])
    canonical, audit = normalize(tx, legs, rates)
    edges = economic_edges(canonical)
    assert len(edges) == 1
    e = edges.iloc[0]
    assert e.source_wallet == "rAlice" and e.destination_wallet == "rBob"
    assert e.transfer_leg_count == 3 and e.conversion_flag
    assert e.paid_asset_id == XRP_ASSET_ID and e.paid_quantity == 100.0 and e.paid_usd == 50.0
    assert e.delivered_asset_id == f"{ISSUER_B}:EUR" and e.delivered_quantity == 90.0
    assert abs(e.delivered_usd - 49.5) < 1e-9
    assert e.sendmax_present and e.sendmax_quantity == 120.0  # ceiling kept, not used as paid
    assert e.fee_drops == 15 and abs(e.fee_xrp - 0.000015) < 1e-12
    # liquidity providers never appear as economic endpoints
    assert "rLP1" not in set(edges.source_wallet) | set(edges.destination_wallet)
    assert len(audit) == 3  # but every leg is retained for audit


def test_partial_payment_uses_delivered_amount():
    tx = pd.DataFrame([
        _tx("H2", T0 + 10, "rAlice", "rBob", "80000000", extra={"Amount": "100000000", "DeliverMax": "100000000"}),
    ])
    legs = pd.DataFrame([_leg("H2", 0, T0 + 10, "rAlice", "rBob", XRP_SENTINEL, "XRP", 80_000_000)])
    canonical, _ = normalize(tx, legs, _rates([(T0, "XRP", 500_000)]))
    e = canonical.iloc[0]
    assert e.delivered_quantity == 80.0 and e.ceiling_quantity == 100.0
    assert e.partial_payment_flag


def test_failed_payment_has_row_but_no_edge():
    tx = pd.DataFrame([
        _tx("H3", T0 + 10, "rAlice", "rBob", "1000000", fee=12, result="tecPATH_DRY", success=False),
        _tx("H4", T0 + 20, "rAlice", "rBob", "1000000", ledger=2),
    ])
    legs = pd.DataFrame([_leg("H4", 0, T0 + 20, "rAlice", "rBob", XRP_SENTINEL, "XRP", 1_000_000, ledger=2)])
    canonical, _ = normalize(tx, legs, _rates([(T0, "XRP", 500_000)]))
    assert len(canonical) == 2
    assert set(canonical.transaction_hash) == {"H3", "H4"}
    failed = canonical[canonical.transaction_hash == "H3"].iloc[0]
    assert not failed.in_economic_graph and failed.fee_drops == 12
    assert len(economic_edges(canonical)) == 1


def test_ledger_order_not_hash_order():
    tx = pd.DataFrame([
        _tx("ZZZ", T0, "rA", "rB", "1000000", ledger=5, idx=0),
        _tx("AAA", T0, "rB", "rC", "1000000", ledger=5, idx=1),
    ])
    legs = pd.DataFrame([
        _leg("ZZZ", 0, T0, "rA", "rB", XRP_SENTINEL, "XRP", 1_000_000, ledger=5, idx=0),
        _leg("AAA", 0, T0, "rB", "rC", XRP_SENTINEL, "XRP", 1_000_000, ledger=5, idx=1),
    ])
    canonical, _ = normalize(tx, legs, _rates([(T0, "XRP", 500_000)]))
    assert canonical.transaction_hash.tolist() == ["ZZZ", "AAA"]
