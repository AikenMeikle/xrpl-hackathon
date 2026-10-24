# Inspection report

## train (C:\Users\anass\OneDrive\Desktop\xrpl-hackathon\data\decode-and-detect-training-data\train)

- **n_transactions**: `40131`
- **n_transfers**: `60557`
- **duplicate_hashes**: `0`
- **time_range_utc**: `["2026-09-01 00:05:44+00:00", "2026-09-07 23:58:00+00:00"]`
- **span_days**: `6.99462962962963`
- **transaction_types**: `{"Payment": 40131}`
- **transaction_results**: `{"tesSUCCESS": 40131}`
- **n_failed**: `0`
- **legs_per_transaction**: `{"1": 23681, "2": 12474, "3": 3976}`
- **transfer_types**: `{"path_payment": 36876, "direct_payment": 23681}`
- **assets**: `{"0x1111222211112222111122221111222211112222:XRP": 20236, "r99vy4o5fXhC3PwEXEpnc1f9Mss4aCzYBW:GBP": 11156, "r99vy4o5fXhC3PwEXEpnc1f9Mss4aCzYBW:USD": 9867, "rLeMq5iFyEhgHLfBxeu4eRQWEGGdeL8hXK:EUR": 9824, "rLeMq5iFyEhgHLfBxeu4eRQWEGGdeL8hXK:USD": 9474}`
- **extra_field_key_sets**: `{"Amount ,DeliverMax ,Destination": 23681, "Amount ,DeliverMax ,Destination ,SendMax": 12474, "Amount ,DeliverMax ,Destination ,Paths ,SendMax": 3976}`
- **fee_drops**: `{"min": 10, "max": 15, "mean": 12.328424410057064}`
- **wallets**: `700`
- **senders**: `676`
- **economic_edges**: `40131`
- **conversions**: `16450`
- **paid_rate_missing**: `0`
- **delivered_rate_missing**: `0`
- **value_column_disagreements**: `0`
- **delivered_matches_last_leg**: `40131`
- **partial_payment_flags**: `0`
- **sendmax_present**: `16450`
- **destination_tag_present**: `0`
- **conversion_spread_quantiles**: `{"0.5": 0.003686365757197901, "0.9": 0.006567392361900832, "0.99": 0.010319153310203895, "1.0": 0.011892866432001355}`
- **paid_usd_quantiles**: `{"0.01": 2.7205929711167998, "0.5": 59.548893485026, "0.99": 1102.4648390684279, "1.0": 4575.674819394018}`
- **rates**: `{"rows": 850, "columns": ["TIMESTAMP", "ASSET", "PRICE_MICRO_USD"], "assets": ["XRP", "r99vy4o5fXhC3PwEXEpnc1f9Mss4aCzYBW:USD", "r99vy4o5fXhC3PwEXEpnc1f9Mss4aCzYBW:GBP", "rLeMq5iFyEhgHLfBxeu4eRQWEGGdeL8hXK:USD", "rLeMq5iFyEhgHLfBxeu4eRQWEGGdeL8hXK:EUR"], "hours": 170}`
- **max_same_timestamp**: `7`
- **ledgers**: `33388`
- **labels**: `{"columns": ["TRANSACTION_HASH", "is_suspicious", "family", "difficulty", "scenario_id", "cohort_id"], "rows": 40131, "positives": 1250, "prevalence": 0.0311479903316638, "family_by_difficulty": {"easy": {"aggregate_relay": 0, "ambiguous": 0, "background": 0, "bipartite": 113, "chain": 63, "cycle": 66, "distributed_relay": 0, "fan_in": 83, "fan_out": 66, "fan_out_in": 156, "multi_stage": 131, "random": 58, "stack": 64}, "hard": {"aggregate_relay": 102, "ambiguous": 25, "background": 0, "bipartite": 0, "chain": 0, "cycle": 0, "distributed_relay": 98, "fan_in": 0, "fan_out": 0, "fan_out_in": 0, "multi_stage": 0, "random": 0, "stack": 0}, "medium": {"aggregate_relay": 0, "ambiguous": 0, "background": 0, "bipartite": 21, "chain": 10, "cycle": 12, "distributed_relay": 0, "fan_in": 18, "fan_out": 26, "fan_out_in": 22, "multi_stage": 42, "random": 22, "stack": 52}, "normal": {"aggregate_relay": 10098, "ambiguous": 0, "background": 18000, "bipartite": 152, "chain": 66, "cycle": 113, "distributed_relay": 9702, "fan_in": 76, "fan_out": 75, "fan_out_in": 152, "multi_stage": 165, "random": 178, "stack": 104}}, "hash_coverage": 40131}`

## test (C:\Users\anass\OneDrive\Desktop\xrpl-hackathon\data\test-data)

- **n_transactions**: `40101`
- **n_transfers**: `65272`
- **duplicate_hashes**: `0`
- **time_range_utc**: `["2026-09-20 00:06:36+00:00", "2026-09-26 23:59:32+00:00"]`
- **span_days**: `6.995092592592592`
- **transaction_types**: `{"Payment": 40101}`
- **transaction_results**: `{"tesSUCCESS": 40101}`
- **n_failed**: `0`
- **legs_per_transaction**: `{"1": 19840, "2": 15351, "3": 4910}`
- **transfer_types**: `{"path_payment": 45432, "direct_payment": 19840}`
- **assets**: `{"0x1111222211112222111122221111222211112222:XRP": 23515, "rDafjsQScwoVYNW1Sg91oe1ALkvqvK2LTW:GBP": 10176, "rDafjsQScwoVYNW1Sg91oe1ALkvqvK2LTW:USD": 9996, "rNnBxvYbzhfnR6d8dLDRHRVVHgagj6aLVh:EUR": 10938, "rNnBxvYbzhfnR6d8dLDRHRVVHgagj6aLVh:USD": 10647}`
- **extra_field_key_sets**: `{"Amount ,DeliverMax ,Destination": 19840, "Amount ,DeliverMax ,Destination ,SendMax": 15351, "Amount ,DeliverMax ,Destination ,Paths ,SendMax": 4910}`
- **fee_drops**: `{"min": 10, "max": 15, "mean": 12.328620233909378}`
- **wallets**: `700`
- **senders**: `676`
- **economic_edges**: `40101`
- **conversions**: `20261`
- **paid_rate_missing**: `0`
- **delivered_rate_missing**: `0`
- **value_column_disagreements**: `0`
- **delivered_matches_last_leg**: `40101`
- **partial_payment_flags**: `0`
- **sendmax_present**: `20261`
- **destination_tag_present**: `0`
- **conversion_spread_quantiles**: `{"0.5": 0.003686366124981767, "0.9": 0.006663462485088319, "0.99": 0.010220830025990732, "1.0": 0.011794626957059202}`
- **paid_usd_quantiles**: `{"0.01": 1.9700977621959999, "0.5": 50.03687142384, "0.99": 1128.5339938982102, "1.0": 9592.973961827276}`
- **rates**: `{"rows": 850, "columns": ["TIMESTAMP", "ASSET", "PRICE_MICRO_USD"], "assets": ["XRP", "rDafjsQScwoVYNW1Sg91oe1ALkvqvK2LTW:USD", "rDafjsQScwoVYNW1Sg91oe1ALkvqvK2LTW:GBP", "rNnBxvYbzhfnR6d8dLDRHRVVHgagj6aLVh:USD", "rNnBxvYbzhfnR6d8dLDRHRVVHgagj6aLVh:EUR"], "hours": 170}`
- **max_same_timestamp**: `6`
- **ledgers**: `33321`

## cross split
- `{"shared_wallets": 0, "shared_hashes": 0}`
