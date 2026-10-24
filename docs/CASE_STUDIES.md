# Case studies (synthetic ledger; every item is 'flagged for review', never an accusation)

All wallets and hashes are organiser simulation identifiers. Scores are rankings, not probabilities of guilt.

## Test alert rank 166 (dominant shape: irregular_complexity)

- hash `77BEEAE7C370E5D07471C02D42E80AB07F346BC027500EFD2BADC77A3921BF05` at 2026-09-20 02:15:56+00:00, score 0.9604
- rNjNitRFEqNuHcva8UtBM9v7isRQfKBxY -> rHhdL3bhTiQvxUnCMpKFZPcGDxW1h7PXmR: paid 24.9658 USD (~$24.90), delivered 24.9658 USD (~$24.90); legs=1 conversion=False
- top SHAP drivers: `full_edge_share_of_pair_usd`=1 (+1.24), `sg_branch_holdtime_median_h`=0.0328 (+1.21), `fanout_equal_size_frac_10m`=0.167 (+1.00), `win_src_out_n_10m`=6 (+0.92), `fwd_dst_best_amount_gap_10m`=0.15 (+0.85)
- motif evidence: fan_in=0.497, fan_out=6, scatter_gather_middlemen=6, gather_scatter_score_6h=0.806, chain_longest_hops=3, bipartite_rectangles=23, irregular_complexity=11.4, forward_match_q=0.497, backward_match_q=7.8e-05
- witnesses: {"forward_match": ["77BEEAE7C370E5D07471C02D42E80AB07F346BC027500EFD2BADC77A3921BF05", "CA0CF495F1CF4076C70C61395F3DAFC692457F78FED3630ADB0CA07B33D6E870"], "backward_match": ["2D082D255821E94B1E94A40A0939B33FED06090483BD6BBC5EE70AEE469EC42F", "77BEEAE7C370E5D07471C02D42E80AB07F346BC027500EFD2BADC77A3921BF05"], "scatter_gather": ["77BEEAE7C370E5D07471C02D42E80AB07F346BC027500EFD2BADC77A3921BF05", "5AB3FA4E25C2432669506E2D36979940964C540FF0739218280115B4D442B089", "31A22B364EEB079100E9B387044CCAA92CF66BD9A6003DA3B7B70FF764745889", "13B1E6BC62BB6ED21C2D898D56918A96A9B07651714A66A431F43F1D62C3DA89

## Test alert rank 42 (dominant shape: bipartite_rectangles)

- hash `645A8633AEA776029A452207531306E7326C149BAF36AFBADF1314E6B6C76493` at 2026-09-20 02:18:16+00:00, score 0.9947
- rpTYD3LBbnVfZJB7xjj6PJCekWeqqc7quN -> rUYRMFCQZJAVvb9hT5KLDtoCFNJN3jyYCo: paid 21.5544 USD (~$21.50), delivered 21.5544 USD (~$21.50); legs=1 conversion=False
- top SHAP drivers: `full_edge_share_of_pair_usd`=1 (+1.73), `sg_branch_holdtime_median_h`=0.0328 (+1.54), `win_dst_in_deg_10m`=6 (+1.53), `win_dst_in_n_10m`=6 (+1.18), `asof_src_best_amount_gap_10m`=0.13 (+0.72)
- motif evidence: fan_out=1, scatter_gather_middlemen=6, gather_scatter_score_6h=3.24, chain_longest_hops=3, bipartite_rectangles=48, irregular_complexity=10.5, forward_match_q=0.943, backward_match_q=0.497
- witnesses: {"forward_match": ["645A8633AEA776029A452207531306E7326C149BAF36AFBADF1314E6B6C76493", "727DCBB8304A06EEDE4CCD65C8EB6FEDF0044DF0A623ECF643C8FEBFAB89D888"], "backward_match": ["5AB3FA4E25C2432669506E2D36979940964C540FF0739218280115B4D442B089", "645A8633AEA776029A452207531306E7326C149BAF36AFBADF1314E6B6C76493"], "scatter_gather": ["77BEEAE7C370E5D07471C02D42E80AB07F346BC027500EFD2BADC77A3921BF05", "5AB3FA4E25C2432669506E2D36979940964C540FF0739218280115B4D442B089", "31A22B364EEB079100E9B387044CCAA92CF66BD9A6003DA3B7B70FF764745889", "13B1E6BC62BB6ED21C2D898D56918A96A9B07651714A66A431F43F1D62C3DA89

## Test alert rank 209 (dominant shape: cycle_count)

- hash `E1022AA66C65B606577AE893BAF0247DC14D4038A493626C47B160061C8B10A6` at 2026-09-21 16:20:40+00:00, score 0.9387
- rwLFURXYy8CWZRvVXofDr9jeTvmo6uxkAm -> ruwa7wG4FJbtuAF4Qu3snzo9en3aeextQ: paid 56.6250 USD (~$56.34), delivered 87.4497 XRP (~$56.24); legs=2 conversion=True
- top SHAP drivers: `sg_branch_holdtime_median_h`=0.0633 (+1.92), `full_edge_share_of_pair_usd`=1 (+1.73), `irr_community_volume_conc_6h`=0.0126 (+1.30), `asof_src_best_amount_gap_10m`=0.209 (+0.52), `full_pair_tx`=1 (+0.50)
- motif evidence: fan_in=0.896, fan_out=2, scatter_gather_middlemen=5, gather_scatter_score_6h=4.55, cycle_count=8, chain_longest_hops=8, bipartite_rectangles=37, irregular_complexity=10.8, forward_match_q=0.33, backward_match_q=0.387
- witnesses: {"forward_match": ["E1022AA66C65B606577AE893BAF0247DC14D4038A493626C47B160061C8B10A6", "DF7B29FEECD815D0C2D9C5E5BCD9E09C2BFD5743BB2BAAAEFE5549E715C63E93"], "backward_match": ["AC7C57015418C745AE9B2E6B57E7E7611698279E9D7847F602E8FD6552C83D14", "E1022AA66C65B606577AE893BAF0247DC14D4038A493626C47B160061C8B10A6"], "cycle": {"motif": "cycle", "hops": 5, "hashes": ["6088EE1CF9472C108C5C3A42222708951FC5B30934EB6129E3C024E947AE8337", "E1022AA66C65B606577AE893BAF0247DC14D4038A493626C47B160061C8B10A6", "DF7B29FEECD815D0C2D9C5E5BCD9E09C2BFD5743BB2BAAAEFE5549E715C63E93", "07320723AE902898D2F734F30DC083162

## Labelled validation: highest-scoring legitimate transactions (false positives)

- `F5ADB603C184AFA156AC7E53C0FCF273010B2F176819E25EF6BA3534AD8EFC5F` OOF score 0.9977, family `bipartite`: drivers `full_edge_share_of_pair_usd`, `irr_community_volume_conc_6h`, `full_pair_tx`, `fanout_equal_size_frac_10m`; sender tx=45/57, receiver pass-through=0.61. Likely a payroll/exchange/merchant look-alike: uniform or rapid flows without purposeless hops.
- `5E29F6F4E779FBC0D8B082DC27A5D763CCFA6850F1D78125F5DF77B34654E267` OOF score 0.9989, family `fan_out_in`: drivers `full_edge_share_of_pair_usd`, `fanout_equal_size_frac_10m`, `win_src_out_n_10m`, `fwd_dst_best_amount_gap_1h`; sender tx=62/56, receiver pass-through=0.96. Likely a payroll/exchange/merchant look-alike: uniform or rapid flows without purposeless hops.
- `488894CFEED4BD0FC9356132C9D7D5C2705FD3AD9D6AFFD8F1CB9E1B63CC02CB` OOF score 0.9982, family `fan_out_in`: drivers `full_edge_share_of_pair_usd`, `fanout_equal_size_frac_10m`, `win_src_out_n_10m`, `fwd_dst_best_amount_gap_1h`; sender tx=62/56, receiver pass-through=0.97. Likely a payroll/exchange/merchant look-alike: uniform or rapid flows without purposeless hops.

## Benign-looking transactions inside busy irregular neighbourhoods (low score, high ego complexity)

- `B2C618AC7E5F5C396439E50C5A00B7B8A41F4183F9C43C748607D2D6622BE74F` OOF score 0.0000, irregular complexity 14.15, label background: the model keeps it low because value-matching and single-use-intermediary evidence is absent.
- `4B8F18C8B2DEA4B457C806B91FD553A7DE972911107CE57CD233595E72C4874D` OOF score 0.0001, irregular complexity 14.24, label background: the model keeps it low because value-matching and single-use-intermediary evidence is absent.

## Highest-ranked planted cases (labelled)

- `DC55BD3F55FDB7D7375D34D1A4329F48EC00F8E377CD13BC3F61C5520724EAF6` family `multi_stage` (easy), OOF score 0.9998: `win_dst_in_deg_10m`, `sg_branch_holdtime_median_h`, `full_edge_share_of_pair_usd`, `win_dst_in_n_10m`
- `A6E5BC363B91F8B4B65DD4E45A50161EDED89795F72E33C6A7B5DCED6AC897C9` family `multi_stage` (easy), OOF score 0.9999: `sg_branch_holdtime_median_h`, `win_dst_in_deg_10m`, `full_edge_share_of_pair_usd`, `win_dst_in_n_10m`
- `EED15F91EABF1415E702EF64797622D29FCF9E77F6EA41A7632705F7D97A4FF5` family `multi_stage` (easy), OOF score 1.0000: `sg_branch_holdtime_median_h`, `win_dst_in_deg_10m`, `full_edge_share_of_pair_usd`, `win_dst_in_n_10m`

## Why legitimate activity imitates laundering shapes

- Exchange / payment-processor hot wallets look like gather-scatter hubs: many senders, many receivers, balanced flow. The counter-evidence is sustained multi-day activity, repeat counterparties and low synchronisation (`full_*_service_likeness`, `full_*_one_use_*_frac`).
- Payroll and refunds look like fan-out: uniform payouts, but recipients keep the money (`fanout_recipient_forward_frac_*` near 0) and the payer was not just funded.
- Merchant intake and fundraisers look like fan-in: many senders, but no prompt sweep of the collected total (`fanin_sweep_quality`, `fanin_last_deposit_to_exit_sec`).
- Market making looks like cycles: value returns, but usually at different scale or with many unrelated flows in between.

## Measured validation (blocked flow-group folds)

- LightGBM blocked OOF AP: 68.39 (folds [70.66, 74.46, 63.85])
- GNN blocked OOF AP: 54.53
- Chosen GNN blend weight: 0.0 (blend AP 68.39)
