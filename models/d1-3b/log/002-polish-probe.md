+++
hypothesis = "d1 may improve Polish task quality over Laya."
change = "Fresh pre-labeled JobRadar snapshot and same-session interleaved product calls."
measurement = "Historical d1 counts from the faulty torch MPS runner (corrected below and in log/004: Polish Noul 24/32 MLX, 25/32 fp32). M5 Max, 8 Oct 2026, d1 bf16 torch MPS vs Laya fp16 Fast: Polish Noul 27/32 vs 17/32; SEP Choice mandatory positives 0/11 vs 9/11."
verdict = "kept"
numbers = { d1_polish_noul_correct = 27, laya_polish_noul_correct = 17, ads = 32, d1_sep_choice_mandatory_found = 0, laya_sep_choice_mandatory_found = 9, mandatory = 11 }
links = ["gate/job_probe.py"]
+++
# Fresh Polish task probe

## Fresh Polish results — M5 Max, 8 Oct 2026

Signed Arsenal 3.0.1 candidate d1-3B bf16 torch MPS versus installed Arsenal 3.0.0 laya-multilingual fp16 Fast. A fresh manually labeled sample, read before inference; no recreation of the deleted 7 Oct inputs. 32 real pracuj.pl ads, 16 service/installation and 16 other. 24 SEP-mention requirements: 11 mandatory, 13 optional/obtainable later. All mention SEP, so presence has no absent-SEP negatives. Identical original question definitions; matched chunks of eight, alternating both models and reversing pair order on alternate chunks; warm load excluded. Non-quiet daytime end-to-end HTTP wall timing, not a catalog benchmark. Laya settings and residency restored. Started 2026-10-08T04:46:49.759517+00:00, completed 2026-10-08T04:47:22.856871+00:00.

| Question | d1 correct | Laya correct | d1 mandatory/service found | Laya mandatory/service found |
|---|---:|---:|---:|---:|
| en_noul | 20/32 | 17/32 | 16/16 | 16/16 |
| pl_noul | 27/32 | 17/32 | 16/16 | 16/16 |
| pl_choice P(serwis)>0.5 | 26/32 | 23/32 | 15/16 | 11/16 |
| pl_choice winning label | 24/32 | 23/32 | 15/16 | 12/16 |
| full sep_mention | 24/24 | 20/24 | 24/24 | 20/24 |
| full sep_required | 18/24 | 11/24 | 6/11 | 9/11 |
| full sep_choice | 13/24 | 19/24 | 0/11 | 9/11 |
| excerpt sep_mention | 24/24 | 18/24 | 24/24 | 18/24 |
| excerpt sep_required | 14/24 | 11/24 | 4/11 | 11/11 |
| excerpt sep_choice | 13/24 | 19/24 | 0/11 | 10/11 |

| Identical question bundle | d1 warm HTTP ms/item | Laya warm HTTP ms/item |
|---|---:|---:|
| ads | 188.8 | 12.6 |
| full | 122.1 | 10.8 |
| excerpt | 124.9 | 9.0 |

| Ranked top 16 service positives | d1 count (of 16) | Laya count (of 16) |
|---|---:|---:|
| en_noul | 15 | 11 |
| pl_noul | 14 | 11 |
| pl_choice P(serwis)>0.5 | 12 | 11 |

Polish Noul improves on this sample. Six-way Choice is only 24/32 versus 23/32 by winning label and its ranked top 16 shows only a one-item gain (12/16 versus 11/16; English Noul reaches 15/16, Polish 14/16). d1’s SEP Choice predicts no mandatory cases: 0/11 detected, 13/24 overall by choosing the majority nonmandatory class. Its mandatory Noul finds 6/11 in full sections and 4/11 in excerpts. Use code for presence and source reading for requirement strength; d1 does not replace literal requirements checks. Its extra image/context capacity is available, but this probe does not establish a general quality advantage.

Inputs, labels, questions, snapshot receipt and all answers and timings were kept privately and are not distributed. Input hashes:
- `inputs.json`: `2ae7b21838a4a1d51db67d3a856935cac2dbb6e576390e4648ba16e4e25cf2c6`
- `labels.json`: `8511423d2c74cbeeebd4f12db676440842d3eb75dfcd3633d8b7e109e4ad89bd`
- `questions.json`: `8371815549c8ad87e89b7efd5da9872d0e42eb014ec8eaca97cd0658300517db`

## Correction (8 Oct 2026)

These d1 counts came from the torch MPS runner, whose causal attention is faulty in bf16 (log/003). Against the vendor code in fp32 they were off by up to three items per question; the MLX rerun and the fp32 counts are in log/004 and `gate/reference-check.json`. Polish Noul is 24/32 (MLX) and 25/32 (fp32), not 27/32.
