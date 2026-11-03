# Quick check: elephant-1 against d1-3B, 2 Nov 2026

Quick check, not a measurement: a handful of requests, no speed figures. Both runners on this Mac (M5 Max, Mac in
use), each through its own worker with default settings: elephant-1 at FP16, d1-3B at BF16 (its installed snapshot).

**The existing decision fixtures** (`Tests/Fixtures/Bench/kinds/decisions/fixtures-v1.json`, the only labelled
decision set in the repository, four cases). Both answer all four correctly. Probability on the labelled answer:

| Case | d1-3B | elephant-1 |
|---|---|---|
| yes: "three red apples", contains apples? | 0.991 | 0.999 |
| no: "the box is empty", contains apples? | 0.982 | 0.984 |
| choice: cat / dog / bird | 0.990 | 1.000 |
| score: shop closed, how blocked (3 levels) | 0.647 | 0.995 |
| mean | 0.900 | 0.994 |

**Agreement on five unlabelled requests** (the card's ticket, nested JSON, a 30-option city choice, Polish, questions
with no instructions; 11 questions): the same top answer on 8 of 11. They differ on the JSON frustration level (d1
"irritated" 0.59, elephant "angry" 0.44), and on the two questions with no instructions over a bare list, which have no
right answer. Both pick Poznań among 30 cities and read the Polish complaint as a refund request.

What this cannot show: four labelled cases cannot rank two models; the makers' published indexes (elephant-1 62.65 on
the Jev Decision Index 0.4; d1-3B 48.57 on Decision Index 0.2.1) are different benchmarks and versions.
