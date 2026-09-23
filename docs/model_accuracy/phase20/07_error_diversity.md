# Phase 20 — Error Diversity Analysis

## Mechanism Under Test
```
Different models → different errors → averaging → partial error cancellation
```

## Residual (Error) Correlations

| Pair | Correlation |
|---|---|
| M0 — Ridge | 0.9137 |
| M0 — RF    | 0.9616 |
| Ridge — RF | 0.9514 |

> Lower correlation = more diversity = more cancellation potential.

## Prediction Correlations

| Pair | Correlation |
|---|---|
| M0 — Ridge | 0.5975 |
| M0 — RF    | 0.8418 |
| Ridge — RF | 0.697 |

## Complementarity

| Cases | Count |
|---|---|
| M0 right, Ridge wrong | 502 |
| M0 wrong, Ridge right | 573 |
| M0 right, RF wrong    | 347 |
| M0 wrong, RF right    | 456 |

## Overall: Does averaging help?

- Observations where A5a < M0 error: 2688 (56.7%)
- Large disagreement cases (>1% GDP): 1236 (26.1%)

## Conclusion
Models are highly correlated — diversity benefit is limited.
