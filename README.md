# GTA VI Twitch CCU forecaster

Forecasts the hourly Twitch concurrent viewers of the GTA VI launch (hours
2–24 of release day, 19 Nov 2026) as five quantiles (5/25/50/75/95%), from
the hour-1 print and the launch curves of 38–42 comparable big releases.

## Method

Comparable-launch log-ratio ensemble with a locally weighted ridge
regression per forecast hour:

1. For every comparable launch, build the curve of log ratios
   `r_h = ln(viewers(h) / viewers(1))` for hours 2..24 of its release day.
2. Bigger launches have systematically flatter curves (their hour-1 print
   is already huge, so the rest of the day grows less in proportion), so
   the median ratio is modelled per hour as a ridge regression of `r_h` on
   `[1, ln(hour-1 viewers), ln(hour-1 viewers / pre-7d mean viewers),
   ln(hour-1 channels)]`, with a Gaussian kernel on `ln(hour-1 viewers)`
   (bandwidth 1.5) so launches of a similar scale dominate, and a small
   ridge penalty (1.0) for stability when the target scale lies beyond the
   largest comparable — as GTA VI's will.
3. Quantiles are lognormal around the fitted value, spread = kernel-weighted
   residual standard deviation × 0.9, multiplied by the target's hour-1 print.

Hyperparameters were tuned by leave-one-out cross-validation over the
comparables and on the four practice cases.

## Results (local practice, SullyGnome "Average viewers" answers)

| case | loss |
|---|---|
| Marvel's Wolverine | 0.163 |
| Red Dead Redemption 2 | 0.190 |
| God of War Ragnarök | 0.125 |
| Marvel's Spider-Man 2 | 0.187 |
| **mean** | **0.166** |

House persistence baseline: **0.387**. Mean pinball loss on the log scale,
same formula as the competition scorer.

## Run

```sh
python3 main.py   # reads $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR (default /output)
```

Python 3.13 standard library only. Deterministic (two runs byte-identical),
no network, ~0.5 s for four cases, far inside the 8 s / 1 CPU / 1024 MB limit.

## License

MIT — see [LICENSE](LICENSE).
