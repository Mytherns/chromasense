# Color Evaluation Report

- Ground truth: `data\ground_truth_palettes.json`
- Images dir: `test_images\color_eval`
- Colors per image: `5`
- Delta E threshold: `6.0`
- Target: 85%+ matched color pairs with Delta E < 6
- Fixed primary algorithm: `kmeans`
- Style feature source remains: `kmeans`

## Summary

| Algorithm | Images | Colors | Delta E < threshold | Mean Delta E |
| --- | ---: | ---: | ---: | ---: |
| kmeans | 2 | 10 | 1.000 | 0.000 |
| gmm | 2 | 10 | 1.000 | 0.000 |

## Limitation

Current committed color-eval fixtures are tiny deterministic swatches for repeatable smoke verification. Replace or extend with real manually labeled photos before making a final report claim.

## Per Image

### kmeans

- `primary_stripes.ppm`: rate `1.000`, mean Delta E `0.000`
- `muted_stripes.ppm`: rate `1.000`, mean Delta E `0.000`

### gmm

- `primary_stripes.ppm`: rate `1.000`, mean Delta E `0.000`
- `muted_stripes.ppm`: rate `1.000`, mean Delta E `0.000`
