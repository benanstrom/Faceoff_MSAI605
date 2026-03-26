# Real LFW Run Comparison

| Run | Threshold | Val Balanced Acc. | Test Balanced Acc. | Test Acc. | Precision | Recall | TP | FP | TN | FN | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_16 | 0.82 | 0.6124 | 0.5668 | 0.5668 | 0.5705 | 0.5408 | 1352 | 1018 | 1482 | 1148 | Baseline exploratory run at 16x16. |
| baseline_24 | 0.81 | 0.6182 | 0.5614 | 0.5614 | 0.5717 | 0.4896 | 1224 | 917 | 1583 | 1276 | Baseline exploratory run at 24x24. |
| baseline_32 | 0.80 | 0.6210 | 0.5648 | 0.5648 | 0.5761 | 0.4904 | 1226 | 902 | 1598 | 1274 | Baseline reporting run at 32x32. |
| improved_32 | 0.77 | 0.6100 | 0.5852 | 0.5852 | 0.5754 | 0.6504 | 1626 | 1200 | 1300 | 874 | Fair data-centric comparison using prefer_unique pairs at 32x32. |
| improved_48 | 0.77 | 0.6112 | 0.5886 | 0.5886 | 0.5871 | 0.5972 | 1493 | 1050 | 1450 | 1007 | Improved exploratory run at 48x48. |
