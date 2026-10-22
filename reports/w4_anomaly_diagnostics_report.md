# W4: Statistical Diagnostics for KPI Anomaly Detection

## 1. Experimental scope

- Input: data\processed\kpi_timeseries.csv
- Training hours: 0–671 (672 hours)
- Test hours: 672–839 (168 hours)
- Beams: 2,880
- Full-test beam-hours: 483,840
- Consensus candidates: 3,165 (0.65%)
- IQR fallback count across beam-KPI training distributions: 0

No verified network-fault labels are available. All detections are anomaly candidates, not confirmed network faults.

## 2. Overall detection rates

| Method | Flagged beam-hours | Full-test rate |
|---|---:|---:|
| 3-sigma | 22,059 | 4.56% |
| IQR | 51,318 | 10.61% |
| Isolation Forest | 9,900 | 2.05% |
| Consensus: IF AND (3-sigma OR IQR) | 3,165 | 0.65% |

The One-Class SVM result from the earlier W3 experiment was evaluated on a sampled subset of 12,000 test records. Its 4.60% rate must not be compared as if it were measured on the full test set.

## 3. Distribution evidence and the IQR-versus-3-sigma hypothesis

The current run marked 10.61% of beam-hours with IQR and 4.56% with 3-sigma (IQR rate / 3-sigma rate = 2.33×).

A plausible methodological explanation is that the conventional IQR fences (Q1 − 1.5×IQR, Q3 + 1.5×IQR) are narrower than ±3 standard deviations for a Gaussian distribution. Non-Gaussian tails, skewness, beam-level heterogeneity, and train/test distribution shift can also affect the observed rates.

The following table tests whether the per-beam training distributions exhibit skewness/heavy tails and reports per-KPI flag rates. Shape statistics are calculated per beam over training hours; flag rates here are per KPI observation, not per beam-hour with any KPI flagged.

| KPI | Median train IQR/beam | Zero-IQR fallback distributions | Beams with absolute skewness > 1 | Beams with excess kurtosis > 3 | 3-sigma KPI flag rate | IQR KPI flag rate |
|---|---:|---:|---:|---:|---:|---:|
| DLPRB | 0.302 | 0 | 86.2% | 59.8% | 1.77% | 4.13% |
| DLThpVol | 0.174 | 0 | 98.3% | 75.5% | 1.60% | 4.59% |
| DLThpTime | 0.245 | 0 | 96.6% | 76.1% | 1.85% | 4.87% |
| MR_number | 0.314 | 0 | 59.8% | 21.8% | 1.39% | 2.76% |

Interpretation must follow these observed values. If skewness/heavy tails are common, that supports a non-Gaussian-distribution explanation; if they are uncommon, the difference may instead be driven more by the relative fence widths or temporal distribution shift. A higher IQR rate alone does not prove any one explanation.

## 4. AND/OR consensus logic

The implemented rule is:

Consensus = IsolationForest AND (3-sigma OR IQR)

A record must be flagged by Isolation Forest and by at least one of the two statistical detectors. The rule is a screening compromise: the statistical detector supplies a second signal, while the AND with Isolation Forest reduces the candidate set. The OR between 3-sigma and IQR prevents the consensus from requiring both statistical rules to agree.

| Rule | Flagged beam-hours | Full-test rate |
|---|---:|---:|
| 3sigma | 22,059 | 4.56% |
| IQR | 51,318 | 10.61% |
| Isolation Forest | 9,900 | 2.05% |
| 3sigma OR IQR | 51,373 | 10.62% |
| 3sigma AND IQR | 22,004 | 4.55% |
| IF AND 3sigma | 2,080 | 0.43% |
| IF AND IQR | 3,159 | 0.65% |
| IF AND (3sigma OR IQR) | 3,165 | 0.65% |
| IF AND 3sigma AND IQR | 2,074 | 0.43% |
| IF OR 3sigma OR IQR | 58,108 | 12.01% |

The table compares this rule against stricter IF AND 3-sigma AND IQR and looser IF OR 3-sigma OR IQR alternatives. These rates describe detector agreement, not fault-classification accuracy.

## 5. KPI correlations among consensus candidates

The correlation matrix uses all 3,165 consensus-candidate beam-hours. Confidence intervals are from a bootstrap that resamples base stations; p-values test a zero regression slope with base-station-clustered standard errors, followed by Holm correction across the six KPI pairs.

| KPI pair | n | Pearson r | 95% station-cluster bootstrap CI | Holm-adjusted cluster p |
|---|---:|---:|---:|---:|
| DLPRB – DLThpVol | 3,165 | 0.423 | [0.169, 0.601] | 7.61e-05 |
| DLPRB – DLThpTime | 3,165 | 0.697 | [0.485, 0.795] | 3.53e-10 |
| DLPRB – MR_number | 3,165 | 0.889 | [0.841, 0.918] | 1.1e-28 |
| DLThpVol – DLThpTime | 3,165 | 0.680 | [0.535, 0.770] | 2.68e-08 |
| DLThpVol – MR_number | 3,165 | 0.327 | [0.065, 0.488] | 0.0101 |
| DLThpTime – MR_number | 3,165 | 0.630 | [0.449, 0.758] | 8.36e-08 |

A statistically significant correlation indicates that KPI values co-vary within the selected candidate sample; it does not identify a causal root cause. Because candidate selection itself depends on the KPI-derived detectors, these correlations are exploratory and should be interpreted alongside the full-test distribution results and engineering context.

There are 30 base-station clusters in the candidate sample, so cluster-level inference remains approximate.

## 6. Figure captions and conclusions

**Figure 1. Per-KPI anomaly flag rates.** Percentage of individual KPI observations in the full test set flagged by the 3-sigma and IQR rules, shown separately for DLPRB, DLThpVol, DLThpTime, and MR_number. The denominator is the number of test beam-hours for that KPI. Compare the per-KPI rates rather than assuming one KPI accounts for the higher overall IQR rate.

**Figure 2. Training-distribution shape by beam.** Boxplots of per-beam skewness and excess kurtosis for each KPI, computed from the training period used to estimate thresholds. Large absolute skewness or positive excess kurtosis indicates departures from a symmetric Gaussian-like distribution. Use the corresponding summary table to quantify, rather than infer from the plot alone, how prevalent those departures are.

**Figure 3. KPI correlation matrix among consensus candidates.** Pearson correlation coefficients for the four KPI values in the selected candidate beam-hours. Asterisks mark Holm-adjusted, base-station-cluster-robust tests (* p < .05, ** p < .01, *** p < .001; ns, not significant); cell labels show r. Correlation strength is not evidence of causality.

## 7. Reproducibility and output files

- reports/w4_anomaly_test_flags.csv: full-test detector flags and KPI values.
- reports/w4_anomaly_logic_comparison.csv: counts and rates for alternative AND/OR rules.
- reports/w4_anomaly_distribution_summary.csv: pooled raw-KPI descriptive statistics for train and test.
- reports/w4_anomaly_threshold_shape_summary.csv: per-beam training-distribution shape and per-KPI flag rates.
- reports/w4_anomaly_kpi_correlation_matrix.csv: Pearson correlation matrix.
- reports/w4_anomaly_kpi_correlation_tests.csv: sample sizes, nominal and cluster-robust p-values, Holm correction, and cluster-bootstrap confidence intervals.
- reports/w4_anomaly_kpi_correlation_pvalues.csv and reports/w4_anomaly_kpi_correlation_pvalues_holm.csv: p-value matrices.
- figures/w4_anomaly_kpi_flag_rates.png
- figures/w4_anomaly_training_distribution_shape.png
- figures/w4_anomaly_kpi_correlation_significance.png

## 8. Limitations

- No verified fault labels; precision, recall, and F1 cannot be calculated.
- Correlation analysis is conditional on a detector-selected candidate set.
- Cluster-robust inference uses base station as the cluster unit; a limited number of clusters reduces precision.
- Detection thresholds may be sensitive to temporal distribution shift.
- One-Class SVM remains a sampled-test result from the earlier experiment and is not re-estimated by this diagnostic script.
