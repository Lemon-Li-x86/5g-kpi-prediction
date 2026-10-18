# W3: KPI Anomaly Detection and Preliminary Root-Cause Analysis

## 1. Experimental scope

- Input dataset: data\processed\kpi_timeseries.csv
- Training period: Time < 672
- Test period: Time >= 672
- Total test beam-hours: 483,840
- Consensus anomaly candidates: 3,165
- Consensus anomaly rate: 0.65%

This dataset does not contain verified network-fault labels.
Anomaly candidates must not be interpreted as confirmed faults.

## 2. Detection methods

### Three-sigma rule

Per-beam training mean and standard deviation are used.
An observation is flagged when any KPI exceeds 3 standard deviations.

Full-test flag rate: 4.56%

### IQR

Per-beam Q1 and Q3 are used to construct IQR fences.
For 0 zero-IQR beam-KPI distributions, the
1st/99th training percentiles are used as fallback bounds.

Full-test flag rate: 10.61%

### Isolation Forest

Input features:
- Four current standardized KPI values
- Four causal first-difference values

Training sample size: 80,000

Full-test flag rate: 2.05%

### One-Class SVM

Kernel: RBF
Training samples: 2,500
Test subset: 12,000

Sampled-test flag rate: 4.60%

The One-Class SVM result is calculated on a sampled subset
and must not be presented as a full-test detection rate.

## 3. Consensus detection

Consensus is defined as:

Isolation Forest flag AND (Three-sigma flag OR IQR flag).

It is a screening criterion, not a verified fault classifier.

## 4. KPI association and possible investigation directions

The four investigated indicators are:
- DLPRB: Downlink physical resource block usage
- DLThpVol: Downlink traffic volume
- DLThpTime: Downlink transmission time
- MR_number: Measurement report count

Inspect the event table and correlation matrix to identify
which KPI deviations tend to occur together.

Possible investigation hypotheses include:
- High DLPRB accompanied by changed throughput demand:
  investigate potential load or resource pressure.
- Sudden DLThpVol or DLThpTime changes:
  investigate changes in traffic or scheduling conditions.
- Unusual MR_number changes:
  investigate measurement activity and related network context.

These are hypotheses only. KPI correlation does not prove
any specific cause.

## 5. Limitations

- No verified network fault labels.
- No precision, recall or F1 can be reported.
- Thresholds may be sensitive to nonstationary traffic.
- One-Class SVM uses sampled training and test records.
- Additional RSRP, SINR, traffic, alarm and equipment logs
  would be required for more reliable root-cause analysis.

## 6. Saved results

- w3_anomaly_method_summary.csv
- w3_anomaly_rate_by_hour.csv
- w3_anomaly_candidate_events.csv
- w3_anomaly_kpi_correlations.csv
- w3_one_class_svm_test_subset.csv

