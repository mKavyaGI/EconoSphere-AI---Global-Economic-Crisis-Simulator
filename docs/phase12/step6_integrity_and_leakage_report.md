# Phase 12 Step 6: Integrity & Leakage Report

## Dataset Hash
- **MD5**: 8ac7e0b2bf09fbe89289f82d0c7cf25e (Verified)

## Protected Artifact Hashes
- `models/phase11/best_t1_gdp_growth_model.joblib`: 7ce62cb877b47e5573a9e0342da9709d
- `models/phase11/gradient_boosting_baseline.joblib`: 7ce62cb877b47e5573a9e0342da9709d
- `models/phase11/t1_model_metadata.json`: 93249112c3bf901f230299098b8878ab
- `data/processed/t1_step11_predictions.csv`: 744bcd3614333a24c73fdf9d14720002
- `data/processed/t1_step11_2026_forecasts.csv`: 4d8a646965884a1d51a2533ff0b33075
- `models/phase11/t1_step11_model_metadata.json`: e278b508301c974bfb3b237a2e57c041
- `models/phase12/step1_walk_forward_metadata.json`: 000da07b89043d0c6ab5a068a97ad314
- `models/phase12/step2_gdp_momentum_metadata.json`: e500bf5da4175bdb8b2ab548ea0bd15e
- `models/phase12/step3_forward_indicators_metadata.json`: 51bf147744bb1402e77e7257dc844410
- `models/phase12/step4_regime_aware_metadata.json`: 82f21c55995f97ddf4a821e6facd940e
- `models/phase12/step5_shap_metadata.json`: f64ee456968468922a4ad7df8ed1106a
- `data/processed/phase12_step2_2026_forecasts.csv`: 431ec40310ef2930efc7814acfc6540d
- `data/processed/phase12_step2_experiment_metrics.csv`: 5ba0187faa100f603d365dbbb370788f
- `data/processed/phase12_step2_momentum_features.csv`: bb899055b29fbec39f3102779b2ce5ec
- `data/processed/phase12_step2_predictions.csv`: 5279b12617d858e0486e85a87b6aa6bf
- `data/processed/phase12_step2_turning_point_analysis.csv`: 406c8fe47da42747669ed96efde758d0
- `data/processed/phase12_step2_walk_forward_metrics.csv`: 88e0361a74c9c2cff6264e60c467fd09
- `data/processed/phase12_step2_walk_forward_predictions.csv`: 91bd9be2d8a2813f5a6b24a99c1dc969
- `data/processed/phase12_step3_2026_forecasts.csv`: 07bdbe6ccc98601e7c7766535cd107bf
- `data/processed/phase12_step3_experiment_metrics.csv`: 6fbf431d82ff821a2ca75d27c7383b49
- `data/processed/phase12_step3_indicator_audit.csv`: 457f3e5cc51e98f1eab111e54d4a1a4d
- `data/processed/phase12_step3_predictions.csv`: 560ed2c0975408c3d3f488203e8ab210
- `data/processed/phase12_step3_recession_analysis.csv`: dada37915f5120fe1fab9f8ec4a68f8f
- `data/processed/phase12_step3_walk_forward_metrics.csv`: cdefbe627c451027ba1a8d02f5eb8171
- `data/processed/phase12_step3_walk_forward_predictions.csv`: 9f7f7d5e3f52dfc69ad60d04cb673be2
- `data/processed/phase12_step4_experiment_metrics.csv`: 2484dbf77202b4d754f12b6d2560b7db
- `data/processed/phase12_step4_predictions.csv`: c8fadca21014910368388b32f6e23cd5
- `data/processed/phase12_step4_regime_feature_audit.csv`: 326f7f4c3392cd2240baaca118cbc68d
- `data/processed/phase12_step4_regime_performance.csv`: 413b414a91679a0fb21d5aa0520bbdbb
- `data/processed/phase12_step4_walk_forward_metrics.csv`: a56e49e461988a6b61285d356e580e03
- `data/processed/phase12_step4_walk_forward_predictions.csv`: 041accaab7717c178bf4c7f79aa43e77
- `data/processed/phase12_step5_2026_experimental_explanations.csv`: 4ca824594b07b0b1c15a37685dd4f7d0
- `data/processed/phase12_step5_control_vs_candidate.csv`: 9f62c9bfa8279e91ac30ea8d45d00f01
- `data/processed/phase12_step5_shap_global_importance.csv`: fcb508b9e28e6b03b497408ad7cd5dc9
- `data/processed/phase12_step5_shap_regime_importance.csv`: 38fb9d4cf3b1615579934253539d6f7f
- `data/processed/phase12_step5_shap_validation_predictions.csv`: bc1c557257ab89c893f1402f9bfdfe01
- `data/processed/phase12_step5_shap_walk_forward_importance.csv`: dd1bf4c45da6f06d749e2b30585e1068

## Temporal Boundaries
- **Training**: <= 2018
- **Validation**: 2019-2022
- **Test**: 2023-2024
- **Walk-forward**: 2013-2024

## Feature Construction
Regime thresholds (33rd/67th/80th percentiles) were computed strictly on active training data.

## Preprocessing Isolation
SimpleImputer median fit strictly on training splits.

## Target Leakage Checks
No next_year target used during feature generation or threshold calculation.

## Statistical Methodology
- Paired differences
- Wilcoxon Signed-Rank
- Permutation Tests
- Bootstrap CIs

## Random Seed Controls
- Tested 5 seeds: [42, 7, 21, 123, 2026]
- Production base remains locked at 42.

## Production Immutability
- `production_model_modified`: False

## Reproducibility
- Seed 42 used for all deterministic processes.

## Test Results
All Step 6 automated tests passed.
