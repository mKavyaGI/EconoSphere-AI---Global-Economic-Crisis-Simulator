# Phase 19 — Prediction Diversity Analysis

Pairwise prediction correlation between candidates across all origins.
High correlation (>0.95) means models make nearly identical predictions,
and an ensemble over them provides little additional diversity.

| origin | pair | corr |
| --- | --- | --- |
| 2002 | Phase 11 (M0) vs Ridge (A1) | 0.5902 |
| 2002 | Phase 11 (M0) vs Elastic Net (A2) | 0.5899 |
| 2002 | Phase 11 (M0) vs Huber (A3) | 0.6412 |
| 2002 | Phase 11 (M0) vs Random Forest (A4) | 0.8497 |
| 2002 | Phase 11 (M0) vs Ensemble Equal (A5a) | 0.9492 |
| 2002 | Phase 11 (M0) vs Ensemble Val-Weighted (A5b) | 1.0000 |
| 2002 | Phase 11 (M0) vs Residual Model (A6) | 0.9693 |
| 2002 | Ridge (A1) vs Elastic Net (A2) | 0.9837 |
| 2002 | Ridge (A1) vs Huber (A3) | 0.9444 |
| 2002 | Ridge (A1) vs Random Forest (A4) | 0.6185 |
| 2002 | Ridge (A1) vs Ensemble Equal (A5a) | 0.7766 |
| 2002 | Ridge (A1) vs Ensemble Val-Weighted (A5b) | 0.5902 |
| 2002 | Ridge (A1) vs Residual Model (A6) | 0.6646 |
| 2002 | Elastic Net (A2) vs Huber (A3) | 0.9380 |
| 2002 | Elastic Net (A2) vs Random Forest (A4) | 0.6184 |
| 2002 | Elastic Net (A2) vs Ensemble Equal (A5a) | 0.7720 |
| 2002 | Elastic Net (A2) vs Ensemble Val-Weighted (A5b) | 0.5899 |
| 2002 | Elastic Net (A2) vs Residual Model (A6) | 0.6499 |
| 2002 | Huber (A3) vs Random Forest (A4) | 0.6860 |
| 2002 | Huber (A3) vs Ensemble Equal (A5a) | 0.8095 |
| 2002 | Huber (A3) vs Ensemble Val-Weighted (A5b) | 0.6412 |
| 2002 | Huber (A3) vs Residual Model (A6) | 0.6945 |
| 2002 | Random Forest (A4) vs Ensemble Equal (A5a) | 0.9289 |
| 2002 | Random Forest (A4) vs Ensemble Val-Weighted (A5b) | 0.8497 |
| 2002 | Random Forest (A4) vs Residual Model (A6) | 0.8071 |
| 2002 | Ensemble Equal (A5a) vs Ensemble Val-Weighted (A5b) | 0.9492 |
| 2002 | Ensemble Equal (A5a) vs Residual Model (A6) | 0.9400 |
| 2002 | Ensemble Val-Weighted (A5b) vs Residual Model (A6) | 0.9693 |
| 2003 | Phase 11 (M0) vs Ridge (A1) | 0.4154 |
| 2003 | Phase 11 (M0) vs Elastic Net (A2) | 0.4004 |
| 2003 | Phase 11 (M0) vs Huber (A3) | 0.5007 |
| 2003 | Phase 11 (M0) vs Random Forest (A4) | 0.7950 |
| 2003 | Phase 11 (M0) vs Ensemble Equal (A5a) | 0.9300 |
| 2003 | Phase 11 (M0) vs Ensemble Val-Weighted (A5b) | 1.0000 |
| 2003 | Phase 11 (M0) vs Residual Model (A6) | 0.8674 |
| 2003 | Ridge (A1) vs Elastic Net (A2) | 0.9961 |
| 2003 | Ridge (A1) vs Huber (A3) | 0.6858 |
| 2003 | Ridge (A1) vs Random Forest (A4) | 0.5099 |
| 2003 | Ridge (A1) vs Ensemble Equal (A5a) | 0.6606 |
| 2003 | Ridge (A1) vs Ensemble Val-Weighted (A5b) | 0.4154 |
| 2003 | Ridge (A1) vs Residual Model (A6) | 0.5556 |
| 2003 | Elastic Net (A2) vs Huber (A3) | 0.6859 |
| 2003 | Elastic Net (A2) vs Random Forest (A4) | 0.4897 |
| 2003 | Elastic Net (A2) vs Ensemble Equal (A5a) | 0.6443 |
| 2003 | Elastic Net (A2) vs Ensemble Val-Weighted (A5b) | 0.4004 |
| 2003 | Elastic Net (A2) vs Residual Model (A6) | 0.5536 |
| 2003 | Huber (A3) vs Random Forest (A4) | 0.6227 |
| 2003 | Huber (A3) vs Ensemble Equal (A5a) | 0.6683 |
| 2003 | Huber (A3) vs Ensemble Val-Weighted (A5b) | 0.5007 |
| 2003 | Huber (A3) vs Residual Model (A6) | 0.4518 |
| 2003 | Random Forest (A4) vs Ensemble Equal (A5a) | 0.9194 |
| 2003 | Random Forest (A4) vs Ensemble Val-Weighted (A5b) | 0.7950 |
| 2003 | Random Forest (A4) vs Residual Model (A6) | 0.6381 |
| 2003 | Ensemble Equal (A5a) vs Ensemble Val-Weighted (A5b) | 0.9300 |
| 2003 | Ensemble Equal (A5a) vs Residual Model (A6) | 0.8370 |
| 2003 | Ensemble Val-Weighted (A5b) vs Residual Model (A6) | 0.8674 |
| 2004 | Phase 11 (M0) vs Ridge (A1) | 0.6727 |
| 2004 | Phase 11 (M0) vs Elastic Net (A2) | 0.5884 |
| 2004 | Phase 11 (M0) vs Huber (A3) | 0.6550 |
| 2004 | Phase 11 (M0) vs Random Forest (A4) | 0.8403 |
| 2004 | Phase 11 (M0) vs Ensemble Equal (A5a) | 0.9504 |
| 2004 | Phase 11 (M0) vs Ensemble Val-Weighted (A5b) | 1.0000 |
| 2004 | Phase 11 (M0) vs Residual Model (A6) | 0.9442 |
| 2004 | Ridge (A1) vs Elastic Net (A2) | 0.7897 |
| 2004 | Ridge (A1) vs Huber (A3) | 0.9810 |
| 2004 | Ridge (A1) vs Random Forest (A4) | 0.6691 |
| 2004 | Ridge (A1) vs Ensemble Equal (A5a) | 0.8198 |
| 2004 | Ridge (A1) vs Ensemble Val-Weighted (A5b) | 0.6727 |
| 2004 | Ridge (A1) vs Residual Model (A6) | 0.6781 |
| 2004 | Elastic Net (A2) vs Huber (A3) | 0.8039 |
| 2004 | Elastic Net (A2) vs Random Forest (A4) | 0.5780 |
| 2004 | Elastic Net (A2) vs Ensemble Equal (A5a) | 0.6923 |
| 2004 | Elastic Net (A2) vs Ensemble Val-Weighted (A5b) | 0.5884 |
| 2004 | Elastic Net (A2) vs Residual Model (A6) | 0.5238 |
| 2004 | Huber (A3) vs Random Forest (A4) | 0.6401 |
| 2004 | Huber (A3) vs Ensemble Equal (A5a) | 0.7959 |
| 2004 | Huber (A3) vs Ensemble Val-Weighted (A5b) | 0.6550 |
| 2004 | Huber (A3) vs Residual Model (A6) | 0.6558 |
| 2004 | Random Forest (A4) vs Ensemble Equal (A5a) | 0.9339 |
| 2004 | Random Forest (A4) vs Ensemble Val-Weighted (A5b) | 0.8403 |
| 2004 | Random Forest (A4) vs Residual Model (A6) | 0.7956 |
| 2004 | Ensemble Equal (A5a) vs Ensemble Val-Weighted (A5b) | 0.9504 |
| 2004 | Ensemble Equal (A5a) vs Residual Model (A6) | 0.9093 |
| 2004 | Ensemble Val-Weighted (A5b) vs Residual Model (A6) | 0.9442 |
| 2005 | Phase 11 (M0) vs Ridge (A1) | 0.7174 |
| 2005 | Phase 11 (M0) vs Elastic Net (A2) | 0.7175 |
| 2005 | Phase 11 (M0) vs Huber (A3) | 0.7211 |
| 2005 | Phase 11 (M0) vs Random Forest (A4) | 0.8796 |
| 2005 | Phase 11 (M0) vs Ensemble Equal (A5a) | 0.9518 |
| 2005 | Phase 11 (M0) vs Ensemble Val-Weighted (A5b) | 1.0000 |
| 2005 | Phase 11 (M0) vs Residual Model (A6) | 0.9774 |
| 2005 | Ridge (A1) vs Elastic Net (A2) | 0.9993 |
| 2005 | Ridge (A1) vs Huber (A3) | 0.9480 |
| 2005 | Ridge (A1) vs Random Forest (A4) | 0.7863 |
| 2005 | Ridge (A1) vs Ensemble Equal (A5a) | 0.8668 |
| 2005 | Ridge (A1) vs Ensemble Val-Weighted (A5b) | 0.7174 |
| 2005 | Ridge (A1) vs Residual Model (A6) | 0.7235 |
| 2005 | Elastic Net (A2) vs Huber (A3) | 0.9403 |
| 2005 | Elastic Net (A2) vs Random Forest (A4) | 0.7869 |
| 2005 | Elastic Net (A2) vs Ensemble Equal (A5a) | 0.8669 |
| 2005 | Elastic Net (A2) vs Ensemble Val-Weighted (A5b) | 0.7175 |
| 2005 | Elastic Net (A2) vs Residual Model (A6) | 0.7205 |
| 2005 | Huber (A3) vs Random Forest (A4) | 0.7932 |
| 2005 | Huber (A3) vs Ensemble Equal (A5a) | 0.8574 |
| 2005 | Huber (A3) vs Ensemble Val-Weighted (A5b) | 0.7211 |
| 2005 | Huber (A3) vs Residual Model (A6) | 0.7458 |
| 2005 | Random Forest (A4) vs Ensemble Equal (A5a) | 0.9616 |
| 2005 | Random Forest (A4) vs Ensemble Val-Weighted (A5b) | 0.8796 |
| 2005 | Random Forest (A4) vs Residual Model (A6) | 0.8508 |
| 2005 | Ensemble Equal (A5a) vs Ensemble Val-Weighted (A5b) | 0.9518 |
| 2005 | Ensemble Equal (A5a) vs Residual Model (A6) | 0.9328 |
| 2005 | Ensemble Val-Weighted (A5b) vs Residual Model (A6) | 0.9774 |
| 2006 | Phase 11 (M0) vs Ridge (A1) | 0.6724 |
| 2006 | Phase 11 (M0) vs Elastic Net (A2) | 0.6756 |
| 2006 | Phase 11 (M0) vs Huber (A3) | 0.6794 |
| 2006 | Phase 11 (M0) vs Random Forest (A4) | 0.8606 |
| 2006 | Phase 11 (M0) vs Ensemble Equal (A5a) | 0.9365 |
| 2006 | Phase 11 (M0) vs Ensemble Val-Weighted (A5b) | 1.0000 |
| 2006 | Phase 11 (M0) vs Residual Model (A6) | 0.9867 |
| 2006 | Ridge (A1) vs Elastic Net (A2) | 0.9989 |
| 2006 | Ridge (A1) vs Huber (A3) | 0.9740 |
| 2006 | Ridge (A1) vs Random Forest (A4) | 0.7708 |
| 2006 | Ridge (A1) vs Ensemble Equal (A5a) | 0.8591 |
| 2006 | Ridge (A1) vs Ensemble Val-Weighted (A5b) | 0.6724 |
| 2006 | Ridge (A1) vs Residual Model (A6) | 0.7024 |
| 2006 | Elastic Net (A2) vs Huber (A3) | 0.9702 |
| 2006 | Elastic Net (A2) vs Random Forest (A4) | 0.7756 |
| 2006 | Elastic Net (A2) vs Ensemble Equal (A5a) | 0.8619 |
| 2006 | Elastic Net (A2) vs Ensemble Val-Weighted (A5b) | 0.6756 |
| 2006 | Elastic Net (A2) vs Residual Model (A6) | 0.7032 |
| 2006 | Huber (A3) vs Random Forest (A4) | 0.7892 |
| 2006 | Huber (A3) vs Ensemble Equal (A5a) | 0.8614 |
| 2006 | Huber (A3) vs Ensemble Val-Weighted (A5b) | 0.6794 |
| 2006 | Huber (A3) vs Residual Model (A6) | 0.6969 |
| 2006 | Random Forest (A4) vs Ensemble Equal (A5a) | 0.9583 |
| 2006 | Random Forest (A4) vs Ensemble Val-Weighted (A5b) | 0.8606 |
| 2006 | Random Forest (A4) vs Residual Model (A6) | 0.8359 |
| 2006 | Ensemble Equal (A5a) vs Ensemble Val-Weighted (A5b) | 0.9365 |
| 2006 | Ensemble Equal (A5a) vs Residual Model (A6) | 0.9303 |
| 2006 | Ensemble Val-Weighted (A5b) vs Residual Model (A6) | 0.9867 |
| 2007 | Phase 11 (M0) vs Ridge (A1) | 0.7848 |
| 2007 | Phase 11 (M0) vs Elastic Net (A2) | 0.8030 |
| 2007 | Phase 11 (M0) vs Huber (A3) | 0.8034 |
| 2007 | Phase 11 (M0) vs Random Forest (A4) | 0.9240 |
| 2007 | Phase 11 (M0) vs Ensemble Equal (A5a) | 0.9693 |
| 2007 | Phase 11 (M0) vs Ensemble Val-Weighted (A5b) | 1.0000 |
| 2007 | Phase 11 (M0) vs Residual Model (A6) | 0.9884 |
| 2007 | Ridge (A1) vs Elastic Net (A2) | 0.9947 |
| 2007 | Ridge (A1) vs Huber (A3) | 0.9664 |
| 2007 | Ridge (A1) vs Random Forest (A4) | 0.7804 |
| 2007 | Ridge (A1) vs Ensemble Equal (A5a) | 0.8855 |
| 2007 | Ridge (A1) vs Ensemble Val-Weighted (A5b) | 0.7848 |
| 2007 | Ridge (A1) vs Residual Model (A6) | 0.7915 |
| 2007 | Elastic Net (A2) vs Huber (A3) | 0.9663 |
| 2007 | Elastic Net (A2) vs Random Forest (A4) | 0.7999 |
| 2007 | Elastic Net (A2) vs Ensemble Equal (A5a) | 0.8989 |
| 2007 | Elastic Net (A2) vs Ensemble Val-Weighted (A5b) | 0.8030 |
| 2007 | Elastic Net (A2) vs Residual Model (A6) | 0.8040 |
| 2007 | Huber (A3) vs Random Forest (A4) | 0.8232 |
| 2007 | Huber (A3) vs Ensemble Equal (A5a) | 0.9002 |
| 2007 | Huber (A3) vs Ensemble Val-Weighted (A5b) | 0.8034 |
| 2007 | Huber (A3) vs Residual Model (A6) | 0.8047 |
| 2007 | Random Forest (A4) vs Ensemble Equal (A5a) | 0.9661 |
| 2007 | Random Forest (A4) vs Ensemble Val-Weighted (A5b) | 0.9240 |
| 2007 | Random Forest (A4) vs Residual Model (A6) | 0.9027 |
| 2007 | Ensemble Equal (A5a) vs Ensemble Val-Weighted (A5b) | 0.9693 |
| 2007 | Ensemble Equal (A5a) vs Residual Model (A6) | 0.9583 |
| 2007 | Ensemble Val-Weighted (A5b) vs Residual Model (A6) | 0.9884 |
| 2008 | Phase 11 (M0) vs Ridge (A1) | 0.0569 |
| 2008 | Phase 11 (M0) vs Elastic Net (A2) | 0.0510 |
| 2008 | Phase 11 (M0) vs Huber (A3) | 0.0864 |
| 2008 | Phase 11 (M0) vs Random Forest (A4) | 0.8502 |
| 2008 | Phase 11 (M0) vs Ensemble Equal (A5a) | 0.7360 |
| 2008 | Phase 11 (M0) vs Ensemble Val-Weighted (A5b) | 1.0000 |
| 2008 | Phase 11 (M0) vs Residual Model (A6) | 0.8074 |
| 2008 | Ridge (A1) vs Elastic Net (A2) | 0.9971 |
| 2008 | Ridge (A1) vs Huber (A3) | 0.9842 |
| 2008 | Ridge (A1) vs Random Forest (A4) | 0.1892 |
| 2008 | Ridge (A1) vs Ensemble Equal (A5a) | 0.6922 |
| 2008 | Ridge (A1) vs Ensemble Val-Weighted (A5b) | 0.0569 |
| 2008 | Ridge (A1) vs Residual Model (A6) | -0.4544 |
| 2008 | Elastic Net (A2) vs Huber (A3) | 0.9699 |
| 2008 | Elastic Net (A2) vs Random Forest (A4) | 0.1818 |
| 2008 | Elastic Net (A2) vs Ensemble Equal (A5a) | 0.6855 |
| 2008 | Elastic Net (A2) vs Ensemble Val-Weighted (A5b) | 0.0510 |
| 2008 | Elastic Net (A2) vs Residual Model (A6) | -0.4795 |
| 2008 | Huber (A3) vs Random Forest (A4) | 0.2248 |
| 2008 | Huber (A3) vs Ensemble Equal (A5a) | 0.7072 |
| 2008 | Huber (A3) vs Ensemble Val-Weighted (A5b) | 0.0864 |
| 2008 | Huber (A3) vs Residual Model (A6) | -0.3815 |
| 2008 | Random Forest (A4) vs Ensemble Equal (A5a) | 0.8112 |
| 2008 | Random Forest (A4) vs Ensemble Val-Weighted (A5b) | 0.8502 |
| 2008 | Random Forest (A4) vs Residual Model (A6) | 0.6499 |
| 2008 | Ensemble Equal (A5a) vs Ensemble Val-Weighted (A5b) | 0.7360 |
| 2008 | Ensemble Equal (A5a) vs Residual Model (A6) | 0.2802 |
| 2008 | Ensemble Val-Weighted (A5b) vs Residual Model (A6) | 0.8074 |
| 2009 | Phase 11 (M0) vs Ridge (A1) | 0.6780 |
| 2009 | Phase 11 (M0) vs Elastic Net (A2) | 0.6056 |
| 2009 | Phase 11 (M0) vs Huber (A3) | 0.6850 |
| 2009 | Phase 11 (M0) vs Random Forest (A4) | 0.8391 |


## Interpretation
If two models are highly correlated, their errors are similar, and ensembling them
provides minimal benefit. Low correlation (complementary errors) is the condition
under which ensembles tend to improve over individual models.
