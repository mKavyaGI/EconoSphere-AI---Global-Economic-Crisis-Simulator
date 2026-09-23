import pandas as pd
import numpy as np
import hashlib
from pathlib import Path

root = Path('.')
raw_path = root / 'data/raw/master_panel.csv'
proc_path = root / 'data/processed/master_panel_processed.csv'
fc_path = root / 'data/processed/master_panel_forecasting.csv'

# 1. Raw Integrity
raw_md5 = hashlib.md5(raw_path.read_bytes()).hexdigest()
print('RAW MD5:', raw_md5, 'MATCH:', raw_md5 == '8ac7e0b2bf09fbe89289f82d0c7cf25e')

# 2. Preprocessing / Schemas
raw_df = pd.read_csv(raw_path)
proc_df = pd.read_csv(proc_path)
fc_df = pd.read_csv(fc_path)

print('\nSCHEMAS:')
print('Raw:', raw_df.shape)
print('Proc:', proc_df.shape)
print('FC:', fc_df.shape)

# 3. Country / Panel Integrity
print('\nPANEL INTEGRITY (FC Data):')
dups = fc_df.duplicated(['country_code', 'year']).sum()
print('Dups (code, year):', dups)
print('Is Sorted?', fc_df[['country_code', 'year']].equals(fc_df[['country_code', 'year']].sort_values(['country_code', 'year'])))

codes = ['IND', 'CHN', 'USA', 'JPN', 'GBR']
for code in codes:
    sub = fc_df[fc_df['country_code'] == code]
    if sub.empty: continue
    min_yr = sub['year'].min()
    max_yr = sub['year'].max()
    tgt_nans = sub['gdp_growth_next_year'].isna().sum()
    print(f'{code}: {min_yr}-{max_yr}, rows: {len(sub)}, tgt_nan: {tgt_nans}')

# 4. Feature Eng / Lags (Check IND 2010 lag1 == 2009 actual)
print('\nFEATURE ENG (Lags):')
ind = fc_df[fc_df['country_code'] == 'IND']
ind_2009 = ind[ind['year'] == 2009]['gdp_growth_pct'].values[0]
ind_2010_lag1 = ind[ind['year'] == 2010]['gdp_growth_lag1'].values[0]
print(f'IND 2009 GDP: {ind_2009}, 2010 lag1: {ind_2010_lag1}, MATCH: {np.isclose(ind_2009, ind_2010_lag1)}')

# 5. Rolling Features
print('\nROLLING (IND 2010 rolling_mean_3):')
ind_2007 = ind[ind['year'] == 2007]['gdp_growth_pct'].values[0]
ind_2008 = ind[ind['year'] == 2008]['gdp_growth_pct'].values[0]
print(f'IND 2007-2009 mean: {np.mean([ind_2007, ind_2008, ind_2009])}, 2010 rolling_mean_3: {ind[ind["year"] == 2010]["gdp_growth_rolling_mean_3"].values[0]}')

# 6. Target T+1 check
print('\nTARGET T+1 CHECK:')
errors = 0
for code, group in fc_df.groupby('country_code'):
    shifted = group['gdp_growth_pct'].shift(-1)
    mask = group['gdp_growth_next_year'].notna() & shifted.notna()
    if not np.allclose(group.loc[mask, 'gdp_growth_next_year'], shifted[mask]):
        errors += 1
print('T+1 errors across all countries:', errors)

# 7. 2025 Behavior
print('\n2025 BEHAVIOR:')
y2025 = fc_df[fc_df['year'] == 2025]
print('2025 rows missing target:', y2025['gdp_growth_next_year'].isna().all())

# 8. Examples
print('\nEXAMPLES:')
for code in codes:
    sub = fc_df[(fc_df['country_code'] == code) & (fc_df['year'] >= 2023)][['country_code', 'year', 'gdp_growth_pct', 'gdp_growth_next_year']]
    print(sub.to_string(index=False))
