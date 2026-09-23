# Phase 21 — OOS Origin Registry

Generated: 2026-09-16T15:59:01.635412+00:00

## Frozen Origin List
Origins lock timestamp: `2026-09-16T15:59:02.103873+00:00`
Origins lock hash: `7d2bd47f4852410db41cfead05d39df3c1273a6103879b537b550883487328ef`

## Status
```
OOS_ORIGINS_AVAILABLE       = False
VALIDATED_OOS_ORIGINS       = []
OOS_OBSERVATION_COUNT       = 0
```

## All Current Origins in Dataset
```
[np.int64(2001), np.int64(2002), np.int64(2003), np.int64(2004), np.int64(2005), np.int64(2006), np.int64(2007), np.int64(2008), np.int64(2009), np.int64(2010), np.int64(2011), np.int64(2012), np.int64(2013), np.int64(2014), np.int64(2015), np.int64(2016), np.int64(2017), np.int64(2018), np.int64(2019), np.int64(2020), np.int64(2021), np.int64(2022), np.int64(2023), np.int64(2024)]
```

## Phase 19 Origins (Already Used — Not New OOS)
```
[2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
```

## Authenticity Checks (Country-Year Level)
Origins strictly after Phase 19 max (2024)
with valid `next_year_target_available == 1` rows: **[]**

Phase 21 operates at the **country × forecast_origin_year** observation level,
not at the aggregate-origin level. For each candidate origin, the count of
individual country rows with `next_year_target_available == 1` was verified.

## Governance Note
This registry was created in Stage C (after Stage A lock and Stage B discovery)
and before Stage D evaluation. The origin list cannot be modified after this
file is created.
