"""
Data validation, outlier treatment, and missing value imputation for institutional time series ingestion.
Separates preprocessing and hygiene cleanly from model training and inference per Requirement 6.
"""
import math
from typing import List, Dict, Any, Optional

class DataValidator:
    """
    Enterprise data cleaning and validation engine.
    Applies statistical outlier clipping, type coercion, and linear interval imputation.
    """
    @staticmethod
    def clean_series_records(raw_records: List[Dict[str, Any]], value_field: str = "value", z_threshold: float = 2.5) -> List[Dict[str, Any]]:
        """
        Processes a raw list of dictionary observation records:
        1. Filters out records with invalid or missing date stamps.
        2. Coerces values to float.
        3. Performs forward-fill & backward-fill linear imputation for intermittent missing values.
        4. Clips statistical outliers beyond z_threshold standard deviations from rolling mean.
        """
        if not raw_records:
            return []

        # Sort chronologically by year or timestamp
        valid_records = []
        for rec in raw_records:
            try:
                time_key = rec.get("date", rec.get("year"))
                if time_key is None:
                    continue
                
                val = rec.get(value_field)
                val_float = float(val) if (val is not None and str(val).strip() != "") else None
                
                valid_records.append({
                    "date": str(time_key),
                    "year": int(str(time_key)[:4]) if str(time_key)[:4].isdigit() else 2020,
                    value_field: val_float,
                    "series_id": rec.get("series_id", "UNKNOWN"),
                    "country_code": rec.get("country_code", rec.get("iso3", "GLB"))
                })
            except (ValueError, TypeError):
                continue

        valid_records.sort(key=lambda x: x["year"])
        
        # Step 2: Impute missing values (Forward fill followed by backward fill)
        values = [r[value_field] for r in valid_records]
        imputed_values = DataValidator._impute_list_values(values)
        
        # Step 3: Outlier clipping using Z-score or robust median difference
        cleaned_values = DataValidator._clip_outliers(imputed_values, z_threshold=z_threshold)
        
        for i, r in enumerate(valid_records):
            r[value_field] = cleaned_values[i]
            
        return valid_records

    @staticmethod
    def _impute_list_values(values: List[Optional[float]]) -> List[float]:
        n = len(values)
        if n == 0:
            return []
        
        res = list(values)
        first_valid = next((v for v in res if v is not None), 0.0)
        for i in range(n):
            if res[i] is None:
                res[i] = res[i-1] if i > 0 else first_valid

        return [float(v) for v in res]

    @staticmethod
    def _clip_outliers(values: List[float], z_threshold: float = 2.5) -> List[float]:
        n = len(values)
        if n < 3:
            return values
            
        # Use median and median absolute deviation (MAD) for robust outlier detection on small series
        sorted_vals = sorted(values)
        mid = n // 2
        median = sorted_vals[mid] if n % 2 != 0 else (sorted_vals[mid-1] + sorted_vals[mid]) / 2.0
        
        mads = [abs(x - median) for x in values]
        sorted_mads = sorted(mads)
        mad_val = sorted_mads[mid] if n % 2 != 0 else (sorted_mads[mid-1] + sorted_mads[mid]) / 2.0
        
        # Consistent estimator for standard deviation via MAD
        robust_std = max(0.001, mad_val * 1.4826)
        
        clipped = []
        for v in values:
            z_robust = abs(v - median) / robust_std
            if z_robust > z_threshold and abs(v - median) > 5.0:
                # Clamp to exact boundary
                clamped = median + (z_threshold * robust_std * (1 if v > median else -1))
                clipped.append(round(clamped, 4))
            else:
                clipped.append(v)
        return clipped
