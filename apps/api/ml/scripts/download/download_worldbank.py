#!/usr/bin/env python3
"""Script to download macroeconomic indicator data from the World Bank API into raw dataset storage."""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union
import urllib.request
import urllib.parse
import urllib.error

# Configure standard logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent.parent / "datasets" / "raw"
WORLD_BANK_API_BASE = "https://api.worldbank.org/v2/country"

# Standard World Development Indicator defaults (GDP, Inflation, Unemployment, Trade % of GDP)
DEFAULT_INDICATORS = [
    "NY.GDP.MKTP.CD",
    "FP.CPI.TOTL.ZG",
    "SL.UEM.TOTL.ZS",
    "NE.TRD.GNFS.ZS",
]


def fetch_worldbank_indicator(
    country_code: str,
    indicator_code: str,
    start_year: int,
    end_year: int,
    timeout: int = 30,
) -> Union[list[Any], dict[str, Any]]:
    """Fetches historical time-series observations for a specific World Bank indicator and country.

    Args:
        country_code: Three-letter ISO country code or 'all' for global aggregations.
        indicator_code: World Bank indicator identifier (e.g., 'NY.GDP.MKTP.CD').
        start_year: Start year for date filtering.
        end_year: End year for date filtering.
        timeout: Socket request timeout in seconds.

    Returns:
        A list or dictionary containing the raw parsed JSON response from the World Bank API.

    Raises:
        RuntimeError: If HTTP communication fails or the response payload is unreadable.
    """
    params = {
        "format": "json",
        "date": f"{start_year}:{end_year}",
        "per_page": 1000,
    }
    query_str = urllib.parse.urlencode(params)
    url = f"{WORLD_BANK_API_BASE}/{country_code}/indicator/{indicator_code}?{query_str}"
    
    logger.info(f"Requesting World Bank API: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EconoSphere-AI-Downloader/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"Received non-200 HTTP status: {response.status}")
            data = json.loads(response.read().decode("utf-8"))
            return data
    except urllib.error.URLError as err:
        raise RuntimeError(f"Network error accessing World Bank API ({url}): {err}") from err
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Failed to parse JSON response from World Bank API: {err}") from err


def save_raw_dataset(
    data: Union[list[Any], dict[str, Any]],
    country_code: str,
    indicator_code: str,
    output_dir: Union[str, Path],
) -> Path:
    """Persists downloaded raw data into an immutable JSON artifact file in the destination directory.

    Args:
        data: Parsed JSON content returned from the API.
        country_code: ISO country code used in the download query.
        indicator_code: Indicator code used in the download query.
        output_dir: Filesystem path to the targeted dataset raw folder.

    Returns:
        The validated Path object of the created dataset raw file.
    """
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    clean_indicator = indicator_code.replace(".", "_").lower()
    filename = f"worldbank_{country_code.lower()}_{clean_indicator}_{timestamp}.raw.json"
    file_path = dest_dir / filename

    with open(file_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, sort_keys=True)

    logger.info(f"Saved raw World Bank dataset artifact to: {file_path}")
    return file_path


def main() -> None:
    """Main CLI execution handler for downloading World Bank macroeconomic data."""
    parser = argparse.ArgumentParser(description="Download raw macroeconomic indicators from the World Bank API.")
    parser.add_argument(
        "--country",
        type=str,
        default="WLD",
        help="Target country ISO3 code or 'all' (default: WLD for World aggregate).",
    )
    parser.add_argument(
        "--indicators",
        nargs="+",
        default=DEFAULT_INDICATORS,
        help=f"List of World Bank indicator codes to retrieve (default: {DEFAULT_INDICATORS}).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2000,
        help="Earliest historical year to query (default: 2000).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=datetime.now().year,
        help="Latest historical year to query (default: current year).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Destination directory for raw dataset artifact retention (default: {DEFAULT_RAW_DIR}).",
    )

    args = parser.parse_args()

    logger.info(f"Starting World Bank data download for country '{args.country}' across {len(args.indicators)} indicators.")
    success_count = 0

    for indicator in args.indicators:
        try:
            raw_data = fetch_worldbank_indicator(
                country_code=args.country,
                indicator_code=indicator,
                start_year=args.start_year,
                end_year=args.end_year,
            )
            save_raw_dataset(
                data=raw_data,
                country_code=args.country,
                indicator_code=indicator,
                output_dir=args.output_dir,
            )
            success_count += 1
        except Exception as err:
            logger.error(f"Failed to download indicator '{indicator}': {err}")

    logger.info(f"Download routine complete. Successfully retrieved {success_count} of {len(args.indicators)} requested indicators.")


if __name__ == "__main__":
    main()
