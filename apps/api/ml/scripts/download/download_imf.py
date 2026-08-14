#!/usr/bin/env python3
"""Script to download international financial statistics and economic time-series from the IMF API."""

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
IMF_API_BASE = "http://dataservices.imf.org/REST/SDMX_JSON.svc"


def fetch_imf_compact_data(
    dataset_id: str,
    frequency: str,
    country_code: str,
    indicator: str,
    start_period: int,
    end_period: int,
    timeout: int = 45,
) -> dict[str, Any]:
    """Retrieves compact statistical observations from the IMF SDMX RESTful JSON API.

    Args:
        dataset_id: IMF statistical dataset code (e.g., 'IFS' for International Financial Statistics).
        frequency: Time-series frequency ('A' for annual, 'Q' for quarterly, 'M' for monthly).
        country_code: ISO or IMF geographic area identifier.
        indicator: IMF indicator code (e.g., 'ENDA_XDC_USD_RATE' for Exchange Rate to USD).
        start_period: Initial observation year or period.
        end_period: Final observation year or period.
        timeout: Request socket timeout in seconds.

    Returns:
        A dictionary representation of the SDMX structured JSON payload returned by the IMF server.

    Raises:
        RuntimeError: If communication issues occur or server returns error statuses.
    """
    query_path = f"CompactData/{dataset_id}/{frequency}.{country_code}.{indicator}"
    params = {
        "startPeriod": str(start_period),
        "endPeriod": str(end_period),
    }
    url_params = urllib.parse.urlencode(params)
    url = f"{IMF_API_BASE}/{query_path}?{url_params}"

    logger.info(f"Requesting IMF SDMX REST API: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EconoSphere-AI-Downloader/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"Received non-200 HTTP status from IMF API: {response.status}")
            payload = json.loads(response.read().decode("utf-8"))
            return payload
    except urllib.error.URLError as err:
        raise RuntimeError(f"Network failure while reaching IMF SDMX service ({url}): {err}") from err
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Unable to decode SDMX JSON response from IMF API: {err}") from err


def save_imf_raw_dataset(
    data: dict[str, Any],
    dataset_id: str,
    country_code: str,
    indicator: str,
    output_dir: Union[str, Path],
) -> Path:
    """Saves downloaded IMF raw dataset payload as an immutable file artifact in storage.

    Args:
        data: SDMX JSON response dictionary from the IMF.
        dataset_id: IMF database identifier.
        country_code: Target country or regional code.
        indicator: Queried economic indicator code.
        output_dir: Filesystem path to the targeted dataset raw folder.

    Returns:
        The validated Path object of the saved artifact file.
    """
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    clean_indicator = indicator.replace("/", "_").lower()
    filename = f"imf_{dataset_id.lower()}_{country_code.lower()}_{clean_indicator}_{timestamp}.raw.json"
    file_path = dest_dir / filename

    with open(file_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, sort_keys=True)

    logger.info(f"Saved raw IMF dataset artifact to: {file_path}")
    return file_path


def main() -> None:
    """Main CLI execution handler for downloading IMF economic statistics."""
    parser = argparse.ArgumentParser(description="Download historical economic time-series from the IMF SDMX API.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="IFS",
        help="IMF statistical dataset identifier (default: IFS - International Financial Statistics).",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="A",
        choices=["A", "Q", "M"],
        help="Observation temporal frequency: A (Annual), Q (Quarterly), or M (Monthly) (default: A).",
    )
    parser.add_argument(
        "--country",
        type=str,
        default="US",
        help="Target geographic region or country code (default: US).",
    )
    parser.add_argument(
        "--indicators",
        nargs="+",
        default=["ENDA_XDC_USD_RATE", "PCPI_IX"],
        help="List of IMF indicators to fetch (default: Exchange rate and CPI index).",
    )
    parser.add_argument(
        "--start-period",
        type=int,
        default=2000,
        help="Initial observation year (default: 2000).",
    )
    parser.add_argument(
        "--end-period",
        type=int,
        default=datetime.now().year,
        help="Final observation year (default: current year).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Destination directory for raw dataset persistence (default: {DEFAULT_RAW_DIR}).",
    )

    args = parser.parse_args()

    logger.info(f"Starting IMF download from dataset '{args.dataset}' for area '{args.country}'.")
    success_count = 0

    for indicator in args.indicators:
        try:
            raw_data = fetch_imf_compact_data(
                dataset_id=args.dataset,
                frequency=args.frequency,
                country_code=args.country,
                indicator=indicator,
                start_period=args.start_period,
                end_period=args.end_period,
            )
            save_imf_raw_dataset(
                data=raw_data,
                dataset_id=args.dataset,
                country_code=args.country,
                indicator=indicator,
                output_dir=args.output_dir,
            )
            success_count += 1
        except Exception as err:
            logger.error(f"Failed to fetch IMF indicator '{indicator}': {err}")

    logger.info(f"IMF download completed. Retrieved {success_count} of {len(args.indicators)} requested indicators.")


if __name__ == "__main__":
    main()
