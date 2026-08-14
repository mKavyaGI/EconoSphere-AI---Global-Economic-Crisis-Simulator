#!/usr/bin/env python3
"""Script to download bilateral trade flows and global exchange records into raw dataset storage."""

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

# UN Comtrade API Public REST Endpoint (or comparable standard bilateral trade repository)
COMTRADE_API_BASE = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"


def fetch_trade_data(
    reporter_code: str,
    partner_code: str,
    period: str,
    commodity_code: str = "TOTAL",
    timeout: int = 45,
) -> dict[str, Any]:
    """Retrieves international trade flows (imports/exports) between reporter and partner entities.

    Args:
        reporter_code: Reporting jurisdiction numeric or ISO code (e.g., '842' or 'USA').
        partner_code: Partner jurisdiction code (e.g., '0' for World total or specific partner).
        period: Four-digit annual reporting year or YYYYMM period string.
        commodity_code: HS classification code or 'TOTAL' for aggregate merchandise trade.
        timeout: Socket request timeout in seconds.

    Returns:
        A dictionary containing the parsed raw bilateral trade dataset records.

    Raises:
        RuntimeError: If trade endpoint requests fail or responses are invalid JSON.
    """
    params = {
        "reporterCode": reporter_code,
        "partnerCode": partner_code,
        "period": period,
        "cmdCode": commodity_code,
    }
    url_params = urllib.parse.urlencode(params)
    url = f"{COMTRADE_API_BASE}?{url_params}"

    logger.info(f"Requesting international trade data endpoint: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EconoSphere-AI-Downloader/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"Received non-200 HTTP status from trade server: {response.status}")
            data = json.loads(response.read().decode("utf-8"))
            return data
    except urllib.error.URLError as err:
        raise RuntimeError(f"Network transmission failure while querying trade API ({url}): {err}") from err
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Failed to decode trade dataset JSON response payload: {err}") from err


def save_trade_raw_dataset(
    data: dict[str, Any],
    reporter: str,
    partner: str,
    period: str,
    commodity: str,
    output_dir: Union[str, Path],
) -> Path:
    """Writes raw bilateral trade responses to disk as immutable timestamped data artifacts.

    Args:
        data: Parsed API trade record response dictionary.
        reporter: Reporter identifier string used in query.
        partner: Partner identifier string used in query.
        period: Target observation period string.
        commodity: Target commodity classification code.
        output_dir: Filesystem path to the dataset raw retention folder.

    Returns:
        The verified Path object pointing to the newly generated raw file.
    """
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"trade_{reporter}_{partner}_{period}_{commodity.lower()}_{timestamp}.raw.json"
    file_path = dest_dir / filename

    with open(file_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, sort_keys=True)

    logger.info(f"Saved raw trade dataset artifact to: {file_path}")
    return file_path


def main() -> None:
    """Main CLI execution handler for retrieving international trade datasets."""
    parser = argparse.ArgumentParser(description="Download bilateral international trade data flows into raw storage.")
    parser.add_argument(
        "--reporter",
        type=str,
        default="842",  # Default UN Comtrade numeric code for USA (842)
        help="Reporting country identifier code (default: 842 for USA).",
    )
    parser.add_argument(
        "--partner",
        type=str,
        default="0",  # 0 usually symbolizes Total World Aggregate
        help="Partner country identifier code (default: 0 for World total).",
    )
    parser.add_argument(
        "--periods",
        nargs="+",
        default=[str(datetime.now().year - 1)],
        help="List of observation years or YYYYMM periods to query (default: previous year).",
    )
    parser.add_argument(
        "--commodity",
        type=str,
        default="TOTAL",
        help="Harmonized System (HS) commodity code or 'TOTAL' (default: TOTAL).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Destination directory for raw trade dataset storage (default: {DEFAULT_RAW_DIR}).",
    )

    args = parser.parse_args()

    logger.info(f"Starting trade flow data download for reporter '{args.reporter}' and partner '{args.partner}'.")
    success_count = 0

    for period in args.periods:
        try:
            raw_trade = fetch_trade_data(
                reporter_code=args.reporter,
                partner_code=args.partner,
                period=period,
                commodity_code=args.commodity,
            )
            save_trade_raw_dataset(
                data=raw_trade,
                reporter=args.reporter,
                partner=args.partner,
                period=period,
                commodity=args.commodity,
                output_dir=args.output_dir,
            )
            success_count += 1
        except Exception as err:
            logger.error(f"Failed to retrieve trade flows for period '{period}': {err}")

    logger.info(f"Trade flow download completed. Successfully collected {success_count} of {len(args.periods)} queried periods.")


if __name__ == "__main__":
    main()
