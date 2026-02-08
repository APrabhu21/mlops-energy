"""
Client for EIA API v2 — fetches hourly electricity demand data.
Handles pagination (5000 row limit per request) and rate limiting.
"""
import time
import requests
import pandas as pd
from datetime import datetime
from typing import Optional
from src.config import EIA_API_KEY, EIA_BASE_URL, EIA_REGION


class EIAClient:
    ENDPOINT = "electricity/rto/region-data/data/"
    MAX_LENGTH = 5000
    RATE_LIMIT_DELAY = 1.0  # seconds between requests

    def __init__(self, api_key: str = EIA_API_KEY, region: str = EIA_REGION):
        if not api_key:
            raise ValueError(
                "EIA_API_KEY not provided. "
                "Register for a free API key at https://www.eia.gov/opendata/ "
                "and set it in your .env file or pass it to EIAClient(api_key='...')."
            )
        self.api_key = api_key
        self.region = region
        self.base_url = EIA_BASE_URL + self.ENDPOINT

    def fetch_demand(
        self,
        start: str,   # format: "2023-01-01T00"
        end: str,      # format: "2025-12-31T23"
    ) -> pd.DataFrame:
        """
        Fetch all hourly demand data between start and end.
        Handles pagination automatically.
        Returns DataFrame with columns: [period, respondent, demand_mwh]
        """
        all_records = []
        offset = 0

        while True:
            params = {
                "api_key": self.api_key,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": self.region,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": offset,
                "length": self.MAX_LENGTH,
                "start": start,
                "end": end,
            }

            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()["response"]["data"]
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data from EIA API: {e}")
                raise
            except KeyError as e:
                print(f"Unexpected API response format: {e}")
                print(f"Response: {response.text[:500]}")
                raise

            if not data:
                break

            all_records.extend(data)
            offset += self.MAX_LENGTH

            if len(data) < self.MAX_LENGTH:
                break

            time.sleep(self.RATE_LIMIT_DELAY)

        df = pd.DataFrame(all_records)
        if not df.empty:
            df["period"] = pd.to_datetime(df["period"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.rename(columns={"value": "demand_mwh"})
        return df

    def fetch_incremental(self, last_timestamp: datetime) -> pd.DataFrame:
        """Fetch only new data since last_timestamp."""
        start = last_timestamp.strftime("%Y-%m-%dT%H")
        end = datetime.utcnow().strftime("%Y-%m-%dT%H")
        return self.fetch_demand(start=start, end=end)
