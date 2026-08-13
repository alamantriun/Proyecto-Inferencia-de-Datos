"""
Cliente SODA reutilizable para datos.gov.co
============================================
Paginación automática, reintentos y delays para respetar rate limits.
"""
import time
import requests
import pandas as pd
from typing import Optional


class SodaClient:
    BASE_URL = "https://www.datos.gov.co/resource"

    def __init__(self, dataset_id: str, app_token: Optional[str] = None):
        self.endpoint = f"{self.BASE_URL}/{dataset_id}.json"
        self.headers = {"X-App-Token": app_token} if app_token else {}

    def query(self, select="*", where=None, order=None, group=None,
              limit=50000, offset=0) -> list:
        params = {"$select": select, "$limit": limit, "$offset": offset}
        if where:
            params["$where"] = where
        if order:
            params["$order"] = order
        if group:
            params["$group"] = group

        for attempt in range(3):
            try:
                r = requests.get(self.endpoint, params=params,
                                 headers=self.headers, timeout=60)
                r.raise_for_status()
                return r.json()
            except requests.exceptions.Timeout:
                print(f"  ⚠ Timeout (intento {attempt+1}/3)…")
                time.sleep(10 * (attempt + 1))
            except requests.exceptions.HTTPError as e:
                if r.status_code == 429:
                    print("  ⚠ Rate limit — esperando 30s…")
                    time.sleep(30)
                else:
                    raise e
        raise RuntimeError(f"Falló 3 intentos: {self.endpoint}")

    def extract_all(self, where=None, order=":id", group=None,
                    batch=50000, max_records=None,
                    select="*") -> pd.DataFrame:
        rows, offset = [], 0
        while True:
            lim = batch
            if max_records:
                rem = max_records - len(rows)
                if rem <= 0:
                    break
                lim = min(batch, rem)

            print(f"  → offset {offset:,} …", end=" ", flush=True)
            batch_data = self.query(select=select, where=where, group=group,
                                    order=order, limit=lim, offset=offset)
            if not batch_data:
                print("fin.")
                break

            rows.extend(batch_data)
            offset += len(batch_data)
            print(f"{len(batch_data):,} registros  (total {len(rows):,})")
            time.sleep(0.3)

            if len(batch_data) < lim:
                break

        return pd.DataFrame(rows) if rows else pd.DataFrame()
