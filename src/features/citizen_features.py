"""
src/features/citizen_features.py
Computes citizen diver features per reef site and date:
- Reports within 50 km in the past 30 days
- Weighted mean bleaching severity (weights by observer confidence)
- Report density
- Spatial neighbor severity (KNN / Haversine)
"""

import numpy as np
import pandas as pd
from src.utils import get_logger

logger = get_logger("citizen_features")

def haversine_np(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vectorized Haversine distance in kilometers from a point (lat1, lon1) to arrays (lats, lons)."""
    R = 6371.0 # Earth radius in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lats)
    delta_phi = np.radians(lats - lat1)
    delta_lambda = np.radians(lons - lon1)

    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return R * c

def compute_citizen_features_for_sites(
    df_sites: pd.DataFrame,
    df_citizen: pd.DataFrame,
    search_radius_km: float = 50.0,
    window_days: int = 30
) -> pd.DataFrame:
    """
    Computes spatial-temporal citizen report features for site-date observation points.
    df_sites must have ['date', 'site_id', 'lat', 'lon'].
    df_citizen must have ['date', 'lat', 'lon', 'bleaching_category', 'observer_confidence'].
    """
    df_c = df_citizen.copy()
    df_c["date"] = pd.to_datetime(df_c["date"])
    df_s = df_sites.copy()
    df_s["date"] = pd.to_datetime(df_s["date"])

    c_lats = df_c["lat"].values
    c_lons = df_c["lon"].values
    c_dates = df_c["date"].values
    c_sevs = df_c["bleaching_category"].values.astype(float)
    c_confs = df_c["observer_confidence"].values.astype(float)

    area_km2 = np.pi * (search_radius_km ** 2)

    results = []
    logger.info(f"Computing citizen diver features for {len(df_s)} site-date instances...")

    # Group by site to avoid recomputing lat/lon distances for every single date
    for site_id, site_group in df_s.groupby("site_id"):
        site_lat = site_group["lat"].iloc[0]
        site_lon = site_group["lon"].iloc[0]

        # Distances to all citizen reports
        distances = haversine_np(site_lat, site_lon, c_lats, c_lons)
        within_radius = distances <= search_radius_km
        within_150km = distances <= 150.0

        for _, row in site_group.iterrows():
            current_date = row["date"]
            start_date = current_date - pd.Timedelta(days=window_days)

            # Filter by time window [t - 30d, t]
            time_mask = (c_dates >= np.datetime64(start_date)) & (c_dates <= np.datetime64(current_date))
            active_mask = within_radius & time_mask

            count = int(np.sum(active_mask))
            if count > 0:
                sevs = c_sevs[active_mask]
                confs = c_confs[active_mask]
                weighted_sev = float(np.sum(sevs * confs) / (np.sum(confs) + 1e-8))
                mean_conf = float(np.mean(confs))
                density = float(count / area_km2)
            else:
                weighted_sev = 0.0
                mean_conf = 0.0
                density = 0.0

            # Spatial neighbor severity (KNN: closest 3 reports in 150km window)
            knn_mask = within_150km & time_mask
            if np.sum(knn_mask) > 0:
                knn_indices = np.argsort(distances[knn_mask])[:3]
                knn_sev = float(np.mean(c_sevs[knn_mask][knn_indices]))
            else:
                knn_sev = 0.0

            results.append({
                "site_id": site_id,
                "date": current_date,
                "diver_reports_count_50km_30d": count,
                "diver_report_density": density,
                "diver_weighted_severity_30d": weighted_sev,
                "diver_mean_confidence_30d": mean_conf,
                "diver_spatial_neighbor_severity_knn": knn_sev,
                "diver_has_reports": int(count > 0)
            })

    out_df = pd.DataFrame(results)
    return out_df

if __name__ == "__main__":
    import os
    from src.utils import load_config
    cfg = load_config("config.yaml")
    cit_p = os.path.join(cfg["paths"]["processed_dir"], "citizen_processed.parquet")
    if os.path.exists(cit_p):
        df_cit = pd.read_parquet(cit_p)
        print(f"Citizen reports available: {len(df_cit)} records")
    else:
        logger.warning(f"File {cit_p} not found. Run ingestion first.")

