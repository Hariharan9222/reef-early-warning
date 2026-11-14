"""
src/synthetic.py
Documented, physics- and ecology-informed synthetic data generator for:
1. Satellite Sea Surface Temperature (SST, SSTA, DHW, Bleaching Alert Area)
2. Citizen Diver Bleaching Reports
3. Underwater Acoustic Soundscape audio clips (.wav) and metadata
"""

import os
import math
import numpy as np
import pandas as pd
from scipy.io import wavfile
from datetime import datetime, timedelta
from src.utils import get_logger, load_config, set_seed

logger = get_logger("synthetic_generator")

def generate_synthetic_sst(config: dict, output_dir: str = "data/raw") -> str:
    """
    Generates realistic daily satellite SST, SST Anomaly, and Degree Heating Weeks (DHW)
    for target coral regions across 2015-2024.
    Incorporates:
    - Regional baseline temperatures and seasonal cycles
    - Realistic historical marine heatwaves (e.g., 2016, 2020, 2024 global events)
    - 12-week accumulating Degree Heating Weeks (DHW)
    """
    set_seed(config.get("seed", 42))
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "synthetic_satellite_sst.csv")

    regions = config["regions"]
    start_year = config["dates"]["start_year"]
    end_year = config["dates"]["end_year"]

    date_range = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31",
        freq="D"
    )

    records = []

    for reg_key, reg_info in regions.items():
        name = reg_info["name"]
        lats = np.arange(reg_info["lat_min"], reg_info["lat_max"] + 0.01, reg_info["grid_res"] * 2) # Representative grid points
        lons = np.arange(reg_info["lon_min"], reg_info["lon_max"] + 0.01, reg_info["grid_res"] * 2)

        # Base climatology
        is_southern = (reg_info["lat_min"] + reg_info["lat_max"]) / 2.0 < 0
        baseline_temp = 27.2 if is_southern else 28.0
        mmm = baseline_temp + 1.2  # Maximum Monthly Mean

        # Generate for sampled grid cells in region
        # Sample 4 representative sites per region to keep clean yet realistic volume
        sample_lats = np.linspace(lats[0], lats[-1], min(3, len(lats)))
        sample_lons = np.linspace(lons[0], lons[-1], min(3, len(lons)))

        for lat in sample_lats:
            for lon in sample_lons:
                site_id = f"{reg_key}_{lat:.2f}_{lon:.2f}"
                
                # Precompute temperature series
                day_of_year = date_range.dayofyear.values
                years = date_range.year.values

                # Seasonality (Southern hemisphere peak in Feb/Mar, Northern in Aug/Sep)
                phase_shift = 60 if is_southern else 230
                seasonal_cycle = 2.2 * np.cos(2 * np.pi * (day_of_year - phase_shift) / 365.25)

                # Interannual heatwave anomalies (2016, 2017, 2020, 2024 known el nino / heatwaves)
                heatwave_factor = np.zeros(len(date_range))
                for idx, (yr, doy) in enumerate(zip(years, day_of_year)):
                    if yr in [2016, 2020, 2024] and (
                        (is_southern and 30 <= doy <= 110) or (not is_southern and 200 <= doy <= 280)
                    ):
                        heatwave_factor[idx] = 1.8 + np.sin(np.pi * (doy % 80) / 80) * 1.5
                    elif yr in [2017, 2023] and (
                        (is_southern and 45 <= doy <= 95) or (not is_southern and 210 <= doy <= 260)
                    ):
                        heatwave_factor[idx] = 1.1 + np.sin(np.pi * (doy % 50) / 50) * 1.0

                noise = np.random.normal(0, 0.35, len(date_range))
                sst = baseline_temp + seasonal_cycle + heatwave_factor + noise
                
                # SST anomaly relative to climatology mean
                ssta = sst - (baseline_temp + seasonal_cycle)

                # HotSpots: positive SST above MMM
                hotspots = np.maximum(0, sst - mmm)

                # Degree Heating Weeks: sum of HotSpots >= 1.0 over rolling 12 weeks (84 days) divided by 7
                # NOAA DHW definition: accumulated thermal stress where HotSpot >= 1 deg C
                qualifying_hotspot = np.where(hotspots >= 1.0, hotspots, 0.0)
                dhw_series = pd.Series(qualifying_hotspot).rolling(window=84, min_periods=1).sum() / 7.0
                dhw = dhw_series.values

                # Bleaching Alert Area
                # 0: No stress, 1: Bleaching Watch (HotSpot > 0), 2: Bleaching Warning (HotSpot > 0 & DHW > 0),
                # 3: Alert Level 1 (DHW >= 4), 4: Alert Level 2 (DHW >= 8)
                baa = np.zeros(len(date_range), dtype=int)
                baa[hotspots > 0] = 1
                baa[(hotspots > 0) & (dhw > 0)] = 2
                baa[dhw >= 4.0] = 3
                baa[dhw >= 8.0] = 4

                for i, dt in enumerate(date_range):
                    records.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "site_id": site_id,
                        "region": name,
                        "lat": round(float(lat), 3),
                        "lon": round(float(lon), 3),
                        "sst": round(float(sst[i]), 2),
                        "ssta": round(float(ssta[i]), 2),
                        "hotspot": round(float(hotspots[i]), 2),
                        "dhw": round(float(dhw[i]), 2),
                        "bleaching_alert_area": int(baa[i])
                    })

    df = pd.DataFrame(records)
    df.to_csv(out_file, index=False)
    logger.info(f"Synthetic SST data generated: {len(df)} rows across {df['site_id'].nunique()} sites saved to {out_file}")
    return out_file


def generate_synthetic_citizen_reports(config: dict, output_dir: str = "data/raw") -> str:
    """
    Generates synthetic citizen diver bleaching reports mimicking Reef Check & Global Bleaching Database.
    Standardizes categories to:
    0 = None (< 10% bleached)
    1 = Mild (10% - 50% bleached)
    2 = Severe (> 50% bleached / mortality)
    """
    set_seed(config.get("seed", 42) + 1)
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "synthetic_citizen_diver_reports.csv")

    regions = config["regions"]
    start_year = config["dates"]["start_year"]
    end_year = config["dates"]["end_year"]

    reports = []
    report_id = 1000

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    total_days = (end_date - start_date).days

    for reg_key, reg_info in regions.items():
        name = reg_info["name"]
        num_reports = 1800  # Diver reports per region over 10 years

        for _ in range(num_reports):
            day_offset = np.random.randint(0, total_days)
            dt = start_date + timedelta(days=day_offset)
            yr = dt.year
            month = dt.month

            # Clustered near reef sites with small GPS jitter
            center_lat = np.random.uniform(reg_info["lat_min"], reg_info["lat_max"])
            center_lon = np.random.uniform(reg_info["lon_min"], reg_info["lon_max"])
            lat = center_lat + np.random.normal(0, 0.05)
            lon = center_lon + np.random.normal(0, 0.05)

            depth_m = round(float(np.random.gamma(shape=3.0, scale=4.0)), 1) # typical recreational diving 5-30m
            depth_m = max(2.0, min(40.0, depth_m))

            is_southern = (reg_info["lat_min"] + reg_info["lat_max"]) / 2.0 < 0
            is_summer = (month in [1, 2, 3, 4]) if is_southern else (month in [7, 8, 9, 10])
            is_heatwave_yr = yr in [2016, 2017, 2020, 2023, 2024]

            # Bleaching likelihood higher in summer during heatwave years and at shallow depths
            bleaching_prob_severe = 0.05
            bleaching_prob_mild = 0.15

            if is_heatwave_yr and is_summer:
                bleaching_prob_severe += 0.45 * (1.0 - (depth_m / 60.0))
                bleaching_prob_mild += 0.30
            elif is_summer:
                bleaching_prob_mild += 0.20

            p_none = max(0.05, 1.0 - (bleaching_prob_mild + bleaching_prob_severe))
            p_mild = max(0.05, bleaching_prob_mild)
            p_severe = max(0.05, bleaching_prob_severe)
            total_p = p_none + p_mild + p_severe
            probs = [p_none / total_p, p_mild / total_p, p_severe / total_p]

            bleaching_class = np.random.choice([0, 1, 2], p=probs)
            if bleaching_class == 0:
                bleaching_pct = np.random.uniform(0.0, 9.9)
            elif bleaching_class == 1:
                bleaching_pct = np.random.uniform(10.0, 49.9)
            else:
                bleaching_pct = np.random.uniform(50.0, 100.0)

            observer_exp = np.random.choice(["Novice", "Intermediate", "Master Diver / Marine Scientist"], p=[0.3, 0.5, 0.2])
            confidence_map = {"Novice": 0.60, "Intermediate": 0.85, "Master Diver / Marine Scientist": 0.98}
            observer_confidence = round(float(confidence_map[observer_exp] + np.random.normal(0, 0.03)), 2)
            observer_confidence = max(0.4, min(1.0, observer_confidence))

            reports.append({
                "report_id": f"REP_{report_id}",
                "date": dt.strftime("%Y-%m-%d"),
                "region": name,
                "lat": round(float(lat), 4),
                "lon": round(float(lon), 4),
                "depth": depth_m,
                "bleaching_percent": round(float(bleaching_pct), 1),
                "bleaching_category": int(bleaching_class), # 0=None, 1=Mild, 2=Severe
                "observer_experience": observer_exp,
                "observer_confidence": observer_confidence
            })
            report_id += 1

    df = pd.DataFrame(reports)
    df.to_csv(out_file, index=False)
    logger.info(f"Synthetic citizen diver reports generated: {len(df)} reports saved to {out_file}")
    return out_file


def generate_synthetic_acoustics(config: dict, output_dir: str = "data/raw/audio") -> str:
    """
    Generates synthetic underwater acoustic .wav clips and a metadata CSV.
    Models distinct biophony:
    - Snapping shrimp broadband impulse trains (2-20 kHz)
    - Low-frequency fish choruses / pulses (50-1000 Hz)
    - Distinguishes healthy reefs (dense biophony) from degraded/bleached reefs (subdued biophony)
    """
    set_seed(config.get("seed", 42) + 2)
    os.makedirs(output_dir, exist_ok=True)
    meta_file = os.path.join("data/raw", "synthetic_acoustics_metadata.csv")

    sr = config["acoustics"]["sample_rate"]
    duration = config["acoustics"]["clip_duration_sec"]
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    regions = config["regions"]
    start_year = config["dates"]["start_year"]
    end_year = config["dates"]["end_year"]

    metadata = []
    clip_counter = 1

    # Sites per region
    for reg_key, reg_info in regions.items():
        name = reg_info["name"]
        sample_lats = np.linspace(reg_info["lat_min"], reg_info["lat_max"], 3)
        sample_lons = np.linspace(reg_info["lon_min"], reg_info["lon_max"], 3)

        for s_idx, (lat, lon) in enumerate(zip(sample_lats, sample_lons)):
            site_id = f"{reg_key}_{lat:.2f}_{lon:.2f}"
            
            # Create a set of periodic acoustic samples (e.g. quarterly surveys across years)
            survey_years = list(range(start_year, end_year + 1))
            for yr in survey_years:
                for month in [3, 9]: # wet/dry or summer/winter acoustic monitoring
                    dt = datetime(yr, month, 15)
                    
                    # Determine ecological status based on heatwave history
                    is_southern = (reg_info["lat_min"] + reg_info["lat_max"]) / 2.0 < 0
                    is_bleaching_epoch = (yr in [2016, 2020, 2024]) and ((is_southern and month == 3) or (not is_southern and month == 9))
                    
                    if is_bleaching_epoch:
                        status = "severe_bleach"
                        label = 2
                        shrimp_rate = 20    # snaps per sec (subdued)
                        shrimp_amp = 0.15
                        fish_amp = 0.05
                    elif yr in [2017, 2023]:
                        status = "mild_bleach"
                        label = 1
                        shrimp_rate = 70
                        shrimp_amp = 0.35
                        fish_amp = 0.20
                    else:
                        status = "healthy"
                        label = 0
                        shrimp_rate = 180   # dense snapping shrimp activity
                        shrimp_amp = 0.80
                        fish_amp = 0.50

                    # 1. Background water flow / ambient hydrophone noise (pink-ish / brown noise)
                    noise = np.random.normal(0, 0.03, n_samples)

                    # 2. Snapping shrimp clicks: high frequency decaying sinusoids (2-10 kHz)
                    n_snaps = int(shrimp_rate * duration)
                    snap_signal = np.zeros(n_samples)
                    snap_times = np.random.randint(0, n_samples - 500, size=n_snaps)
                    snap_t = np.linspace(0, 0.005, int(0.005 * sr), endpoint=False) # 5ms snap
                    for st in snap_times:
                        freq = np.random.uniform(3000, 9000)
                        burst = np.sin(2 * np.pi * freq * snap_t) * np.exp(-snap_t / 0.001)
                        snap_signal[st:st+len(burst)] += burst * shrimp_amp * np.random.uniform(0.6, 1.2)

                    # 3. Fish choruses / biological grunts: low frequency harmonic tones (100-500 Hz)
                    fish_signal = np.zeros(n_samples)
                    if fish_amp > 0:
                        f_pulse_times = np.random.randint(0, n_samples - int(0.3 * sr), size=int(duration * 2))
                        pulse_t = np.linspace(0, 0.25, int(0.25 * sr), endpoint=False) # 250ms grunt
                        for pt in f_pulse_times:
                            f_fund = np.random.uniform(120, 280)
                            grunt = (
                                np.sin(2 * np.pi * f_fund * pulse_t) +
                                0.5 * np.sin(2 * np.pi * 2 * f_fund * pulse_t) +
                                0.2 * np.sin(2 * np.pi * 3 * f_fund * pulse_t)
                            ) * np.hanning(len(pulse_t))
                            fish_signal[pt:pt+len(grunt)] += grunt * fish_amp

                    audio = noise + snap_signal + fish_signal
                    # Normalize to [-1.0, 1.0] and convert to int16
                    max_abs = np.max(np.abs(audio)) + 1e-6
                    audio_norm = (audio / max_abs * 0.95 * 32767).astype(np.int16)

                    filename = f"clip_{clip_counter:05d}_{status}.wav"
                    filepath = os.path.join(output_dir, filename)
                    wavfile.write(filepath, sr, audio_norm)

                    metadata.append({
                        "clip_id": f"CLIP_{clip_counter:05d}",
                        "file_path": filepath.replace("\\", "/"),
                        "site_id": site_id,
                        "region": name,
                        "lat": round(float(lat), 3),
                        "lon": round(float(lon), 3),
                        "date": dt.strftime("%Y-%m-%d"),
                        "status": status,
                        "label": label,
                        "shrimp_activity_rate": shrimp_rate,
                        "fish_chorus_amp": fish_amp
                    })
                    clip_counter += 1

    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(meta_file, index=False)
    logger.info(f"Generated {len(meta_df)} acoustic clips with metadata at {meta_file}")
    return meta_file

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    generate_synthetic_sst(cfg)
    generate_synthetic_citizen_reports(cfg)
    generate_synthetic_acoustics(cfg)

