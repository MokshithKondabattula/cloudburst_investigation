"""
  MOSDAC:
    - X (1618,) = Mercator longitude in METRES, central meridian lon0=82°E
    - Y (1616,) = Mercator latitude  in METRES, S→N (row 0 = southernmost)
    - Images: uint16 raw counts → convert via LUT (e.g. IMG_TIR1_TEMP array)
    - Geometry arrays: centidegrees stored as uint16/int16 → divide by 100
    - India crop: rows 431:1410, cols 363:1254  (6–38°N, 66–98°E)

  IMERG V07B Final Run:
    - Shape: (1, 3600, 1800) = (time, lon, lat)  ← lon axis is FIRST
    - Field: Grid/precipitation
    - lat array: -89.95 to 89.95, step 0.1°
    - lon array: -179.95 to 179.95, step 0.1°
    - India crop: lon indices 2459:2779, lat indices 959:1279  → transpose

"""

import os
import math
import numpy as np
import h5py
from tqdm import tqdm


# ── Settings ──────────────────────────────────────────────────────────────────
DATA_ROOT = r"F:\\"
N_EVENTS  = 20

# MOSDAC India crop pixel indices (verified via Mercator projection math)
M_R0, M_R1 = 431, 1410   # rows  → 979 pixels  (lat 6°N to 38°N)
M_C0, M_C1 = 363, 1254   # cols  → 891 pixels  (lon 66°E to 98°E)

# IMERG India crop pixel indices (0.1° global grid)
I_LAT0, I_LAT1 = 959,  1279   # 320 pixels  (lat 6°N to 38°N)
I_LON0, I_LON1 = 2459, 2779   # 320 pixels  (lon 66°E to 98°E)

# Mercator projection constants (for converting X/Y metres → lat/lon degrees)
_R    = 6378137.0   # WGS84 equatorial radius
_LON0 = 82.0        # central meridian of INSAT-3D ASIA_MER product

# Physical clamp ranges before float16 conversion
_CLAMP = {
    "temp":     (150.0,  340.0),
    "radiance": (0.0,    500.0),
    "albedo":   (0.0,    1.5),
}

# MOSDAC channels: (output_key, image_dataset, lut_dataset, lut_type)
_MOSDAC_CH = [
    ("TIR1",          "IMG_TIR1",  "IMG_TIR1_TEMP",     "temp"),
    ("TIR2",          "IMG_TIR2",  "IMG_TIR2_TEMP",     "temp"),
    ("WV",            "IMG_WV",    "IMG_WV_TEMP",        "temp"),
    ("MIR",           "IMG_MIR",   "IMG_MIR_TEMP",       "temp"),
    ("VIS",           "IMG_VIS",   "IMG_VIS_ALBEDO",     "albedo"),
    ("SWIR",          "IMG_SWIR",  "IMG_SWIR_RADIANCE",  "radiance"),
    ("TIR1_RADIANCE", "IMG_TIR1",  "IMG_TIR1_RADIANCE",  "radiance"),
    ("TIR2_RADIANCE", "IMG_TIR2",  "IMG_TIR2_RADIANCE",  "radiance"),
    ("WV_RADIANCE",   "IMG_WV",    "IMG_WV_RADIANCE",    "radiance"),
    ("MIR_RADIANCE",  "IMG_MIR",   "IMG_MIR_RADIANCE",   "radiance"),
    ("VIS_RADIANCE",  "IMG_VIS",   "IMG_VIS_RADIANCE",   "radiance"),
]


# ── Coordinate conversion ─────────────────────────────────────────────────────

def _merc_y_to_lat(y):
    return math.degrees(2 * math.atan(math.exp(y / _R)) - math.pi / 2)

def _merc_x_to_lon(x):
    return math.degrees(x / _R) + _LON0

def _make_latlon(x_full, y_full):
    """Convert Mercator metre arrays → cropped degree lat/lon arrays."""
    lat = np.array([_merc_y_to_lat(y) for y in y_full[M_R0:M_R1]], dtype=np.float32)
    lon = np.array([_merc_x_to_lon(x) for x in x_full[M_C0:M_C1]], dtype=np.float32)
    return lat, lon


# ── MOSDAC crop ───────────────────────────────────────────────────────────────

def crop_mosdac(h5_path, out_path):
    save = {}

    with h5py.File(h5_path, "r") as f:

        # Coordinate arrays
        lat, lon = _make_latlon(f["X"][:], f["Y"][:])
        save["lat"] = lat
        save["lon"] = lon

        # Timestamp
        try:
            save["time"] = np.array([float(f["time"][0])], dtype=np.float64)
        except Exception:
            save["time"] = np.array([0.0], dtype=np.float64)

        # Image channels via LUT
        for key, img_ds, lut_ds, lut_type in _MOSDAC_CH:
            if img_ds not in f or lut_ds not in f:
                continue

            # Read raw counts for the India crop region only
            raw = f[img_ds][0, M_R0:M_R1, M_C0:M_C1].astype(np.int32)   # (979, 891)
            lut = f[lut_ds][:]                                             # (1024,) physical values

            # Map counts → physical values
            counts_safe = np.clip(raw, 0, 1023)
            arr = lut[counts_safe].astype(np.float32)

            # Replace fill (count=0) with minimum valid physical value
            lo, hi = _CLAMP[lut_type]
            arr[raw == 0] = lo
            arr = np.clip(arr, lo, hi)

            save[key] = arr.astype(np.float16)

        # Geometry channels (centidegrees → degrees)
        for key, ds in [("Sat_Azimuth",   "Sat_Azimuth"),
                        ("Sat_Elevation", "Sat_Elevation"),
                        ("Sun_Azimuth",   "Sun_Azimuth"),
                        ("Sun_Elevation", "Sun_Elevation")]:
            if ds not in f:
                continue
            raw = f[ds][0, M_R0:M_R1, M_C0:M_C1].astype(np.float32)
            arr = raw / 100.0                        # centidegrees → degrees
            arr = np.clip(arr, -180.0, 360.0)
            save[key] = arr.astype(np.float16)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **save)
    return True


# ── IMERG crop ────────────────────────────────────────────────────────────────

def crop_imerg(hdf5_path, out_path):
    with h5py.File(hdf5_path, "r") as f:
        g = f["Grid"]

        # Field name changed between IMERG versions
        if "precipitation" in g:
            field = "precipitation"
        elif "precipitationCal" in g:
            field = "precipitationCal"
        else:
            print(f"  [WARN] No precip field in {os.path.basename(hdf5_path)}")
            return False

        # Shape (1, 3600, 1800) = (time, lon, lat) — lon is FIRST
        # Crop lon[LON0:LON1], lat[LAT0:LAT1]  → transpose to (lat, lon)
        raw    = g[field][0, I_LON0:I_LON1, I_LAT0:I_LAT1].astype(np.float32)
        precip = raw.T                        # (320_lat, 320_lon)

        # Clean fill values
        precip[precip < -9000] = 0.0
        precip[precip < 0]     = 0.0
        precip = np.clip(precip, 0.0, 200.0)

        lat = g["lat"][I_LAT0:I_LAT1].astype(np.float32)   # (320,)
        lon = g["lon"][I_LON0:I_LON1].astype(np.float32)   # (320,)

        try:
            time_val = float(g["time"][0])
        except Exception:
            time_val = 0.0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        precipitation = precip.astype(np.float16),
        lat           = lat,
        lon           = lon,
        time          = np.array([time_val], dtype=np.float64),
    )
    return True


# ── Helpers ───────────────────────────────────────────────────────────────────

def event_dir(root, n):
    s = {1:"st", 2:"nd", 3:"rd"}.get(
        n % 10 if n % 100 not in (11, 12, 13) else 0, "th")
    for name in [f"{n}{s}_event", f"event_{n:02d}"]:
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    return None

def list_files(folder, exts):
    if not os.path.isdir(folder):
        return []
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if any(f.upper().endswith(e.upper()) for e in exts)
    ])

def folder_size_mb(folder):
    if not os.path.isdir(folder):
        return 0.0
    return sum(os.path.getsize(os.path.join(folder, f))
               for f in os.listdir(folder)
               if os.path.isfile(os.path.join(folder, f))) / 1e6


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ok_m = ok_i = fail_m = fail_i = skip = 0

    for n in range(1, N_EVENTS + 1):
        edir = event_dir(DATA_ROOT, n)
        if edir is None:
            print(f"[SKIP] Event {n} not found")
            skip += 1
            continue

        mosdac_files = list_files(os.path.join(edir, "mosdac"), [".h5", ".H5"])
        imerg_files  = list_files(os.path.join(edir, "imerg"),  [".HDF5", ".hdf5"])

        out_m = os.path.join(edir, "cropped", "mosdac")
        out_i = os.path.join(edir, "cropped", "imerg")

        print(f"\n{'─'*55}")
        print(f"Event {n}: {edir}")
        print(f"  MOSDAC: {len(mosdac_files)} files | IMERG: {len(imerg_files)} files")

        # MOSDAC
        for fp in tqdm(mosdac_files, desc="  MOSDAC", ncols=65, leave=False):
            stem = os.path.splitext(os.path.basename(fp))[0]
            out  = os.path.join(out_m, stem + ".npz")
            if os.path.exists(out):
                ok_m += 1; continue
            try:
                crop_mosdac(fp, out)
                ok_m += 1
            except Exception as e:
                print(f"\n  [ERR MOSDAC] {os.path.basename(fp)}: {e}")
                fail_m += 1

        # IMERG
        for fp in tqdm(imerg_files, desc="  IMERG ", ncols=65, leave=False):
            stem = os.path.splitext(os.path.basename(fp))[0]
            out  = os.path.join(out_i, stem + ".npz")
            if os.path.exists(out):
                ok_i += 1; continue
            try:
                crop_imerg(fp, out)
                ok_i += 1
            except Exception as e:
                print(f"\n  [ERR IMERG] {os.path.basename(fp)}: {e}")
                fail_i += 1

        # Size report for this event
        orig_m  = folder_size_mb(os.path.join(edir, "mosdac"))
        orig_i  = folder_size_mb(os.path.join(edir, "imerg"))
        crop_m  = folder_size_mb(out_m)
        crop_i  = folder_size_mb(out_i)
        red_m   = (1 - crop_m / max(orig_m, 1)) * 100
        red_i   = (1 - crop_i / max(orig_i, 1)) * 100
        print(f"  MOSDAC: {orig_m:.0f}MB → {crop_m:.0f}MB  ({red_m:.0f}% smaller)")
        print(f"  IMERG:  {orig_i:.0f}MB → {crop_i:.0f}MB  ({red_i:.0f}% smaller)")

    # Final summary
    print(f"\n{'='*55}")
    print(f"COMPLETE")
    print(f"  MOSDAC saved: {ok_m}  failed: {fail_m}")
    print(f"  IMERG  saved: {ok_i}  failed: {fail_i}")
    print(f"  Events skipped: {skip}")
    print(f"\nOutput format:")
    print(f"  MOSDAC .npz  → lat(979) lon(891) TIR1/TIR2/WV/MIR/VIS/SWIR")
    print(f"                   + radiance channels + geometry  [float16]")
    print(f"  IMERG  .npz  → lat(320) lon(320) precipitation  [float16, mm/hr]")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()