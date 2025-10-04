# -*- coding: utf-8 -*-
"""
Lakhatósági (habitable) ellenőrző függvények Kepler/KOI datasethez.

Modell: egyszerű egyensúlyi-hőmérséklet (állandó albedó) alapú HZ,
a Valle et al. (2014) cikkben ismertetett képletek szerint:

- d = sqrt( (1 - A) * L / (16 * pi * sigma * T_p^4) )
  ahol T_p a határhőmérséklet (belső: 269 K, külső: 203 K vagy 169 K).
- L = L_sun * (R_*/R_sun)^2 * (T_eff/T_sun)^4

A kód:
- kiszámolja a csillag luminositását (L/L_sun),
- a fél nagytengelyt AU-ban (koi_sma vagy Kepler 3. törvény a periódusból),
- HZ belső/külső határt (AU),
- besorolást ad: in_hz (jelenleg HZ-ben van-e),
- "likely_habitable" (kőzetes méret + HZ + értelmes besorolás).

Megjegyzés: Ez egy gyors, klimamodell-mentes becslés. A pontosabb
(spektrális függésű) HZ-hez alkalmazható a Kopparapu et al. (2013) polinom,
de itt nem használunk külső táblázatot/koefficienseket.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd

# Fizikai állandók
SIGMA = 5.670374419e-8        # Stefan–Boltzmann konstans [W m^-2 K^-4]
L_SUN = 3.828e26              # Nap luminositása [W]
R_SUN = 6.957e8               # Nap sugara [m]
AU = 1.495978707e11           # Csillagászati egység [m]
T_SUN = 5772.0                # Nap effektív hőmérséklete [K]
DAY = 86400.0                 # másodperc / nap
YEAR = 365.25                 # nap / év
G = 6.67430e-11               # gravitációs állandó
M_SUN = 1.98847e30            # Nap tömege

def _stellar_luminosity_watt(srad_rsun: float, steff_k: float) -> float:
    """
    Csillag luminositás W-ban, a (R/Rsun)^2 (T/Tsun)^4 képletből.
    """
    if pd.isna(srad_rsun) or pd.isna(steff_k):
        return np.nan
    L = L_SUN * (srad_rsun**2) * (steff_k / T_SUN) ** 4
    return L

def _semi_major_axis_au(row: pd.Series) -> float:
    """
    Fél nagytengely AU-ban. 
    Elsőbbség: koi_sma (csillagsugarakban) -> átváltás AU-ra.
    Ha hiányzik: Kepler 3. törvénye a periódusból és csillagtömegből (M/Msun).
        a^3 = G * (Mstar + Mplanet) * P^2 / (4 pi^2)  ~ G*Mstar*P^2/(4*pi^2)
    Visszatér: np.nan, ha nem számítható.
    """
    srad_rsun = row.get("koi_srad", np.nan)     # [R_sun]
    sma_stellar_radii = row.get("koi_sma", np.nan)  # [R_star]
    period_days = row.get("koi_period", np.nan)     # [days]
    mstar_msun = row.get("koi_smass", np.nan)       # [M_sun]

    # 1) koi_sma -> AU
    if pd.notna(sma_stellar_radii) and pd.notna(srad_rsun):
        a_m = sma_stellar_radii * srad_rsun * R_SUN
        return a_m / AU

    # 2) Kepler 3. törvénye: periódus + tömeg
    if pd.notna(period_days) and pd.notna(mstar_msun):
        P = period_days * DAY
        a_cubed = G * (mstar_msun * M_SUN) * P**2 / (4 * np.pi**2)
        a_m = a_cubed ** (1/3)
        return a_m / AU

    return np.nan

def _hz_boundary_au(L_watt: float, A: float, T_boundary: float) -> float:
    """
    Habitable Zone határ távolsága (AU) az egyensúlyi hőmérséklet modellből.
    d = sqrt( (1-A) * L / (16 pi sigma T^4) )
    """
    if pd.isna(L_watt):
        return np.nan
    numerator = (1.0 - A) * L_watt
    denom = 16.0 * np.pi * SIGMA * (T_boundary**4)
    d_m = np.sqrt(numerator / denom)
    return d_m / AU

def classify_size_rocky(prad_rearth: float, rocky_cut: float = 1.8) -> Optional[bool]:
    """
    Egyszerű kőzetes/köztes/gáz jelölés: True ha kőzetesnek tekintjük.
    Alapértelmezett vágás: 1.8 R_earth (irodalomban 1.5–2.0 közötti küszöb gyakori).
    """
    if pd.isna(prad_rearth):
        return None
    return bool(prad_rearth <= rocky_cut)

def compute_habitability(
    df: pd.DataFrame,
    albedo: float = 0.3,
    T_inner: float = 269.0,
    T_outer: float = 203.0,
    rocky_cut_rearth: float = 1.8,
    require_confirmed: bool = False,
) -> pd.DataFrame:
    """
    Fő belépési pont: visszaad egy új DataFrame-et lakhatósági oszlopokkal.

    Paraméterek:
    - albedo: Bond-albedó (0.3 ~ Föld-szerű feltételezés)
    - T_inner: belső HZ egyensúlyi hőmérsékleti küszöb (K); tipikusan 269 K
    - T_outer: külső HZ egyensúlyi hőmérsékleti küszöb (K); tipikusan 203 K (vagy 169 K)
    - rocky_cut_rearth: bolygó kőzetesnek tekintett maximum rádiusza [R_⊕]
    - require_confirmed: ha True, csak CONFIRMED objektumokra ad True "likely_habitable"-t
    """
    out = df.copy()

    # Csillag luminositás (W) és L/Lsun
    out["star_lum_watt"] = df.apply(
        lambda r: _stellar_luminosity_watt(r.get("koi_srad", np.nan), r.get("koi_steff", np.nan)),
        axis=1
    )
    out["star_lum_solar"] = out["star_lum_watt"] / L_SUN

    # Fél nagytengely (AU)
    out["a_au"] = df.apply(_semi_major_axis_au, axis=1)

    # HZ határok (AU)
    out["hz_inner_au"] = out["star_lum_watt"].apply(lambda L: _hz_boundary_au(L, albedo, T_inner))
    out["hz_outer_au"] = out["star_lum_watt"].apply(lambda L: _hz_boundary_au(L, albedo, T_outer))

    # HZ tagság (jelenleg)
    out["in_hz_now"] = (out["a_au"] >= out["hz_inner_au"]) & (out["a_au"] <= out["hz_outer_au"])

    # Kőzetes? (rádiusz alapján)
    out["rocky_radius"] = df["koi_prad"].apply(lambda r: classify_size_rocky(r, rocky_cut=rocky_cut_rearth))

    # Minősítés szűrő (opcionális)
    if require_confirmed:
        out["status_ok"] = df["koi_disposition"].astype(str).str.upper().eq("CONFIRMED")
    else:
        # Candidate vagy Confirmed elfogadott
        disp = df["koi_disposition"].astype(str).str.upper()
        out["status_ok"] = disp.isin(["CONFIRMED", "CANDIDATE"])

    # "likely_habitable" heurisztika: HZ + kőzetes + státusz OK
    out["likely_habitable"] = out["in_hz_now"] & out["status_ok"] & (out["rocky_radius"] == True)

    # Hasznos kimeneti oszlopok rendezése
    cols_front = [
        "rowid", "kepid", "kepoi_name", "kepler_name",
        "koi_disposition", "koi_prad", "koi_period",
        "koi_smass", "koi_srad", "koi_steff"
    ]
    cols_front = [c for c in cols_front if c in out.columns]

    result_cols = cols_front + [
        "a_au", "star_lum_solar", "hz_inner_au", "hz_outer_au",
        "in_hz_now", "rocky_radius", "status_ok", "likely_habitable"
    ]
    result_cols = [c for c in result_cols if c in out.columns]
    return out[result_cols]


# ---- Példa használat (mű-adatokkal) ----
example = pd.DataFrame([
    {
        "rowid": 1,
        "kepid": 1234567,
        "kepoi_name": "K01234.01",
        "kepler_name": "",
        "koi_disposition": "CANDIDATE",
        "koi_prad": 1.4,       # R_earth
        "koi_period": 365.0,   # nap
        "koi_smass": 1.0,      # M_sun
        "koi_srad": 1.0,       # R_sun
        "koi_steff": 5772.0,   # K
        "koi_sma": np.nan,     # hiányzik -> Kepler 3. törvénye
    },
    {
        "rowid": 2,
        "kepid": 2345678,
        "kepoi_name": "K07654.01",
        "kepler_name": "Kepler-XYZ b",
        "koi_disposition": "CONFIRMED",
        "koi_prad": 2.6,       # nem kőzetes
        "koi_period": 30.0,    # nap
        "koi_smass": 0.9,      # M_sun
        "koi_srad": 0.9,       # R_sun
        "koi_steff": 5400.0,   # K
        "koi_sma": np.nan,
    },
    {
        "rowid": 3,
        "kepid": 3456789,
        "kepoi_name": "K05555.01",
        "kepler_name": "",
        "koi_disposition": "CANDIDATE",
        "koi_prad": 1.2,        # R_earth
        "koi_period": np.nan,   # nincs periódus
        "koi_smass": 0.8,       # M_sun
        "koi_srad": 0.8,        # R_sun
        "koi_steff": 5200.0,    # K
        "koi_sma": 215.0,       # csillagsugarakban
    },
])

result = compute_habitability(example, albedo=0.3, T_inner=269.0, T_outer=203.0, rocky_cut_rearth=1.8, require_confirmed=False)
import caas_jupyter_tools as cj
cj.display_dataframe_to_user("Lakhatósági becslés — példa kimenet", result)

print("Kész. A fenti táblázatban látható a példakimenet. A saját adatkeretedre hívd meg a compute_habitability(df, ...) függvényt.")

