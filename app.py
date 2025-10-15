# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import gpxpy
import gpxpy.gpx
import io
import datetime
import plotly.express as px
import pydeck as pdk

st.set_page_config(page_title="MetaSail Race Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("MetaSail Race Dashboard 🇮🇹")
st.markdown(
    "Carica uno o più file GPX (tracciati GPS). La dashboard mostra tutte le tracce sulla mappa, "
    "statistiche per barca, distanza (km) e velocità in nodi (kts)."
)

# ---------- Helpers ----------
def haversine_distance(lat1, lon1, lat2, lon2):
    # distanza in metri tra due punti (formula dell'haversine)
    R = 6371000.0  # raggio della Terra in metri
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d  # metri

def ms_to_knots(ms):
    # 1 m/s = 1.943844 knots
    return ms * 1.943844

def parse_gpx(file_content):
    gpx = gpxpy.parse(file_content)
    rows = []
    name = None
    # prefer track name, fallback to filename
    if gpx.tracks and gpx.tracks[0].name:
        name = gpx.tracks[0].name
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                rows.append({
                    "time": point.time,
                    "latitude": point.latitude,
                    "longitude": point.longitude,
                    "elevation": point.elevation if point.elevation is not None else np.nan
                })
    # also consider waypoints if no track points found
    if not rows and gpx.waypoints:
        for w in gpx.waypoints:
            rows.append({
                "time": w.time,
                "latitude": w.latitude,
                "longitude": w.longitude,
                "elevation": w.elevation if w.elevation is not None else np.nan
            })
    df = pd.DataFrame(rows)
    return df, name

def compute_track_stats(df):
    # Assumes df sorted by time
    df = df.sort_values("time").reset_index(drop=True)
    n = len(df)
    if n < 2:
        return df.assign(dist_m=0, delta_s=0, speed_ms=0, speed_kts=0, cumdist_km=0), {
            "distance_km": 0.0,
            "duration_h": 0.0,
            "avg_speed_kts": 0.0,
            "max_speed_kts": 0.0,
            "points": n
        }
    dist_list = []
    delta_list = []
    speed_ms_list = []
    cum = 0.0
    cum_list = []
    for i in range(n):
        if i == 0:
            dist_list.append(0.0)
            delta_list.append(0.0)
            speed_ms_list.append(0.0)
            cum_list.append(0.0)
        else:
            p0 = df.loc[i-1]
            p1 = df.loc[i]
            d = haversine_distance(p0.latitude, p0.longitude, p1.latitude, p1.longitude)  # metri
            t0 = p0.time
            t1 = p1.time
            # sicurezza: se timestamp mancante, delta 0
            if pd.isna(t0) or pd.isna(t1):
                dt = 0.0
            else:
                dt = (t1 - t0).total_seconds()
                if dt < 0:
                    dt = 0.0
            # velocità m/s calcolata da distanza / dt
            if dt > 0:
                speed_ms = d / dt
            else:
                speed_ms = 0.0
            cum += d
            dist_list.append(d)
            delta_list.append(dt)
            speed_ms_list.append(speed_ms)
            cum_list.append(cum / 1000.0)  # km
    df = df.assign(dist_m=dist_list, delta_s=delta_list, speed_ms=speed_ms_list, speed_kts=[ms_to_knots(x) for x in speed_ms_list], cumdist_km=cum_list)
    total_distance_km = cum / 1000.0
    # duration: last time - first time
    t_first = df.loc[0, "time"]
    t_last = df.loc[n-1, "time"]
    if pd.isna(t_first) or pd.isna(t_last):
        duration_s = 0.0
    else:
        duration_s = max(0.0, (t_last - t_first).total_seconds())
    duration_h = duration_s / 3600.0 if duration_s > 0 else 0.0
    # average speed in kts: use total distance / duration
    if duration_s > 0:
        avg_speed_ms = (cum / duration_s)  # m/s
        avg_speed_kts = ms_to_knots(avg_speed_ms)
    else:
        avg_speed_kts = 0.0
    max_speed_kts = float(df["speed_kts"].max()) if len(df)>0 else 0.0
    stats = {
        "distance_km": total_distance_km,
        "duration_h": duration_h,
        "avg_speed_kts": avg_speed_kts,
        "max_speed_kts": max_speed_kts,
        "points": n
    }
    return df, stats

# ---------- Upload ----------
uploaded = st.file_uploader("Carica uno o più file GPX", type=["gpx"], accept_multiple_files=True)
if not uploaded:
    st.info("Carica i file .gpx per iniziare (es. Bonomo Laura.gpx, Santomarco.gpx).")
    st.stop()

# ---------- Process all files ----------
all_tracks = {}
for f in uploaded:
    try:
        content = f.read().decode("utf-8")
    except Exception:
        # binary read fallback
        f.seek(0)
        content = f.read()
    df, name = parse_gpx(io.StringIO(content) if isinstance(content, str) else io.BytesIO(content))
    if name is None:
        name = f.name
    if df.empty:
        st.warning(f"Il file {f.name} non contiene punti di traccia leggibili.")
        continue
    df_parsed, stats = compute_track_stats(df)
    all_tracks[name] = {"df": df_parsed, "stats": stats, "filename": f.name}

if not all_tracks:
    st.error("Nessuna traccia valida trovata nei file.")
    st.stop()

# ---------- Sidebar: selezioni ----------
st.sidebar.header("Impostazioni")
show_all = st.sidebar.checkbox("Mostra tutte le barche sulla mappa (se deselezionato, seleziona una sola)", value=True)
boat_names = list(all_tracks.keys())
selected = st.sidebar.multiselect("Seleziona barche da visualizzare", options=boat_names, default=boat_names if show_all else [boat_names[0]])
if not selected:
    st.sidebar.error("Seleziona almeno una barca.")
    st.stop()

# ---------- Mappa con pydeck ----------
st.subheader("Mappa: tracciati (tutte le barche)")
# prepare lines for pydeck
layers = []
colors = px.colors.qualitative.Plotly
color_map = {}
i = 0
for name in selected:
    tr = all_tracks[name]["df"]
    coords = [[float(lon), float(lat)] for lat, lon in zip(tr.latitude, tr.longitude)]
    color = [int(x*255) for x in px.colors.hex_to_rgb(colors[i % len(colors)])] if hasattr(px.colors, "hex_to_rgb") else None
    # fallback color generation
    hexcolor = colors[i % len(colors)]
    # pydeck expects [r,g,b] 0-255
    try:
        rgb = [int(255*val) for val in pd.Series(px.colors.convert_colors_to_same_type(hexcolor)).tolist()]  # best-effort; not always required
        color = rgb
    except Exception:
        # simple hashing
        hashv = abs(hash(name)) % 255
        color = [hashv, (hashv*2) % 255, (hashv*3) % 255]
    color_map[name] = color
    if len(coords) >= 2:
        layers.append(pdk.Layer(
            "PathLayer",
            data=[{"path": coords, "name": name}],
            get_path="path",
            get_color=color,
            width_scale=20,
            width_min_pixels=2,
            pickable=True,
            auto_highlight=True
        ))
    i += 1

# auto view state
# choose center as mean of first selected track points
all_lats = []
all_lons = []
for name in selected:
    tr = all_tracks[name]["df"]
    all_lats.extend(tr.latitude.tolist())
    all_lons.extend(tr.longitude.tolist())

if all_lats and all_lons:
    mid_lat = float(np.mean(all_lats))
    mid_lon = float(np.mean(all_lons))
    initial_view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=12, pitch=0)
else:
    initial_view = pdk.ViewState(latitude=0, longitude=0, zoom=1, pitch=0)

r = pdk.Deck(layers=layers, initial_view_state=initial_view, map_style='mapbox://styles/mapbox/light-v10' if False else "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json")
st.pydeck_chart(r)

# ---------- Stats table ----------
st.subheader("Statistiche riassuntive per barca")
stats_rows = []
for name in selected:
    s = all_tracks[name]["stats"]
    stats_rows.append({
        "Barca": name,
        "Distanza (km)": round(s["distance_km"], 3),
        "Durata (h)": round(s["duration_h"], 3),
        "Velocità media (kts)": round(s["avg_speed_kts"], 2),
        "Velocità massima (kts)": round(s["max_speed_kts"], 2),
        "Punti": s["points"]
    })
stats_df = pd.DataFrame(stats_rows).sort_values(by="Distanza (km)", ascending=False).reset_index(drop=True)
st.dataframe(stats_df, use_container_width=True)

# CSV download
csv = stats_df.to_csv(index=False).encode("utf-8")
st.download_button("Scarica statistiche (CSV)", data=csv, file_name="metasail_stats.csv", mime="text/csv")

# ---------- Dettaglio singolo / grafici ----------
st.subheader("Grafici dettagliati")
for name in selected:
    tr = all_tracks[name]["df"].copy()
    if tr.empty:
        continue
    st.markdown(f"### {name}")
    # velocità nel tempo (nodi)
    if tr["time"].isnull().all():
        st.warning("Nessun timestamp disponibile per questa traccia; alcuni grafici temporali non saranno mostrati.")
    else:
        fig_v = px.line(tr, x="time", y="speed_kts", title=f"Velocità nel tempo — {name}", labels={"speed_kts":"Velocità (kts)", "time":"Tempo"})
        st.plotly_chart(fig_v, use_container_width=True)
        # distanza cumulativa nel tempo
        fig_d = px.line(tr, x="time", y="cumdist_km", title=f"Distanza cumulativa (km) — {name}", labels={"cumdist_km":"Distanza (km)", "time":"Tempo"})
        st.plotly_chart(fig_d, use_container_width=True)

st.markdown("---")
st.caption("Nota: le velocità sono calcolate da distanza/tempo tra punti consecutivi se non presenti direttamente nel GPX. Le distanze sono calcolate con la formula dell'haversine e convertite in km; le velocità sono espresse in nodi (1 m/s = 1.943844 kts).")
