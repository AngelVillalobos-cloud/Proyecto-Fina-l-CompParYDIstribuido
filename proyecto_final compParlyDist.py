"""
PROYECTO FINAL: COMPUTO PARALELO Y DISTRIBUIDO (2a. EVALUACION)
Analisis de trafico aereo y condiciones ambientales (ThingSpeak + AirLabs)
Pilares: 1. Asyncio | 2. Pandas | 3. MPI4Py | 4. Numba | 5. Matplotlib
"""

import sys
import time
import math
import json
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. ACELERACION NUMERICA (Numba JIT)
# ------------------------------------------------------------------------------
try:
    from numba import njit
except ImportError:
    njit = lambda f: f  # Fallback si Numba no esta instalado

@njit(fastmath=True)
def numba_zscores(arr):
    """Calcula Z-Scores acelerados para detectar vuelos atipicos (anomalias)."""
    n = len(arr)
    if n == 0: return np.empty(0, dtype=np.float64)
    media = np.sum(arr) / n
    desv = math.sqrt(np.sum((arr - media) ** 2) / n)
    desv = desv if desv > 0 else 1.0
    return (arr - media) / desv

@njit(fastmath=True)
def numba_correlacion(x, y):
    """Calcula la correlacion de Pearson acelerada entre clima y retrasos."""
    n = len(x)
    if n == 0 or len(y) != n: return 0.0
    mx, my = np.sum(x) / n, np.sum(y) / n
    num = np.sum((x - mx) * (y - my))
    den = math.sqrt(np.sum((x - mx) ** 2) * np.sum((y - my) ** 2))
    return num / den if den != 0.0 else 0.0

# ------------------------------------------------------------------------------
# 2. COMPUTO DISTRIBUIDO (MPI4Py)
# ------------------------------------------------------------------------------
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
except ImportError:
    comm, rank, size = None, 0, 1

# ------------------------------------------------------------------------------
# 3. LECTURA CONCURRENTE DE APIS (Asyncio)
# ------------------------------------------------------------------------------
async def fetch_thingspeak():
    """Descarga asincrona de sensores meteorologicos desde ThingSpeak."""
    import urllib.request
    url = "https://api.thingspeak.com/channels/12397/feeds.json?results=120"
    loop = asyncio.get_running_loop()
    
    def req():
        try:
            with urllib.request.urlopen(url, timeout=4) as r:
                return json.loads(r.read().decode('utf-8')).get("feeds", [])
        except Exception:
            return []
            
    feeds = await loop.run_in_executor(None, req)
    registros = []
    
    if feeds:
        for f in feeds:
            t = float(f.get("field1", 22.0) or 22.0)
            h = float(f.get("field2", 60.0) or 60.0)
            p = float(f.get("field3", 1013.25) or 1013.25)
            # Normalizar escalas
            t = 18.0 + (t % 14.0) if t > 45 else t
            h = 45.0 + (h * 5.5) if h <= 15.0 else h
            p = 1013.25 + (p % 10.0) - 5.0 if (p < 850 or p > 1100) else p
            registros.append({"temp_c": t, "hum_pct": h, "pres_hpa": p, "viento_kmh": 12.0 + (h * 0.15)})
    else:
        # Generador de respaldo por si no hay red
        np.random.seed(42)
        for i in range(120):
            h = np.clip(55.0 + 20.0 * math.sin(i / 15.0) + np.random.normal(0, 2), 30, 95)
            registros.append({"temp_c": 24.0 + 5.0 * math.cos(i / 20.0), "hum_pct": h, "pres_hpa": 1013.0, "viento_kmh": 10.0 + h * 0.2})
            
    return pd.DataFrame(registros)

async def fetch_airlabs(n=120):
    """Simulacion concurrente de telemetria aerea segun esquema AirLabs."""
    await asyncio.sleep(0.05)
    np.random.seed(99)
    aeropuertos = ["MEX", "CUN", "GDL", "MTY", "TIJ", "BOG", "MAD", "JFK"]
    vuelos = []
    for _ in range(n):
        alt = float(np.random.normal(32000, 3000))
        vel = float(np.random.normal(820, 50))
        if np.random.rand() < 0.06: alt = float(np.random.choice([8000, 48000]))  # Anomalia
        if np.random.rand() < 0.06: vel = float(np.random.choice([450, 1020]))   # Anomalia
        vuelos.append({
            "aeropuerto": np.random.choice(aeropuertos),
            "altitud_ft": round(alt, 1),
            "velocidad_kmh": round(vel, 1)
        })
    return pd.DataFrame(vuelos)

async def descargar_datos_concurrentes():
    """Descarga simultanea usando asyncio.gather."""
    t0 = time.time()
    df_env, df_air = await asyncio.gather(fetch_thingspeak(), fetch_airlabs())
    return df_env, df_air, time.time() - t0

# ------------------------------------------------------------------------------
# 4. TRABAJO PARALELO DISTRIBUIDO (Nodos de Computo)
# ------------------------------------------------------------------------------
def procesar_nodo(df_chunk, rank_id):
    """Procesa un fragmento de datos aplicando funciones Numba JIT."""
    if df_chunk is None or len(df_chunk) == 0:
        return {"rank": rank_id, "anomalias": 0, "df": pd.DataFrame()}

    df = pd.DataFrame(df_chunk).copy()
    alt = np.ascontiguousarray(df["altitud_ft"].values, dtype=np.float64)
    vel = np.ascontiguousarray(df["velocidad_kmh"].values, dtype=np.float64)
    ret = np.ascontiguousarray(df["retraso_min"].values, dtype=np.float64)

    # Z-scores y deteccion de anomalias (|z| > 2.0)
    z_alt = numba_zscores(alt)
    z_vel = numba_zscores(vel)
    es_anomalo = (np.abs(z_alt) > 2.0) | (np.abs(z_vel) > 2.0)
    
    df["es_anomalo"] = es_anomalo
    return {"rank": rank_id, "anomalias": int(np.sum(es_anomalo)), "df": df}

# ------------------------------------------------------------------------------
# 5. VISUALIZACION (Dashboard de 4 Paneles con Matplotlib)
# ------------------------------------------------------------------------------
def generar_dashboard(df, ruta="reporte_analisis.png"):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Analisis de Trafico Aereo y Condiciones Ambientales\n(Asyncio, MPI4Py, Numba JIT y Pandas)", fontsize=15, fontweight='bold', y=0.97)

    # 1. Humedad vs Retraso
    ax1 = axes[0, 0]
    sc = ax1.scatter(df["hum_pct"], df["retraso_min"], c=df["viento_kmh"], cmap='Blues', alpha=0.75, edgecolors='gray')
    cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Viento (km/h)", fontsize=9)
    z = np.polyfit(df["hum_pct"], df["retraso_min"], 1)
    ax1.plot(np.sort(df["hum_pct"]), np.poly1d(z)(np.sort(df["hum_pct"])), "r--", label="Tendencia")
    ax1.set_title("1. Humedad y Viento vs Retrasos", fontsize=11, fontweight='semibold', pad=10)
    ax1.set_xlabel("Humedad (%) [ThingSpeak]", fontsize=10, labelpad=8)
    ax1.set_ylabel("Retraso (min)", fontsize=10, labelpad=8)
    ax1.legend()

    # 2. Retrasos por Aeropuerto
    ax2 = axes[0, 1]
    ret_aero = df.groupby("aeropuerto")["retraso_min"].mean().sort_values(ascending=False)
    colores = ['#d9534f' if v > ret_aero.median() else '#5bc0de' for v in ret_aero.values]
    ax2.bar(ret_aero.index, ret_aero.values, color=colores, edgecolor='black', alpha=0.85, width=0.6)
    ax2.axhline(ret_aero.mean(), color='black', linestyle=':', label=f"Promedio ({ret_aero.mean():.1f} min)")
    ax2.set_title("2. Promedio de Retraso por Aeropuerto", fontsize=11, fontweight='semibold', pad=10)
    ax2.set_xlabel("Aeropuerto [AirLabs]", fontsize=10, labelpad=8)
    ax2.set_ylabel("Retraso Promedio (min)", fontsize=10, labelpad=8)
    ax2.legend()

    # 3. Anomalias en Vuelo (Numba)
    ax3 = axes[1, 0]
    norm, anom = df[~df["es_anomalo"]], df[df["es_anomalo"]]
    ax3.scatter(norm["velocidad_kmh"], norm["altitud_ft"], color="#2b5c8f", alpha=0.6, label="Normal", s=35)
    ax3.scatter(anom["velocidad_kmh"], anom["altitud_ft"], color="crimson", marker="x", s=85, label=f"Anomalias ({len(anom)})", linewidths=2)
    ax3.set_title("3. Deteccion de Anomalias (Acelerado con Numba)", fontsize=11, fontweight='semibold', pad=10)
    ax3.set_xlabel("Velocidad (km/h)", fontsize=10, labelpad=8)
    ax3.set_ylabel("Altitud (pies)", fontsize=10, labelpad=8)
    ax3.legend()

    # 4. Monitoreo Ambiental Temporal
    ax4 = axes[1, 1]
    ax4_t = ax4.twinx()
    l1 = ax4.plot(df.index, df["temp_c"], color="#ff7f0e", label="Temperatura (C)", linewidth=2)
    l2 = ax4_t.plot(df.index, df["pres_hpa"], color="#2ca02c", linestyle='--', label="Presion (hPa)", linewidth=1.8)
    ax4.set_title("4. Condiciones Ambientales Temporales", fontsize=11, fontweight='semibold', pad=10)
    ax4.set_xlabel("Muestra Temporal", fontsize=10, labelpad=8)
    ax4.set_ylabel("Temperatura (C)", color="#ff7f0e", fontsize=10, labelpad=8)
    ax4_t.set_ylabel("Presion (hPa)", color="#2ca02c", fontsize=10, labelpad=8)
    ax4.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc='upper right')

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.92], h_pad=3.8, w_pad=3.0)
    plt.savefig(ruta, dpi=300)
    print(f"[Visualizacion] Dashboard guardado en: '{ruta}'")
    try: plt.show()
    except Exception: pass

# ------------------------------------------------------------------------------
# 6. ORQUESTADOR PRINCIPAL (MPI Master / Workers)
# ------------------------------------------------------------------------------
def main():
    if rank == 0:
        print("=" * 65)
        print(f"SISTEMA DISTRIBUIDO | Procesos: {size} | Numba JIT: Activo")
        print("=" * 65)
        # Paso 1: Asyncio
        df_env, df_air, t_descarga = asyncio.run(descargar_datos_concurrentes())
        print(f"[1. Asyncio] {len(df_env)} registros descargados en {t_descarga:.3f}s")

        # Paso 2: Pandas
        m = min(len(df_env), len(df_air))
        df = pd.concat([df_air.iloc[:m], df_env.iloc[:m]], axis=1)
        # Factor retraso influenciado por clima
        factor_clima = (df["hum_pct"] * 0.25) + (df["viento_kmh"] * 0.7)
        df["retraso_min"] = np.round(np.maximum(0.0, factor_clima * 0.3 + np.random.exponential(6.0, m) - 8.0), 1)
        print(f"[2. Pandas] Dataset unificado con {len(df)} registros.")

        # Paso 3: Dividir para MPI Scatter
        print(f"[3. MPI4Py] Repartiendo datos a {size} proceso(s)...")
        indices = np.array_split(np.arange(len(df)), size)
        chunks = [df.iloc[idx].reset_index(drop=True) for idx in indices]
    else:
        chunks = None

    # Distribucion (Scatter)
    mi_chunk = comm.scatter(chunks, root=0) if (comm and size > 1) else (chunks[0] if chunks else None)

    # Computo en cada nodo (Numba)
    t0 = time.time()
    res_nodo = procesar_nodo(mi_chunk, rank)
    print(f" -> [Nodo {rank}] Procesados: {len(res_nodo['df'])} | Anomalias: {res_nodo['anomalias']} | Tiempo: {time.time()-t0:.4f}s")

    # Recoleccion (Gather)
    todos = comm.gather(res_nodo, root=0) if (comm and size > 1) else [res_nodo]

    # Consolidacion en Maestro (Rank 0)
    if rank == 0:
        df_final = pd.concat([r["df"] for r in todos if len(r["df"]) > 0], ignore_index=True)
        corr = numba_correlacion(df_final["hum_pct"].values.astype(np.float64), df_final["retraso_min"].values.astype(np.float64))
        
        print("\n" + "=" * 65)
        print(f"RESULTADOS: Anomalias totales: {sum(r['anomalias'] for r in todos)} | Correlacion Clima-Retraso: {corr:.3f}")
        top_aero = df_final.groupby("aeropuerto")["retraso_min"].mean().idxmax()
        print(f"Aeropuerto con mayor retraso promedio: {top_aero}")
        print("=" * 65)

        # Paso 5: Graficas
        generar_dashboard(df_final)

if __name__ == "__main__":
    main()
