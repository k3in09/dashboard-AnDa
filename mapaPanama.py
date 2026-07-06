import geopandas as gpd
import glob
import json
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
#mapa de distritos 
gdf = gpd.read_file('geoBoundaries-PAN-ADM2_simplified.geojson')

archivo_inec = glob.glob("datos_sociodemográfica_*.csv")[0]
df_inec = pd.read_csv(archivo_inec)
print(type(gdf)) 
print(gdf.shape) 
print(gdf.columns.tolist()) 
print(gdf.head())
print(gdf.crs)
print(gdf['shapeName'].sort_values().tolist())

# Limpieza para el GeoDataFrame (geoBoundaries)
gdf["distrito_clean"] = (
    gdf["shapeName"]
    .str.strip()
    .str.upper()
    .str.normalize("NFKD")
    .str.encode("ascii", errors="ignore")
    .str.decode("utf-8")
)

# Limpieza para el DataFrame del INEC
df_inec["distrito_clean"] = (
    df_inec["Nombre Distrito"]
    .str.strip()
    .str.upper()
    .str.normalize("NFKD")
    .str.encode("ascii", errors="ignore")
    .str.decode("utf-8")
)

gdf_merged = gdf.merge(
    df_inec,
    on="distrito_clean",
    how="left"
)

# Verificar que no haya NaN en la columna numérica del INEC (que se llama 'Valor')
print("\nCantidad de distritos sin datos (NaN) después del merge:")
print(gdf_merged["Valor"].isna().sum())
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Mapa coroplético, el color representa el acceso a celular (columna 'Valor')
gdf_merged.plot(
    column="Valor",
    ax=ax,
    legend=True,
    cmap="YlOrRd", 
    edgecolor="white",
    linewidth=0.8,
    legend_kwds={
        "label": "Porcentaje de la población con acceso a celular",
        "orientation": "horizontal",
        "fraction": 0.046,
        "pad": 0.04,
    },
    missing_kwds={"color": "lightgrey", "label": "Sin datos"},
)

# Configurar título del mapa al estilo del profesor
ax.set_title(
    "Población con Acceso a Celular por Distrito\nPanamá, Censo 2023",
    fontsize=16,
    fontweight="bold",
    pad=20,
)
ax.set_axis_off()
plt.tight_layout()

# Guardar el mapa como imagen fija tal como exige la guía
plt.savefig("mapa_panama_distritos.png", dpi=150, bbox_inches="tight")
plt.show()
df_top = df_inec.sort_values(by="Valor", ascending=False).head(15)

plt.figure(figsize=(12, 6))
bars = plt.bar(df_top["Nombre Distrito"], df_top["Valor"], color="#EC3317", edgecolor="black")

# Colocar etiquetas del porcentaje exacto encima de cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 1,
        f"{yval:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold"
    )

plt.title("Top 15 Distritos con Mayor Tasa de Acceso a Celular\nPanamá, Censo 2023", fontsize=14, fontweight="bold", pad=15)
plt.ylabel("Porcentaje (%)", fontweight="bold")
plt.xlabel("Distritos", fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 110) # Espacio arriba para las etiquetas
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

# Guardar la gráfica complementaria
plt.savefig("grafica_resumen_distritos.png", dpi=150, bbox_inches="tight")
plt.show()