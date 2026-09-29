import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})

COR_REAL = "#1f77b4"
COR_DESTAQUE = "#d62728"
COR_NEUTRA = "#4c72b0"


def formatar_eixo_tempo(eixos):
    localizador_datas = mdates.AutoDateLocator(minticks=4, maxticks=7)
    formato_datas = mdates.ConciseDateFormatter(localizador_datas)

    eixos.xaxis.set_major_locator(localizador_datas)
    eixos.xaxis.set_major_formatter(formato_datas)
    eixos.grid(axis="y", linestyle="--", alpha=0.35)
    eixos.grid(axis="x", alpha=0.12)

def nome_edificio(id_edificio):
    partes = [parte for parte in id_edificio.split("_") if parte.lower() != "education"]
    return " ".join(partes)

pasta_intermedia = "ficheiros_intermedios"
pasta_figuras = "figuras"

os.makedirs(pasta_intermedia, exist_ok=True)
os.makedirs(pasta_figuras, exist_ok=True)

caminho_base_modelacao = os.path.join(pasta_intermedia, "02_base_modelacao_consumo_e_temperatura.csv")
caminho_estatisticas = os.path.join(pasta_intermedia, "03_estatisticas_descritivas_consumo_real_por_edificio.csv")

print("Análise exploratória")
base_modelacao = pd.read_csv(caminho_base_modelacao, sep=";", decimal=",", parse_dates=["timestamp"])

colunas_necessarias = {"id_edificio", "timestamp", "consumo_kwh", "temperatura_ar_exterior"}
colunas_em_falta = colunas_necessarias - set(base_modelacao.columns)
if colunas_em_falta:
    raise ValueError("Colunas em falta: " + str(sorted(colunas_em_falta)))

print("Dimensão do dataset:", base_modelacao.shape)

edificios = base_modelacao["id_edificio"].unique()
print("Número de edifícios:", len(edificios))

consumo_medio = base_modelacao.groupby("id_edificio")["consumo_kwh"].mean()
edificio_maior_consumo = consumo_medio.idxmax()
edificio_menor_consumo = consumo_medio.idxmin()


figura, eixos = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)

# 30 dias dão uma janela legível sem perder a sazonalidade semanal
serie_edificio_maior_consumo = base_modelacao[base_modelacao["id_edificio"] == edificio_maior_consumo].head(24 * 30)
serie_edificio_menor_consumo = base_modelacao[base_modelacao["id_edificio"] == edificio_menor_consumo].head(24 * 30)

eixos[0].plot(serie_edificio_maior_consumo["timestamp"], serie_edificio_maior_consumo["consumo_kwh"], color=COR_DESTAQUE, linewidth=1.2)
eixos[0].set_title("A. Edifício com maior consumo médio: " + nome_edificio(edificio_maior_consumo))
eixos[0].set_ylabel("Consumo (kWh)")
formatar_eixo_tempo(eixos[0])

eixos[1].plot(serie_edificio_menor_consumo["timestamp"], serie_edificio_menor_consumo["consumo_kwh"], color=COR_REAL, linewidth=1.2)
eixos[1].set_title("B. Edifício com menor consumo médio: " + nome_edificio(edificio_menor_consumo))
eixos[1].set_ylabel("Consumo (kWh)")
eixos[1].set_xlabel("Data e hora")
formatar_eixo_tempo(eixos[1])

figura.suptitle("Perfis temporais de consumo em edifícios extremos", y=0.98, fontweight="bold")
figura.text(0.01, 0.01, "Nota: o painel B inclui um patamar residual prolongado observado nos dados brutos e não corresponde a zero exato.", fontsize=9)
sns.despine()
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(os.path.join(pasta_figuras, "figura_03_3_series_temporais_edificios_maior_menor_consumo.png"), dpi=300)
plt.close()

plt.figure(figsize=(12.5, 6.5))
ordem_edificios = base_modelacao.groupby("id_edificio")["consumo_kwh"].median().sort_values(ascending=False).index

sns.boxplot(data=base_modelacao, x="id_edificio", y="consumo_kwh", order=ordem_edificios, color=COR_NEUTRA, width=0.65, fliersize=1.8, linewidth=0.8)
plt.yscale("log")
plt.title("Distribuição do consumo por edifício", fontsize=14, fontweight="bold")
plt.suptitle("Edifícios ordenados pela mediana, com eixo vertical em escala logarítmica", y=0.97, fontsize=10)
plt.xlabel("Edifício")
plt.ylabel("Consumo (kWh)")
plt.xticks(range(len(ordem_edificios)), [nome_edificio(edificio) for edificio in ordem_edificios], rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.35)

sns.despine()
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(os.path.join(pasta_figuras, "figura_03_1_distribuicao_consumo_real_por_edificio.png"), dpi=300)
plt.close()

# normalizar por edifício para retirar o efeito da escala absoluta
base_modelacao["consumo_real_normalizado"] = base_modelacao.groupby("id_edificio")["consumo_kwh"].transform(lambda serie_consumo: serie_consumo / serie_consumo.mean())

correlacao_temperatura_consumo = base_modelacao["temperatura_ar_exterior"].corr(base_modelacao["consumo_real_normalizado"])
print("Correlação temperatura/consumo normalizado:", round(correlacao_temperatura_consumo, 3))

valor_absoluto_r = abs(correlacao_temperatura_consumo)
if valor_absoluto_r < 0.1:
    forca_correlacao = "desprezável"
elif valor_absoluto_r < 0.3:
    forca_correlacao = "fraca"
elif valor_absoluto_r < 0.5:
    forca_correlacao = "moderada"
else:
    forca_correlacao = "forte"

figura, eixos = plt.subplots(figsize=(10.5, 6.2))
amostra_consumo_temperatura = base_modelacao.sample(min(5000, len(base_modelacao)), random_state=42)
sns.scatterplot(data=amostra_consumo_temperatura, x="temperatura_ar_exterior", y="consumo_real_normalizado", color=COR_REAL, alpha=0.22, s=24, edgecolor=None, linewidth=0, ax=eixos)
eixos.set_title("Relação entre temperatura do ar e consumo normalizado", fontsize=14, fontweight="bold")
eixos.set_xlabel("Temperatura do ar (°C)")
eixos.set_ylabel("Consumo normalizado\n(consumo / média do edifício)")
eixos.text(0.02, 0.98, "Amostra aleatória: n=" + str(len(amostra_consumo_temperatura)) + " pontos\nCorrelação de Pearson: r = " + str(round(correlacao_temperatura_consumo, 2)) + " (" + forca_correlacao + ")",
        transform=eixos.transAxes, ha="left", va="top", fontsize=9.5,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="#cccccc", boxstyle="round,pad=0.35"))
eixos.grid(axis="both", linestyle="--", alpha=0.25)
sns.despine()
plt.tight_layout()

plt.savefig(os.path.join(pasta_figuras, "figura_03_2_relacao_temperatura_consumo_normalizado.png"), dpi=300)
plt.close()

print("Figuras guardadas em:", pasta_figuras)

estatisticas_consumo = (
    base_modelacao.groupby("id_edificio")["consumo_kwh"]
    .agg(media_kwh="mean",
         desvio_padrao_kwh="std",
         maximo_kwh="max",
         minimo_kwh="min",
         p99_kwh=lambda serie_consumo: serie_consumo.quantile(0.99),
         cv=lambda serie_consumo: serie_consumo.std() / serie_consumo.mean() if serie_consumo.mean() != 0 else np.nan)
    .reset_index()
)

estatisticas_consumo.to_csv(caminho_estatisticas, index=False, sep=";", decimal=",")
print("Estatísticas exportadas para:", caminho_estatisticas)
