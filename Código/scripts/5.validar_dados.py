import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import acf


sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})

COR_REAL = "#1f77b4"
COR_SINTETICO = "#ff7f0e"
COR_DIVERGENTE = "coolwarm"
COR_MAPA = "YlOrRd"

pasta_intermedia = "ficheiros_intermedios"
pasta_figuras = "figuras"

caminho_base_modelacao = os.path.join(pasta_intermedia, "02_base_modelacao_consumo_e_temperatura.csv")
caminho_series_sinteticas = os.path.join(pasta_intermedia, "04_series_sinteticas_consumo_normalizado_por_edificio.csv")
caminho_metricas = os.path.join(pasta_intermedia, "05_metricas_validacao_gerador_sintetico.csv")
caminho_acf = os.path.join(pasta_intermedia, "05_1_validacao_autocorrelacao_real_vs_sintetico.csv")

os.makedirs(pasta_figuras, exist_ok=True)
os.makedirs(pasta_intermedia, exist_ok=True)


def estilo_eixo(eixo, grelha_x=False):
    eixo.grid(axis="y", linestyle="--", alpha=0.35)
    eixo.grid(axis="x", linestyle="--" if grelha_x else "-", alpha=0.12 if grelha_x else 0.08)


print("Validar dados sintéticos")

dados_reais = pd.read_csv(caminho_base_modelacao, sep=";", decimal=",", parse_dates=["timestamp"])
dados_sinteticos = pd.read_csv(caminho_series_sinteticas, sep=";", decimal=",", parse_dates=["timestamp"])

colunas_reais_obrigatorias = {"id_edificio", "timestamp", "consumo_kwh"}
colunas_sinteticas_obrigatorias = {"id_edificio", "timestamp", "consumo_sintetico_normalizado", "periodo"}

colunas_reais_em_falta = colunas_reais_obrigatorias - set(dados_reais.columns)
colunas_sinteticas_em_falta = colunas_sinteticas_obrigatorias - set(dados_sinteticos.columns)

if colunas_reais_em_falta:
    raise ValueError("Faltam colunas na base real: " + str(sorted(colunas_reais_em_falta)))

if colunas_sinteticas_em_falta:
    raise ValueError("Faltam colunas nas séries sintéticas: " + str(sorted(colunas_sinteticas_em_falta)))

dados_validacao = pd.merge(
    dados_reais[["id_edificio", "timestamp", "consumo_kwh"]],
    dados_sinteticos[["id_edificio", "timestamp", "consumo_sintetico_normalizado", "periodo"]],
    on=["id_edificio", "timestamp"],
    how="inner",
)
print("Registos para validação:", dados_validacao.shape)

if dados_validacao.empty:
    raise RuntimeError("A junção real/sintético ficou vazia.")

# escala calculada no período de treino para evitar contaminação do teste
media_treino = dados_validacao[dados_validacao["periodo"] == "treino"].groupby("id_edificio")["consumo_kwh"].mean()
dados_validacao["media_referencia"] = dados_validacao["id_edificio"].map(media_treino)
dados_validacao["media_referencia"] = dados_validacao["media_referencia"].replace(0, np.nan)

# alternativa quando a média de treino é zero ou inválida
media_alternativa = dados_validacao.groupby("id_edificio")["consumo_kwh"].transform("mean")
media_normalizacao = dados_validacao["media_referencia"].fillna(media_alternativa).replace(0, np.nan)

dados_validacao["consumo_real_normalizado"] = dados_validacao["consumo_kwh"] / media_normalizacao

dados_validacao = dados_validacao.dropna(subset=["consumo_real_normalizado", "consumo_sintetico_normalizado"])

if dados_validacao.empty:
    raise RuntimeError("Sem registos após a normalização.")

cv_por_edificio = dados_validacao.groupby("id_edificio")["consumo_kwh"].agg(lambda serie_consumo: serie_consumo.std() / serie_consumo.mean() if serie_consumo.mean() != 0 else np.nan)
cv_por_edificio = cv_por_edificio.dropna()

if cv_por_edificio.empty:
    raise RuntimeError("Não foi possível calcular o CV por edifício.")

edificio_menor_cv = cv_por_edificio.idxmin()
edificio_maior_cv = cv_por_edificio.idxmax()

figura, eixos = plt.subplots(2, 1, figsize=(14, 10.2), sharex=False)

for eixo, (id_edificio, titulo) in zip(eixos, [(edificio_menor_cv, "A. Edifício com menor coeficiente de variação"), (edificio_maior_cv, "B. Edifício com maior coeficiente de variação")]):
    serie_edificio = dados_validacao[dados_validacao["id_edificio"] == id_edificio].reset_index(drop=True)
    if len(serie_edificio) > 336:
        inicio = max((len(serie_edificio) - 336) // 2, 0)
        serie_edificio = serie_edificio.iloc[inicio:inicio + 336]

    media_movel_real = serie_edificio["consumo_real_normalizado"].rolling(window=24, min_periods=1).mean()
    media_movel_sintetica = serie_edificio["consumo_sintetico_normalizado"].rolling(window=24, min_periods=1).mean()

    eixo.plot(serie_edificio["timestamp"], serie_edificio["consumo_real_normalizado"], color=COR_REAL, linewidth=0.9, alpha=0.28)
    eixo.plot(serie_edificio["timestamp"], serie_edificio["consumo_sintetico_normalizado"], color=COR_SINTETICO, linewidth=0.9, alpha=0.28, linestyle="--")
    eixo.plot(serie_edificio["timestamp"], media_movel_real, label="Real (média móvel 24 h)", color=COR_REAL, linewidth=2.2)
    eixo.plot(serie_edificio["timestamp"], media_movel_sintetica, label="Sintético (média móvel 24 h)", color=COR_SINTETICO, linewidth=2.2, linestyle="--")
    coeficiente_variacao_atual = cv_por_edificio.loc[id_edificio]
    nome_edificio = " ".join(parte for parte in id_edificio.split("_") if parte.lower() != "education")

    eixo.set_title(titulo + " (CV=" + str(round(coeficiente_variacao_atual, 3)) + "): " + nome_edificio)
    eixo.set_ylabel("Consumo normalizado")
    estilo_eixo(eixo, grelha_x=True)
    eixo.legend(loc="upper right", frameon=True, title="Médias móveis")

eixos[1].set_xlabel("Data e hora")
figura.suptitle("Comparação entre séries reais e sintéticas em edifícios com menor e maior coeficiente de variação", y=0.98, fontweight="bold")
figura.text(0.01, 0.01, "Linhas ténues: séries horárias normalizadas. Linhas destacadas: médias móveis de 24 horas.", fontsize=9)
sns.despine()
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(os.path.join(pasta_figuras, "figura_05_3_series_reais_vs_sinteticas_menor_maior_cv.png"), dpi=300)
plt.close()

plt.figure(figsize=(8.5, 5.2))
sns.kdeplot(data=dados_validacao, x="consumo_real_normalizado", label="Consumo real normalizado", fill=True, color=COR_REAL, alpha=0.28, linewidth=2, cut=0, clip=(0, None))
sns.kdeplot(data=dados_validacao, x="consumo_sintetico_normalizado", label="Consumo sintético normalizado", fill=True, color=COR_SINTETICO, alpha=0.28, linewidth=2, cut=0, clip=(0, None))

plt.title("Distribuição dos consumos normalizados", fontsize=14, fontweight="bold")
plt.suptitle("Zona central da distribuição", y=0.97, fontsize=10)
plt.xlabel("Consumo normalizado")
plt.ylabel("Densidade")
plt.xlim(0, 2.0)
plt.legend(frameon=True)
plt.grid(axis="y", linestyle="--", alpha=0.35)
sns.despine()
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(pasta_figuras, "figura_05_1a_distribuicao_real_vs_sintetico_zona_central.png"), dpi=300)
plt.close()

plt.figure(figsize=(8.5, 5.2))
sns.kdeplot(data=dados_validacao, x="consumo_real_normalizado", label="Consumo real normalizado", fill=True, color=COR_REAL, alpha=0.24, linewidth=2, cut=0, clip=(0, None))
sns.kdeplot(data=dados_validacao, x="consumo_sintetico_normalizado", label="Consumo sintético normalizado", fill=True, color=COR_SINTETICO, alpha=0.24, linewidth=2, cut=0, clip=(0, None))

limite_superior = max(dados_validacao["consumo_real_normalizado"].quantile(0.999), dados_validacao["consumo_sintetico_normalizado"].quantile(0.999))

plt.title("Distribuição dos consumos normalizados", fontsize=14, fontweight="bold")
plt.suptitle("Distribuição com inclusão das caudas", y=0.97, fontsize=10)
plt.xlabel("Consumo normalizado")
plt.ylabel("Densidade")
plt.xlim(0, limite_superior * 1.05)
plt.legend(frameon=True)
plt.grid(axis="y", linestyle="--", alpha=0.35)
sns.despine()
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(pasta_figuras, "figura_05_1b_distribuicao_real_vs_sintetico_com_caudas.png"), dpi=300)
plt.close()

dados_perfil_horario_semanal = dados_validacao.copy()
dados_perfil_horario_semanal["hora_do_dia"] = dados_perfil_horario_semanal["timestamp"].dt.hour
dados_perfil_horario_semanal["dia_da_semana"] = dados_perfil_horario_semanal["timestamp"].dt.dayofweek

perfil_real = dados_perfil_horario_semanal.pivot_table(values="consumo_real_normalizado", index="hora_do_dia", columns="dia_da_semana", aggfunc="mean")
perfil_sintetico = dados_perfil_horario_semanal.pivot_table(values="consumo_sintetico_normalizado", index="hora_do_dia", columns="dia_da_semana", aggfunc="mean")

diferenca = perfil_real - perfil_sintetico

dias_semana = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

perfil_real = perfil_real.reindex(index=range(24), columns=range(7))
perfil_sintetico = perfil_sintetico.reindex(index=range(24), columns=range(7))
diferenca = diferenca.reindex(index=range(24), columns=range(7))

for tabela_perfil in [perfil_real, perfil_sintetico, diferenca]:
    tabela_perfil.columns = dias_semana

figura, eixos = plt.subplots(1, 3, figsize=(20, 6.4), sharey=True)
limite_mapa_calor = max(perfil_real.max().max(), perfil_sintetico.max().max())

sns.heatmap(perfil_real, ax=eixos[0], cmap=COR_MAPA, vmin=0, vmax=limite_mapa_calor, cbar_kws={"label": "Consumo médio normalizado"})
eixos[0].set_title("A. Perfil horário médio real", fontsize=13, fontweight="bold")
eixos[0].set_xlabel("Dia da semana")
eixos[0].set_ylabel("Hora do dia")
eixos[0].invert_yaxis()

sns.heatmap(perfil_sintetico, ax=eixos[1], cmap=COR_MAPA, vmin=0, vmax=limite_mapa_calor, cbar_kws={"label": "Consumo médio normalizado"})
eixos[1].set_title("B. Perfil horário médio sintético", fontsize=13, fontweight="bold")
eixos[1].set_xlabel("Dia da semana")
eixos[1].set_ylabel("")
eixos[1].invert_yaxis()

limite_diferenca = np.nanmax(np.abs(diferenca.values))
sns.heatmap(diferenca, ax=eixos[2], cmap=COR_DIVERGENTE, center=0, vmin=-limite_diferenca, vmax=limite_diferenca, cbar_kws={"label": "Diferença normalizada"})
eixos[2].set_title("C. Diferença direta (real - sintético)", fontsize=13, fontweight="bold")
eixos[2].set_xlabel("Dia da semana")
eixos[2].set_ylabel("")
eixos[2].invert_yaxis()

figura.suptitle("Perfis horários médios por dia da semana", y=0.98, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(pasta_figuras, "figura_05_2_perfil_horario_semanal_real_vs_sintetico.png"), dpi=300)
plt.close()

dados_treino_validacao = dados_validacao[dados_validacao["periodo"] == "treino"].sort_values(["id_edificio", "timestamp"]).reset_index(drop=True)

if dados_treino_validacao.empty:
    raise RuntimeError("Sem registos de treino para validar.")

media_real = dados_treino_validacao.groupby("id_edificio")["consumo_real_normalizado"].mean().mean()
media_sintetica = dados_treino_validacao.groupby("id_edificio")["consumo_sintetico_normalizado"].mean().mean()

desvio_real = dados_treino_validacao.groupby("id_edificio")["consumo_real_normalizado"].std().mean()
desvio_sintetico = dados_treino_validacao.groupby("id_edificio")["consumo_sintetico_normalizado"].std().mean()

percentil_99_real = dados_treino_validacao.groupby("id_edificio")["consumo_real_normalizado"].quantile(0.99).mean()
percentil_99_sintetico = dados_treino_validacao.groupby("id_edificio")["consumo_sintetico_normalizado"].quantile(0.99).mean()

rmse = dados_treino_validacao.groupby("id_edificio").apply(
    lambda grupo_edificio: np.sqrt(np.mean((grupo_edificio["consumo_real_normalizado"] - grupo_edificio["consumo_sintetico_normalizado"]) ** 2))
).mean()

print("RMSE normalizado do gerador:", round(rmse, 4))

ks_por_edificio = dados_treino_validacao.groupby("id_edificio").apply(
    lambda grupo_edificio: ks_2samp(
        grupo_edificio["consumo_real_normalizado"].sample(min(1000, len(grupo_edificio)), random_state=42),
        grupo_edificio["consumo_sintetico_normalizado"].sample(min(1000, len(grupo_edificio)), random_state=42),
    )
)
estatistica_ks = ks_por_edificio.apply(lambda resultado_ks: resultado_ks[0]).mean()
taxa_rejeicao = ks_por_edificio.apply(lambda resultado_ks: resultado_ks[1] < 0.05).mean()

print("Estatística KS média:", round(estatistica_ks, 4))
print("Taxa de rejeição H0 (alfa=0.05):", round(taxa_rejeicao * 100, 2), "%")
print("KS: a rejeição de H0 é esperada - o gerador não replica a distribuição dos valores reais.")

metricas = pd.DataFrame({
    "metrica": ["media_normalizada", "desvio_padrao_normalizado", "percentil_99_normalizado", "rmse_normalizado_gerador", "estatistica_ks", "taxa_rejeicao_ks"],
    "real": [media_real, desvio_real, percentil_99_real, np.nan, np.nan, np.nan],
    "sintetico": [media_sintetica, desvio_sintetico, percentil_99_sintetico, rmse, estatistica_ks, taxa_rejeicao],
})
metricas.loc[:2, "diferenca_pct"] = abs(metricas.loc[:2, "sintetico"] - metricas.loc[:2, "real"]) / (metricas.loc[:2, "real"] + 1e-9) * 100

numero_maximo_lags_acf = 48
registos_acf = []
for id_edificio in dados_treino_validacao["id_edificio"].unique():
    serie_real_limpa = dados_treino_validacao[dados_treino_validacao["id_edificio"] == id_edificio]["consumo_real_normalizado"].values
    serie_sintetica_limpa = dados_treino_validacao[dados_treino_validacao["id_edificio"] == id_edificio]["consumo_sintetico_normalizado"].values

    # 48 lags cobrem dois ciclos diários completos
    autocorrelacao_real = acf(serie_real_limpa, nlags=numero_maximo_lags_acf, fft=True)
    autocorrelacao_sintetica = acf(serie_sintetica_limpa, nlags=numero_maximo_lags_acf, fft=True)
    erro_medio_autocorrelacao = np.mean(np.abs(autocorrelacao_real - autocorrelacao_sintetica))
    registos_acf.append({"id_edificio": id_edificio, "erro_medio_autocorrelacao_48_lags": erro_medio_autocorrelacao})

tabela_acf = pd.DataFrame(registos_acf)
tabela_acf.to_csv(caminho_acf, index=False, sep=";", decimal=",")
print("MAE médio da ACF (48 lags):", round(tabela_acf["erro_medio_autocorrelacao_48_lags"].mean(), 4))

metricas.to_csv(caminho_metricas, index=False, sep=";", decimal=",")
print("Métricas guardadas. Figuras em:", pasta_figuras)
