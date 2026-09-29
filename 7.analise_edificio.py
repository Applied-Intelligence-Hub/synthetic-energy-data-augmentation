import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


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
COR_MELHORIA = "#2ca02c"
COR_PIORA = "#d62728"

# ordem fixa dos splits para os gráficos saírem sempre na mesma sequência
ORDEM_DIVISOES_FIGURAS = [42, 100, 200, 400, 500]


def ordenar_divisoes_fig(tabela):
    mapa_ordem_divisoes = {semente: indice for indice, semente in enumerate(ORDEM_DIVISOES_FIGURAS)}
    return (
        tabela.copy()
        .assign(ordem_divisao=tabela["semente_divisao"].map(mapa_ordem_divisoes))
        .sort_values("ordem_divisao")
        .drop(columns="ordem_divisao")
        .reset_index(drop=True)
    )

pasta_resultados = "resultados"
pasta_figuras = "figuras"

os.makedirs(pasta_resultados, exist_ok=True)
os.makedirs(pasta_figuras, exist_ok=True)

caminho_previsao_consolidada = os.path.join(pasta_resultados, "10_5_previsao_prophet_por_edificio_todas_divisoes.csv")

if os.path.exists(caminho_previsao_consolidada):
    resultados_previsao = pd.read_csv(caminho_previsao_consolidada, sep=";", decimal=",")

    colunas_necessarias = {"semente_divisao", "id_edificio", "cenario_treino", "NRMSE", "RMSE"}

    colunas_em_falta = colunas_necessarias - set(resultados_previsao.columns)
    if colunas_em_falta:
        raise ValueError("Faltam colunas no ficheiro de previsão: " + str(sorted(colunas_em_falta)))
    n_divisoes = resultados_previsao["semente_divisao"].nunique()
    print("Consolidado:", n_divisoes, "splits.")
else:
    raise FileNotFoundError("Ficheiro consolidado não encontrado. Corre primeiro o script 10.")

print("Calcular ganhos por split (20%, NRMSE)")

registos_divisao = []
registos_edificio = []

for semente_divisao, grupo_divisao in resultados_previsao.groupby("semente_divisao"):
    nrmse_real = grupo_divisao[grupo_divisao["cenario_treino"] == "20_real"][["id_edificio", "NRMSE"]].rename(columns={"NRMSE": "NRMSE_real"})
    nrmse_sintetico = grupo_divisao[grupo_divisao["cenario_treino"] == "20_real_sintetico"][["id_edificio", "NRMSE"]].rename(columns={"NRMSE": "NRMSE_sintetico"})
    comparacao_cenarios = pd.merge(nrmse_real, nrmse_sintetico, on="id_edificio")
    if comparacao_cenarios.empty:
        print(" Split", semente_divisao, "sem dados para o cenário 20%")
        continue

    rmse_real = grupo_divisao[grupo_divisao["cenario_treino"] == "20_real"][["id_edificio", "RMSE"]].rename(columns={"RMSE": "RMSE_real"})
    rmse_sintetico = grupo_divisao[grupo_divisao["cenario_treino"] == "20_real_sintetico"][["id_edificio", "RMSE"]].rename(columns={"RMSE": "RMSE_sintetico"})
    comparacao_cenarios = comparacao_cenarios.merge(rmse_real, on="id_edificio").merge(rmse_sintetico, on="id_edificio")

    comparacao_cenarios["ganho_nrmse_pct"] = ((comparacao_cenarios["NRMSE_real"] - comparacao_cenarios["NRMSE_sintetico"]) / comparacao_cenarios["NRMSE_real"]) * 100
    comparacao_cenarios["semente_divisao"] = semente_divisao
    registos_edificio.append(comparacao_cenarios)

    ganhos_nrmse = comparacao_cenarios["ganho_nrmse_pct"].values
    registos_divisao.append({
        "semente_divisao": semente_divisao,
        "n_edificios": len(comparacao_cenarios),
        "ganho_mediano_nrmse_pct": float(np.median(ganhos_nrmse)),
        "ganho_medio_nrmse_pct": float(np.mean(ganhos_nrmse)),
        "percentagem_edificios_com_melhoria": float((ganhos_nrmse > 0).mean() * 100),
        "NRMSE_mediano_real": float(comparacao_cenarios["NRMSE_real"].median()),
        "NRMSE_mediano_sintetico": float(comparacao_cenarios["NRMSE_sintetico"].median()),
        "RMSE_mediano_real": float(comparacao_cenarios["RMSE_real"].median()),
        "RMSE_mediano_sintetico": float(comparacao_cenarios["RMSE_sintetico"].median()),
    })

ganhos_por_divisao = pd.DataFrame(registos_divisao)
ganhos_por_edificio = pd.concat(registos_edificio, ignore_index=True)

if not registos_divisao or not registos_edificio:
    raise RuntimeError("Sem dados válidos para comparar 20_real com 20_real_sintetico.")

n_divisoes_validas = len(ganhos_por_divisao)

# bootstrap aplicado às medianas dos splits, evitando contar edifícios do mesmo split como independentes
np.random.seed(42)
n_iteracoes_bootstrap = 5000
medianas = ganhos_por_divisao["ganho_mediano_nrmse_pct"].values

amostras_bootstrap = np.zeros(n_iteracoes_bootstrap)
for indice in range(n_iteracoes_bootstrap):
    amostra_bootstrap = np.random.choice(medianas, size=n_divisoes_validas, replace=True)
    amostras_bootstrap[indice] = np.mean(amostra_bootstrap)

intervalo_confianca_95 = np.percentile(amostras_bootstrap, [2.5, 97.5])

ganho_mediano = float(np.median(medianas))
ganho_medio = float(np.mean(medianas))
desvio_divisoes = float(np.std(medianas, ddof=1)) if n_divisoes_validas > 1 else np.nan
percentagem_edificios_melhoram = float(ganhos_por_divisao["percentagem_edificios_com_melhoria"].mean())
percentagem_divisoes_melhoria_maioritaria = float((ganhos_por_divisao["percentagem_edificios_com_melhoria"] > 50).mean() * 100)

if n_divisoes_validas < 5:
    print("[Aviso] Só", n_divisoes_validas, "splits - IC bootstrap com cautela.")

resumo_bootstrap = pd.DataFrame({
    "metrica": ["total_divisoes", "total_edificios_analisados", "ganho_mediano_geral_nrmse_pct", "ganho_medio_divisoes_nrmse_pct", "desvio_entre_divisoes_nrmse_pct", "ic95_inferior", "ic95_superior", "percentagem_divisoes_melhoria_maioritaria", "percentagem_media_edificios_com_melhoria"],
    "valor": [n_divisoes_validas, len(ganhos_por_edificio), ganho_mediano, ganho_medio, desvio_divisoes, intervalo_confianca_95[0], intervalo_confianca_95[1], percentagem_divisoes_melhoria_maioritaria, percentagem_edificios_melhoram],
})

resumo_bootstrap.to_csv(os.path.join(pasta_resultados, "07_3_bootstrap_ganhos_nrmse_20_real_vs_20_sintetico.csv"), index=False, sep=";", decimal=",")
ganhos_por_divisao.to_csv(os.path.join(pasta_resultados, "07_1_ganhos_nrmse_20_real_vs_20_sintetico_por_divisao.csv"), index=False, sep=";", decimal=",")
ganhos_por_edificio.to_csv(os.path.join(pasta_resultados, "07_2_ganhos_nrmse_20_real_vs_20_sintetico_por_edificio.csv"), index=False, sep=";", decimal=",")
print("Resultados guardados.")

figura, eixo = plt.subplots(figsize=(10, 7.2))

dados_grafico_comparacao_nrmse = ordenar_divisoes_fig(ganhos_por_divisao)

for indice, linha_divisao in dados_grafico_comparacao_nrmse.iterrows():
    cor_linha = COR_MELHORIA if linha_divisao["ganho_mediano_nrmse_pct"] > 0 else COR_PIORA
    eixo.plot([linha_divisao["NRMSE_mediano_real"], linha_divisao["NRMSE_mediano_sintetico"]], [indice, indice], color=cor_linha, zorder=1, alpha=0.6, linewidth=2)
    eixo.scatter(linha_divisao["NRMSE_mediano_real"], indice, color=COR_REAL, s=110, zorder=2)
    eixo.scatter(linha_divisao["NRMSE_mediano_sintetico"], indice, color=COR_SINTETICO, s=110, zorder=3)
    ganho_str = "{:+.1f}%".format(linha_divisao["ganho_mediano_nrmse_pct"])
    eixo.annotate(ganho_str, xy=(max(linha_divisao["NRMSE_mediano_real"], linha_divisao["NRMSE_mediano_sintetico"]), indice), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=cor_linha)

rotulos = ["Split " + str(int(linha_divisao["semente_divisao"])) + "\n(n=" + str(int(linha_divisao["n_edificios"])) + ")" for _, linha_divisao in dados_grafico_comparacao_nrmse.iterrows()]
eixo.set_yticks(range(len(dados_grafico_comparacao_nrmse)))
eixo.set_yticklabels(rotulos, fontsize=10)
eixo.set_xlabel("NRMSE mediano por split (%)", fontsize=12, fontweight="bold")
eixo.set_title("NRMSE mediano por split", fontsize=13, fontweight="bold", pad=15)

legend_elements = [
    Line2D([0], [0], marker="o", color="w", label="20% real", markerfacecolor=COR_REAL, markersize=10),
    Line2D([0], [0], marker="o", color="w", label="20% real + sintético", markerfacecolor=COR_SINTETICO, markersize=10),
    Line2D([0], [0], color=COR_MELHORIA, linewidth=2, label="Redução com sintético"),
]
eixo.legend(handles=legend_elements, loc="lower right", title="Leitura")

texto_resumo_grafico = ("Resumo Geral:\n"
                        "Ganho mediano: {:+.1f}%\n"
                        "IC 95% (bootstrap): [{:+.1f}%, {:+.1f}%]\n"
                        "Splits com melhoria maioritária: {:.0f}%").format(ganho_mediano, intervalo_confianca_95[0], intervalo_confianca_95[1], percentagem_divisoes_melhoria_maioritaria)
eixo.text(0.98, 0.97, texto_resumo_grafico, transform=eixo.transAxes, fontsize=9, va="top", ha="right", bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", boxstyle="round,pad=0.5"))

sns.despine(left=True)
eixo.grid(axis="x", linestyle="--", alpha=0.7)
eixo.grid(axis="y", linestyle="", alpha=0)
figura.suptitle("Comparação por split no cenário de 20% de dados reais", y=0.98, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(pasta_figuras, "figura_07_1_nrmse_mediano_20_real_vs_20_sintetico_por_divisao.png"), dpi=300)
plt.close()

figura_distribuicao_ganhos, eixo_boxplot = plt.subplots(figsize=(10, 5.6))

ordem_divisao = ordenar_divisoes_fig(ganhos_por_divisao)["semente_divisao"].tolist()
etiquetas = ["Split " + str(semente_divisao) for semente_divisao in ordem_divisao]

dados_boxplot_ganhos = [ganhos_por_edificio[ganhos_por_edificio["semente_divisao"] == semente_divisao]["ganho_nrmse_pct"].values for semente_divisao in ordem_divisao]

boxplot = eixo_boxplot.boxplot(dados_boxplot_ganhos, vert=True, patch_artist=True, showfliers=False,
                 medianprops=dict(color="black", linewidth=2),
                 boxprops=dict(facecolor="#dbe9f6", edgecolor="#6f8fb3"),
                 whiskerprops=dict(color="#6f8fb3"),
                 capprops=dict(color="#6f8fb3"))
for patch in boxplot["boxes"]:
    patch.set_alpha(0.95)

gerador_jitter = np.random.default_rng(0)
for indice, dados in enumerate(dados_boxplot_ganhos):
    jitter = gerador_jitter.uniform(-0.15, 0.15, size=len(dados))
    eixo_boxplot.scatter(np.full(len(dados), indice + 1) + jitter, dados, color=COR_REAL, alpha=0.75, s=42, zorder=3)

eixo_boxplot.axhline(0, color="gray", linestyle="--", linewidth=1.5, label="Sem ganho")
eixo_boxplot.set_xticks(range(1, len(etiquetas) + 1))
eixo_boxplot.set_xticklabels(etiquetas, fontsize=10)
eixo_boxplot.set_xlabel("Split")
eixo_boxplot.set_ylabel("Ganho em NRMSE (%)", fontsize=11)
eixo_boxplot.set_title("Distribuição dos ganhos em NRMSE por split", fontsize=12, fontweight="bold")
eixo_boxplot.legend(loc="upper left", fontsize=9)
eixo_boxplot.grid(axis="y", linestyle="--", alpha=0.35)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(pasta_figuras, "figura_07_2_ganhos_nrmse_20_real_vs_20_sintetico_por_edificio.png"), dpi=300)
plt.close()

figura_rmse_mediano, eixo_histograma = plt.subplots(figsize=(10, 5.6))
posicao_x = np.arange(len(dados_grafico_comparacao_nrmse))
largura = 0.35
eixo_histograma.bar(posicao_x - largura/2, dados_grafico_comparacao_nrmse["RMSE_mediano_real"], largura, label="20% real", color=COR_REAL, alpha=0.88)
eixo_histograma.bar(posicao_x + largura/2, dados_grafico_comparacao_nrmse["RMSE_mediano_sintetico"], largura, label="20% real + sintético", color=COR_SINTETICO, alpha=0.88)

eixo_histograma.set_xticks(posicao_x)
eixo_histograma.set_xticklabels(["Split " + str(int(semente_divisao)) for semente_divisao in dados_grafico_comparacao_nrmse["semente_divisao"]], rotation=15)
eixo_histograma.set_xlabel("Split")
eixo_histograma.set_ylabel("RMSE mediano (kWh)", fontsize=11)
eixo_histograma.set_title("RMSE mediano por split: comparação entre 20% real e 20% real + sintético", fontsize=12, fontweight="bold")
eixo_histograma.legend(title="Cenário de treino")
eixo_histograma.grid(axis="y", linestyle="--", alpha=0.35)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(pasta_figuras, "figura_07_3_rmse_mediano_20_real_vs_20_sintetico_por_divisao.png"), dpi=300)
plt.close()

print("Gráficos gerados em:", pasta_figuras)
print("-" * 60)
print("  Ganho mediano geral:", round(ganho_mediano, 2), "%")
print("  IC 95% bootstrap: [", round(intervalo_confianca_95[0], 2), ",", round(intervalo_confianca_95[1], 2), "]")
print("  Desvio entre splits:", round(desvio_divisoes, 2), "p.p.")
print("  Splits com melhoria maioritária:", round(percentagem_divisoes_melhoria_maioritaria), "%")
print("  Edifícios que melhoram:", round(percentagem_edificios_melhoram), "% (média entre splits)")
print("-" * 60)
