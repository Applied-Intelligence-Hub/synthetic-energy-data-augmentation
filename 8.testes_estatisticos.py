import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2, shapiro, wilcoxon


sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})

COR_REAL_SINTETICO = "#1f77b4"
COR_RUIDO = "#2ca02c"
COR_SIG = "#2ca02c"
COR_NAO_SIG = "#d62728"

pasta_resultados = "resultados"
pasta_figuras = "figuras"

os.makedirs(pasta_figuras, exist_ok=True)
os.makedirs(pasta_resultados, exist_ok=True)

caminho_previsao_consolidada = os.path.join(pasta_resultados, "10_5_previsao_prophet_por_edificio_todas_divisoes.csv")

print("Testes estatísticos (Wilcoxon + Fisher)")

if not os.path.exists(caminho_previsao_consolidada):
    raise FileNotFoundError("Falta o ficheiro consolidado de previsão. Corre primeiro o script 10.")

resultados_previsao = pd.read_csv(caminho_previsao_consolidada, sep=";", decimal=",")

colunas_necessarias = {"semente_divisao", "id_edificio", "cenario_treino", "NRMSE"}
colunas_em_falta = colunas_necessarias - set(resultados_previsao.columns)
if colunas_em_falta:
    raise ValueError("Faltam colunas: " + str(sorted(colunas_em_falta)))

n_divisoes = resultados_previsao["semente_divisao"].nunique()
print("Splits:", n_divisoes)


def wilcoxon_seguro(valores_cenario_a, valores_cenario_b):
    # protege contra o caso em que todas as diferenças são zero, onde o Wilcoxon falha
    n_observacoes = len(valores_cenario_a)
    diferencas = np.array(valores_cenario_a) - np.array(valores_cenario_b)
    if np.all(diferencas == 0):
        return 0.0, 1.0, 0.0, 0.0

    estatistica_wilcoxon, p_valor = wilcoxon(valores_cenario_a, valores_cenario_b)
    z_wilcoxon = (estatistica_wilcoxon - (n_observacoes * (n_observacoes + 1)) / 4) / np.sqrt((n_observacoes * (n_observacoes + 1) * (2 * n_observacoes + 1)) / 24)
    tamanho_efeito = abs(z_wilcoxon) / np.sqrt(n_observacoes)
    return float(estatistica_wilcoxon), float(p_valor), float(tamanho_efeito), float(z_wilcoxon)


def classificar_efeito(tamanho_efeito):
    if tamanho_efeito >= 0.5:
        return "Forte"
    if tamanho_efeito >= 0.3:
        return "Moderado"
    return "Fraco"


def analisar_cenario(tabela, nivel, sufixo_cenario_a, sufixo_cenario_b, nome_cenario_a, nome_cenario_b):
    linhas_resultado = []

    for semente_divisao, grupo_divisao in tabela.groupby("semente_divisao"):
        valores_cenario_a = grupo_divisao[grupo_divisao["cenario_treino"] == str(nivel) + "_" + sufixo_cenario_a][["id_edificio", "NRMSE"]].rename(columns={"NRMSE": "NRMSE_" + nome_cenario_a})
        valores_cenario_b = grupo_divisao[grupo_divisao["cenario_treino"] == str(nivel) + "_" + sufixo_cenario_b][["id_edificio", "NRMSE"]].rename(columns={"NRMSE": "NRMSE_" + nome_cenario_b})

        comparacao_cenarios = pd.merge(valores_cenario_a, valores_cenario_b, on="id_edificio", how="inner")
        if comparacao_cenarios.empty or len(comparacao_cenarios) < 2:
            continue

        n_observacoes = len(comparacao_cenarios)
        diferencas = comparacao_cenarios["NRMSE_" + nome_cenario_a].values - comparacao_cenarios["NRMSE_" + nome_cenario_b].values
        if n_observacoes >= 3:
            _, p_shapiro = shapiro(diferencas)
        else:
            p_shapiro = np.nan

        _, p_wilcoxon, tamanho_efeito, _ = wilcoxon_seguro(comparacao_cenarios["NRMSE_" + nome_cenario_a].values, comparacao_cenarios["NRMSE_" + nome_cenario_b].values)

        melhoria = (diferencas / (comparacao_cenarios["NRMSE_" + nome_cenario_a].values + 1e-6) * 100).mean()

        linhas_resultado.append({
            "semente_divisao": semente_divisao,
            "n_edificios": n_observacoes,
            "p_valor_shapiro": round(p_shapiro, 4),
            "p_valor_wilcoxon": round(p_wilcoxon, 4),
            "significativo_0_05": "Sim" if p_wilcoxon < 0.05 else "Não",
            "tamanho_efeito_r": round(tamanho_efeito, 4),
            "classificacao_efeito_r": classificar_efeito(tamanho_efeito),
            "melhoria_media_nrmse_pct": round(melhoria, 2),
        })

    if not linhas_resultado:
        return pd.DataFrame(), {}

    tabela_por_divisao = pd.DataFrame(linhas_resultado)

    # Fisher combina os p-valores dos vários splits num único teste
    p_valores_wilcoxon = tabela_por_divisao["p_valor_wilcoxon"].values
    p_valores_ajustados = np.clip(p_valores_wilcoxon, 1e-15, 1.0)
    fisher = -2 * np.sum(np.log(p_valores_ajustados))
    graus_liberdade_fisher = 2 * len(p_valores_wilcoxon)
    p_valor_fisher = 1 - chi2.cdf(fisher, graus_liberdade_fisher)

    tamanho_efeito_medio = tabela_por_divisao["tamanho_efeito_r"].mean()
    numero_divisoes_significativas = (tabela_por_divisao["p_valor_wilcoxon"] < 0.05).sum()
    melhoria_media = tabela_por_divisao["melhoria_media_nrmse_pct"].mean()

    resumo_geral = {
        "comparacao_cenario": str(nivel) + "% " + nome_cenario_a + " vs " + nome_cenario_b,
        "n_divisoes": len(tabela_por_divisao),
        "n_divisoes_significativas": int(numero_divisoes_significativas),
        "percentagem_divisoes_significativas": round(numero_divisoes_significativas / len(tabela_por_divisao) * 100, 1),
        "fisher_chi2": round(fisher, 4),
        "fisher_graus_liberdade": graus_liberdade_fisher,
        "p_valor_fisher": round(p_valor_fisher, 4),
        "significativo_fisher_0_05": "Sim" if p_valor_fisher < 0.05 else "Não",
        "tamanho_efeito_medio_r": round(tamanho_efeito_medio, 4),
        "classificacao_efeito_r": classificar_efeito(tamanho_efeito_medio),
        "melhoria_media_nrmse_pct": round(melhoria_media, 2),
    }

    return tabela_por_divisao, resumo_geral


resultados_real_vs_sintetico = []
testes_por_divisao_real_vs_sintetico = []

# real escasso vs real escasso + sintético
for nivel in [5, 10, 20]:
    tabela_por_divisao, resumo_geral = analisar_cenario(resultados_previsao, nivel, sufixo_cenario_a="real", sufixo_cenario_b="real_sintetico", nome_cenario_a="real", nome_cenario_b="sintetico")
    if resumo_geral:
        resumo_geral["rotulo_cenario"] = str(nivel) + "% de dados"
        resultados_real_vs_sintetico.append(resumo_geral)
        tabela_por_divisao["nivel"] = nivel
        tabela_por_divisao["comparacao"] = "real vs real+sintetico"
        testes_por_divisao_real_vs_sintetico.append(tabela_por_divisao)

tabela_sintetico = pd.DataFrame(resultados_real_vs_sintetico)
tabela_sintetico_por_divisao = pd.concat(testes_por_divisao_real_vs_sintetico, ignore_index=True) if testes_por_divisao_real_vs_sintetico else pd.DataFrame()

# sintético vs ruído branco como controlo
resultados_ruido_vs_sintetico = []
testes_por_divisao_ruido_vs_sintetico = []

for nivel in [5, 10, 20]:
    tabela_por_divisao, resumo_geral = analisar_cenario(resultados_previsao, nivel, sufixo_cenario_a="real_ruido", sufixo_cenario_b="real_sintetico", nome_cenario_a="ruido", nome_cenario_b="sintetico")
    if resumo_geral:
        resumo_geral["rotulo_cenario"] = str(nivel) + "% sintético vs ruído"
        resultados_ruido_vs_sintetico.append(resumo_geral)
        tabela_por_divisao["nivel"] = nivel
        tabela_por_divisao["comparacao"] = "ruido vs sintetico"
        testes_por_divisao_ruido_vs_sintetico.append(tabela_por_divisao)

tabela_ruido = pd.DataFrame(resultados_ruido_vs_sintetico)
tabela_ruido_por_divisao = pd.concat(testes_por_divisao_ruido_vs_sintetico, ignore_index=True) if testes_por_divisao_ruido_vs_sintetico else pd.DataFrame()

tabela_testes_estatisticos_consolidados = pd.concat([tabela_sintetico, tabela_ruido], ignore_index=True)
tabela_testes_estatisticos_consolidados.to_csv(os.path.join(pasta_resultados, "08_testes_wilcoxon_fisher_previsao_consolidados.csv"), index=False, sep=";", decimal=",")

tabela_testes_estatisticos_por_divisao = pd.concat([tabela_sintetico_por_divisao, tabela_ruido_por_divisao], ignore_index=True)
if not tabela_testes_estatisticos_por_divisao.empty:
    tabela_testes_estatisticos_por_divisao.to_csv(os.path.join(pasta_resultados, "08_1_testes_wilcoxon_previsao_por_divisao.csv"), index=False, sep=";", decimal=",")

figura, eixos = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)

for eixo, (tabela_para_grafico, titulo, cor_base) in zip(eixos, [(tabela_sintetico, "A. Real vs. real + sintético", COR_REAL_SINTETICO), (tabela_ruido, "B. Sintético vs. ruído", COR_RUIDO)]):
    if tabela_para_grafico.empty:
        eixo.text(0.5, 0.5, "Sem dados", ha="center", va="center", transform=eixo.transAxes)
        eixo.set_title(titulo, fontsize=12)
        continue

    cores_pontos = [COR_SIG if p_valor < 0.05 else COR_NAO_SIG for p_valor in tabela_para_grafico["p_valor_fisher"]]
    eixo.scatter(tabela_para_grafico["rotulo_cenario"], tabela_para_grafico["p_valor_fisher"], color=cores_pontos, s=170, zorder=3, edgecolor="black", linewidth=0.7)
    eixo.plot(tabela_para_grafico["rotulo_cenario"], tabela_para_grafico["p_valor_fisher"], color=cor_base, alpha=0.35, linewidth=1.5, zorder=2)
    eixo.axhline(0.05, color="gray", linestyle="--", linewidth=1.5, label="a = 0.05")
    eixo.axhspan(0, 0.05, facecolor=COR_SIG, alpha=0.08)
    limite_superior_eixo_y = max(0.20, tabela_para_grafico["p_valor_fisher"].max() + 0.05)
    eixo.set_ylim(0, limite_superior_eixo_y)
    eixo.set_title(titulo, fontsize=12, fontweight="bold")
    eixo.set_xlabel("Cenário")
    eixo.tick_params(axis="x", rotation=20)
    eixo.legend(loc="upper right", fontsize=9)
    eixo.grid(axis="y", linestyle="--", alpha=0.35)

    tamanho_efeito_medio = tabela_para_grafico["tamanho_efeito_medio_r"].mean()
    classif = classificar_efeito(tamanho_efeito_medio)
    n_significativos_fisher = int((tabela_para_grafico["p_valor_fisher"] < 0.05).sum())
    n_total_comparacoes = len(tabela_para_grafico)
    nota_direcao = "\nDireção varia entre splits" if titulo.startswith("B.") else ""
    eixo.text(0.5, 0.90,
              "Tamanho do efeito médio: r ≈ " + str(round(tamanho_efeito_medio, 2)) + " (" + classif + ")\nCenários com Fisher significativo: " + str(n_significativos_fisher) + "/" + str(n_total_comparacoes) + nota_direcao,
              ha="center", va="center", transform=eixo.transAxes, fontsize=9,
              bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", boxstyle="round,pad=0.4"))

eixos[0].set_ylabel("p-valor (Fisher combinado)", fontweight="bold")
figura.suptitle("Significância estatística por cenário de escassez (" + str(n_divisoes) + " splits)", fontsize=13, fontweight="bold")
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(pasta_figuras, "figura_08_1_p_valores_fisher_previsao.png"), dpi=300)
plt.close()

if not tabela_testes_estatisticos_por_divisao.empty:
    niveis = sorted(tabela_testes_estatisticos_por_divisao["nivel"].unique())
    comparacoes = tabela_testes_estatisticos_por_divisao["comparacao"].unique().tolist()
    numero_comparacoes = len(comparacoes)
    
    figura_detalhe, eixo_detalhe = plt.subplots(1, numero_comparacoes, figsize=(6.2 * numero_comparacoes, 5.4), sharey=True)
    if numero_comparacoes == 1:
        eixo_detalhe = [eixo_detalhe]

    for eixo, comparacao_cenarios in zip(eixo_detalhe, comparacoes):
        subtabela = tabela_testes_estatisticos_por_divisao[tabela_testes_estatisticos_por_divisao["comparacao"] == comparacao_cenarios]
        for nivel in niveis:
            subtabela_nivel = subtabela[subtabela["nivel"] == nivel]
            gerador_jitter = np.random.default_rng(100 + int(nivel))
            jitter = gerador_jitter.uniform(-0.08, 0.08, size=len(subtabela_nivel))
            for indice, (_, linha) in enumerate(subtabela_nivel.iterrows()):
                cor_ponto = COR_SIG if linha["p_valor_wilcoxon"] < 0.05 else COR_NAO_SIG
                posicao_base = niveis.index(nivel) + 1
                posicao_x = posicao_base + jitter[indice]
                eixo.scatter(posicao_x, linha["p_valor_wilcoxon"], color=cor_ponto, s=70, alpha=0.75, zorder=3, edgecolor="white", linewidth=0.4)
        eixo.axhline(0.05, color="gray", linestyle="--", linewidth=1.5, label="a = 0.05")
        eixo.axhspan(0, 0.05, facecolor=COR_SIG, alpha=0.08)
        titulos_legiveis = {"real vs real+sintetico": "real vs.\nreal+sintético", "ruido vs sintetico": "ruído vs.\nsintético"}
        eixo.set_title(titulos_legiveis.get(comparacao_cenarios, comparacao_cenarios.replace(" vs ", " vs.\n")), fontsize=11, fontweight="bold")
        eixo.set_xlabel("Cenário")
        eixo.set_xticks(range(1, len(niveis) + 1))
        eixo.set_xticklabels([str(n_observacoes) + "%" for n_observacoes in niveis])
        eixo.legend(fontsize=8)
        eixo.grid(axis="y", linestyle="--", alpha=0.35)

    eixo_detalhe[0].set_ylabel("p-valor Wilcoxon (por split)", fontweight="bold")
    figura_detalhe.suptitle("Distribuição dos p-valores de Wilcoxon por split", fontsize=12, fontweight="bold")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_figuras, "figura_08_2_p_valores_wilcoxon_previsao_por_divisao.png"), dpi=300)
    plt.close()

print("\n" + "=" * 70)
print("TABELA 1 - Real vs Real+Sintético")
print("-" * 70)
colunas_impressao = ["rotulo_cenario", "n_divisoes_significativas", "percentagem_divisoes_significativas", "p_valor_fisher", "significativo_fisher_0_05", "tamanho_efeito_medio_r", "classificacao_efeito_r", "melhoria_media_nrmse_pct"]
if not tabela_sintetico.empty:
    print(tabela_sintetico[colunas_impressao].to_string(index=False))
print("=" * 70)
print("TABELA 2 - Sintético vs Ruído")
print("-" * 70)
if not tabela_ruido.empty:
    print(tabela_ruido[colunas_impressao].to_string(index=False))
else:
    print("  Sem dados - verifica se o script 6 gera o cenário 'real_ruido'")
print("=" * 70)
print("Figuras guardadas na pasta:", pasta_figuras)
