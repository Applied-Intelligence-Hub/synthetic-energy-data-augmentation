import os
import matplotlib.pyplot as plt
import pandas as pd


sementes_esperadas = [42, 100, 200, 400, 500]

caminho_previsao_divisao = "resultados/10_1_resumo_previsao_prophet_por_divisao.csv"
caminho_anomalias_divisao = "resultados/10_2_resumo_isolation_forest_anomalias_por_divisao.csv"
caminho_previsao_edificios = "resultados/10_5_previsao_prophet_por_edificio_todas_divisoes.csv"

pasta_resultados = "resultados"
pasta_figuras = "figuras"
os.makedirs(pasta_resultados, exist_ok=True)
os.makedirs(pasta_figuras, exist_ok=True)

print("Gerar tabelas finais")

CORES = {
    "Apenas real": "#1f77b4",
    "Com sintético": "#ff7f0e",
    "Com ruído": "#2ca02c",
}


def formatar_painel(ax):
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)


def gerar_figuras_previsao(caminho):
    resumo_previsao = pd.read_csv(caminho, sep=";", decimal=",")
    resumo_previsao = resumo_previsao.sort_values(by=["percentagem_treino_real", "abordagem_treino"]).reset_index(drop=True)

    # o cenário 100% real serve de referência e não pertence aos cenários de escassez
    resumo_grafico = resumo_previsao[resumo_previsao["percentagem_treino_real"] < 100].copy()
    resumo_grafico["percentagem_str"] = resumo_grafico["percentagem_treino_real"].astype(int).astype(str) + "%"

    ordem_percentagens = ["5%", "10%", "20%"]

    rotulos = {
        "Apenas real": "Real escasso",
        "Com sintético": "Real escasso + sintético",
        "Com ruído": "Real escasso + ruído branco",
    }

    linha_referencia = resumo_previsao[(resumo_previsao["percentagem_treino_real"] == 100) & (resumo_previsao["abordagem_treino"] == "Apenas real")]

    if linha_referencia.empty:
        raise ValueError("Cenário 100% real não encontrado no resumo final.")

    def desenhar(metrica, ylabel, valor_referencia, titulo, nome_ficheiro):
        figura, (eixo_superior, eixo_zoom) = plt.subplots(2, 1, figsize=(10.5, 9.4), gridspec_kw={"height_ratios": [1, 1.15]})

        abordagens = ["Apenas real", "Com sintético", "Com ruído"]
        largura = 0.24
        posicoes = range(len(ordem_percentagens))

        for indice, abordagem in enumerate(abordagens):
            valores = []

            for percentagem_rotulo in ordem_percentagens:
                valores_metrica = resumo_grafico[(resumo_grafico["percentagem_str"] == percentagem_rotulo) & (resumo_grafico["abordagem_treino"] == abordagem)][metrica]
                valores.append(valores_metrica.iloc[0] if not valores_metrica.empty else float("nan"))

            deslocamento = (indice - 1) * largura

            eixo_superior.bar([posicao + deslocamento for posicao in posicoes], valores, width=largura, color=CORES[abordagem], label=rotulos[abordagem], alpha=0.90)

        eixo_superior.axhline(y=valor_referencia, color="gray", linestyle="--", linewidth=1.5, label="Prophet com 100% real")

        eixo_superior.set_title("A. Erro nos cenários de treino avaliados")
        eixo_superior.set_xticks(list(posicoes))
        eixo_superior.set_xticklabels(ordem_percentagens)
        eixo_superior.set_xlabel("Histórico real disponível no treino (%)")
        eixo_superior.set_ylabel(ylabel)
        formatar_painel(eixo_superior)

        # o painel de zoom evita que a escala do cenário real escasso esconda diferenças entre sintético e ruído
        abordagens_zoom = ["Com sintético", "Com ruído"]
        largura_barras_zoom = 0.28

        for indice, abordagem in enumerate(abordagens_zoom):
            valores = []

            for percentagem_rotulo in ordem_percentagens:
                valores_metrica = resumo_grafico[(resumo_grafico["percentagem_str"] == percentagem_rotulo) & (resumo_grafico["abordagem_treino"] == abordagem)][metrica]
                valores.append(valores_metrica.iloc[0] if not valores_metrica.empty else float("nan"))

            deslocamento = (indice - 0.5) * largura_barras_zoom

            eixo_zoom.bar([posicao + deslocamento for posicao in posicoes], valores, width=largura_barras_zoom, color=CORES[abordagem], label=rotulos[abordagem], alpha=0.90)

        eixo_zoom.axhline(y=valor_referencia, color="gray", linestyle="--", linewidth=1.5, label="Prophet com 100% real")

        resumo_zoom = resumo_grafico[resumo_grafico["abordagem_treino"].isin(abordagens_zoom)]

        ymin = min(resumo_zoom[metrica].min(), valor_referencia) * 0.85
        ymax = resumo_zoom[metrica].max() * 1.15

        eixo_zoom.set_ylim(ymin, ymax)
        eixo_zoom.set_title("B. Aproximação aos cenários com complemento")
        eixo_zoom.set_xticks(list(posicoes))
        eixo_zoom.set_xticklabels(ordem_percentagens)
        eixo_zoom.set_xlabel("Histórico real disponível no treino (%)")
        eixo_zoom.set_ylabel(ylabel)
        formatar_painel(eixo_zoom)

        handles, labels = eixo_superior.get_legend_handles_labels()
        figura.legend(handles, labels, title="Treino do Prophet", loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.965))

        figura.suptitle(titulo, fontsize=14, fontweight="bold", y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.92])

        plt.savefig(os.path.join(pasta_figuras, nome_ficheiro), dpi=300)

        plt.close()

    referencia_nrmse = linha_referencia["NRMSE_media"].iloc[0]

    desenhar(metrica="NRMSE_media", ylabel="NRMSE médio (%)", valor_referencia=referencia_nrmse, titulo="Erro de previsão do Prophet por cenário de treino (NRMSE)", nome_ficheiro="figura_06_1_nrmse_prophet_por_cenario_final.png")

    referencia_rmse = linha_referencia["RMSE_media"].iloc[0]

    desenhar(metrica="RMSE_media", ylabel="RMSE médio (kWh)", valor_referencia=referencia_rmse, titulo="Erro de previsão do Prophet por cenário de treino (RMSE)", nome_ficheiro="figura_06_2_rmse_prophet_por_cenario_final.png")

    print("Figuras finais de previsão em:", pasta_figuras)


def extrair_sementes(tabela):
    if "semente_divisao" not in tabela.columns:
        return []

    sementes_encontradas = pd.to_numeric(tabela["semente_divisao"], errors="coerce").dropna().astype(int).unique()
    return sorted(sementes_encontradas.tolist())


def validar_sementes(tabela_previsao, tabela_anomalias, caminho_previsao_edificios):
    sementes_previsao = extrair_sementes(tabela_previsao)
    sementes_anomalias = extrair_sementes(tabela_anomalias)

    inconsistencias = []
    if not sementes_previsao:
        inconsistencias.append(caminho_previsao_divisao + " sem coluna semente_divisao válida")
    if not sementes_anomalias:
        inconsistencias.append(caminho_anomalias_divisao + " sem coluna semente_divisao válida")

    if sementes_previsao and sementes_previsao != sementes_esperadas:
        inconsistencias.append("sementes previsão esperadas " + str(sementes_esperadas) + ", obtidas " + str(sementes_previsao))

    if sementes_anomalias and sementes_anomalias != sementes_esperadas:
        inconsistencias.append("sementes anomalias esperadas " + str(sementes_esperadas) + ", obtidas " + str(sementes_anomalias))

    if sementes_previsao and sementes_anomalias and sementes_previsao != sementes_anomalias:
        inconsistencias.append("10_1 tem " + str(sementes_previsao) + " e 10_2 tem " + str(sementes_anomalias))

    if os.path.exists(caminho_previsao_edificios):
        previsao_por_edificio = pd.read_csv(caminho_previsao_edificios, sep=";", decimal=",")
        sementes_edificios = extrair_sementes(previsao_por_edificio)

        if not sementes_edificios:
            inconsistencias.append(caminho_previsao_edificios + " sem coluna semente_divisao válida")
        elif sementes_edificios != sementes_esperadas:
            inconsistencias.append("sementes por edifício esperadas " + str(sementes_esperadas) + ", obtidas " + str(sementes_edificios))
        elif sementes_previsao and sementes_previsao != sementes_edificios:
            inconsistencias.append("10_5 tem " + str(sementes_edificios) + " e 10_1 tem " + str(sementes_previsao))
        elif sementes_anomalias and sementes_anomalias != sementes_edificios:
            inconsistencias.append("10_5 tem " + str(sementes_edificios) + " e 10_2 tem " + str(sementes_anomalias))

    if inconsistencias:
        raise SystemExit("[Erro] Outputs entre divisões inconsistentes.\n" + "\n".join(inconsistencias))


def formatar_media_desvio_para_tabela(tabela, coluna_media, coluna_desvio):
    # formato "média ± desvio" com vírgula decimal
    return tabela[coluna_media].apply(lambda valor: "{:.3f}".format(valor).replace(".", ",")) + " ± " + tabela[coluna_desvio].apply(lambda valor: "{:.3f}".format(valor).replace(".", ","))


try:
    tabela_previsao = pd.read_csv(caminho_previsao_divisao, sep=";", decimal=",")
    tabela_anomalias_validas = pd.read_csv(caminho_anomalias_divisao, sep=";", decimal=",")
    validar_sementes(tabela_previsao, tabela_anomalias_validas, caminho_previsao_edificios)

    colunas_previsao = tabela_previsao.columns.tolist()

    rmse_formatado = (
        tabela_previsao.groupby(["percentagem_treino_real", "abordagem_treino"])["RMSE_medio"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["percentagem_treino_real", "abordagem_treino"])
        .reset_index(drop=True)
        .rename(columns={"mean": "RMSE_media", "std": "RMSE_desvio"})
    )
    rmse_formatado["RMSE_tabela"] = formatar_media_desvio_para_tabela(rmse_formatado, "RMSE_media", "RMSE_desvio")

    if "NRMSE_medio" in tabela_previsao.columns:
        valores_nrmse = tabela_previsao["NRMSE_medio"].dropna()
        if (valores_nrmse <= 0).any():
            print("[Aviso] Há valores de NRMSE_medio <=0 - verifica o script 6")

    if "NRMSE_medio" in colunas_previsao:
        nrmse_formatado = (
            tabela_previsao.groupby(["percentagem_treino_real", "abordagem_treino"])["NRMSE_medio"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values(["percentagem_treino_real", "abordagem_treino"])
            .reset_index(drop=True)
            .rename(columns={"mean": "NRMSE_media", "std": "NRMSE_desvio"})
        )
        nrmse_formatado["NRMSE_tabela"] = formatar_media_desvio_para_tabela(nrmse_formatado, "NRMSE_media", "NRMSE_desvio")

        if "NRMSE_mediano" in colunas_previsao:
            mediana_nrmse = (
                tabela_previsao.groupby(["percentagem_treino_real", "abordagem_treino"])["NRMSE_mediano"]
                .agg(["mean", "std"])
                .reset_index()
                .rename(columns={"mean": "NRMSE_mediana", "std": "NRMSE_mediana_desvio"})
            )
            nrmse_formatado = nrmse_formatado.merge(mediana_nrmse, on=["percentagem_treino_real", "abordagem_treino"], how="left")

        resumo_previsao_consolidado = rmse_formatado.merge(nrmse_formatado, on=["percentagem_treino_real", "abordagem_treino"], how="left")

        intervalo = (
            tabela_previsao.groupby(["percentagem_treino_real", "abordagem_treino"])["NRMSE_medio"]
            .agg(NRMSE_min="min", NRMSE_max="max")
            .reset_index()
        )
        resumo_previsao_consolidado = resumo_previsao_consolidado.merge(intervalo, on=["percentagem_treino_real", "abordagem_treino"], how="left")
    else:
        resumo_previsao_consolidado = rmse_formatado
        print("[Aviso] Coluna NRMSE_medio não encontrada - verifica o script 6")

    caminho_resumo_previsao_final = os.path.join(pasta_resultados, "11_1_tabela_final_previsao_prophet.csv")
    resumo_previsao_consolidado.to_csv(caminho_resumo_previsao_final, sep=";", decimal=",", index=False, encoding="utf-8-sig")
    print("Resumo de previsão guardado em:", caminho_resumo_previsao_final)
    gerar_figuras_previsao(caminho_resumo_previsao_final)

except FileNotFoundError:
    print(" Ficheiro não encontrado:", caminho_previsao_divisao)
    print(" Corre primeiro o 10.multiplas_execucoes.py")


try:
    tabela_anomalias = pd.read_csv(caminho_anomalias_divisao, sep=";", decimal=",")

    colunas_anomalias = ["precisao", "sensibilidade", "f1"]
    for coluna in ["f1_contaminacao_baixa", "f1_contaminacao_alta", "amplitude_f1_contaminacao"]:
        if coluna in tabela_anomalias.columns:
            colunas_anomalias.append(coluna)

    resumo_anomalias_consolidado = (
        tabela_anomalias.groupby("cenario_treino")[colunas_anomalias]
        .agg(["mean", "std"])
        .reset_index()
    )
    resumo_anomalias_consolidado.columns = ["_".join(coluna).strip("_") if isinstance(coluna, tuple) else coluna for coluna in resumo_anomalias_consolidado.columns]

    mapa_tabelas_finais = {
        "precisao_mean": "precisao_media",
        "precisao_std": "precisao_desvio",
        "sensibilidade_mean": "sensibilidade_media",
        "sensibilidade_std": "sensibilidade_desvio",
        "f1_mean": "f1_media",
        "f1_std": "f1_desvio",
        "f1_contaminacao_baixa_mean": "f1_contaminacao_baixa_media",
        "f1_contaminacao_baixa_std": "f1_contaminacao_baixa_desvio",
        "f1_contaminacao_alta_mean": "f1_contaminacao_alta_media",
        "f1_contaminacao_alta_std": "f1_contaminacao_alta_desvio",
        "amplitude_f1_contaminacao_mean": "amplitude_f1_contaminacao_media",
        "amplitude_f1_contaminacao_std": "amplitude_f1_contaminacao_desvio",
    }
    resumo_anomalias_consolidado = resumo_anomalias_consolidado.rename(columns=mapa_tabelas_finais)

    ordem_cenarios = {"100_real": 0, "5_real": 1, "5_real_sintetico": 2, "10_real": 3, "10_real_sintetico": 4, "20_real": 5, "20_real_sintetico": 6}

    resumo_anomalias_consolidado["ordem_cenario"] = resumo_anomalias_consolidado["cenario_treino"].map(ordem_cenarios).fillna(999)
    resumo_anomalias_consolidado = resumo_anomalias_consolidado.sort_values("ordem_cenario").drop(columns=["ordem_cenario"]).reset_index(drop=True)

    for metrica in ["precisao", "sensibilidade", "f1"]:
        resumo_anomalias_consolidado[metrica + "_tabela"] = formatar_media_desvio_para_tabela(resumo_anomalias_consolidado, metrica + "_media", metrica + "_desvio")

    if {"f1_contaminacao_baixa_media", "f1_contaminacao_alta_media"}.issubset(resumo_anomalias_consolidado.columns):
        limite_inferior_f1 = resumo_anomalias_consolidado[["f1_contaminacao_baixa_media", "f1_contaminacao_alta_media"]].min(axis=1)
        limite_superior_f1 = resumo_anomalias_consolidado[["f1_contaminacao_baixa_media", "f1_contaminacao_alta_media"]].max(axis=1)

        resumo_anomalias_consolidado["intervalo_f1_sensibilidade_contaminacao"] = limite_inferior_f1.apply(lambda valor: "{:.3f}".format(valor).replace(".", ",")) + " – " + limite_superior_f1.apply(lambda valor: "{:.3f}".format(valor).replace(".", ","))

    caminho_resumo_anomalias_final = os.path.join(pasta_resultados, "11_2_tabela_final_isolation_forest_anomalias.csv")
    resumo_anomalias_consolidado.to_csv(caminho_resumo_anomalias_final, sep=";", decimal=",", index=False, encoding="utf-8-sig")
    print("Resumo de anomalias guardado em:", caminho_resumo_anomalias_final)

except FileNotFoundError:
    print(" Ficheiro não encontrado:", caminho_anomalias_divisao)
    print(" Corre primeiro o 10.multiplas_execucoes.py")
