import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import logging
import time
import seaborn as sns


logging.getLogger('prophet').setLevel(logging.WARNING)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})

CORES = {
    "Apenas real": "#1f77b4",
    "Com sintético": "#ff7f0e",
    "Com ruído": "#2ca02c",
}

colunas_modelo = ["ds", "y", "temperatura_ar_exterior"]


def formatar_painel(ax):
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", alpha=0.10)


def guardar_csv(tabela_csv, caminho, max_tentativas=5, espera=3, **kwargs):
    # tenta de novo se o CSV estiver aberto no Excel ou bloqueado
    for tentativa in range(max_tentativas):
        try:
            tabela_csv.to_csv(caminho, **kwargs)
            return
        except PermissionError:
            if tentativa < max_tentativas - 1:
                print("  Tentativa", tentativa + 1, "/", max_tentativas, ":", os.path.basename(caminho), "bloqueado, a esperar", espera, "s")
                time.sleep(espera)
            else:
                raise


def valor_ref_100(tabela, coluna_valor):
    base_modelacao = tabela[(tabela["percentagem_treino_real"] == 100) & (tabela["abordagem_treino"] == "Apenas real")]

    if base_modelacao.empty:
        raise ValueError("Sem cenário 100% real para " + coluna_valor)

    return base_modelacao[coluna_valor].iloc[0]


semente_previsao = int(os.getenv("FORECAST_SEED", os.getenv("EXPERIMENT_SEED", "42")))
np.random.seed(semente_previsao)
print("Seed previsão:", semente_previsao)

caminho_base_modelacao = "ficheiros_intermedios/02_base_modelacao_consumo_e_temperatura.csv"
caminho_series_sinteticas = "ficheiros_intermedios/04_series_sinteticas_consumo_normalizado_por_edificio.csv"

pasta_resultados = "resultados"
pasta_figuras = "figuras"

os.makedirs(pasta_resultados, exist_ok=True)
os.makedirs(pasta_figuras, exist_ok=True)

fracoes_treino_real = [0.05, 0.10, 0.20]
minimo_observacoes_treino = 24

executar_sensibilidade = os.getenv("RUN_PROPHET_CPS05", "0") == "1"
cps_sensibilidade = 0.5

print("Carregar dados")

base_modelacao = pd.read_csv(caminho_base_modelacao, sep=';', decimal=',', parse_dates=["timestamp"])
series_sinteticas_normalizadas = pd.read_csv(caminho_series_sinteticas, sep=';', decimal=',', parse_dates=["timestamp"])

dados_previsao = pd.merge(base_modelacao[['id_edificio', 'timestamp', 'consumo_kwh', 'temperatura_ar_exterior']],
                 series_sinteticas_normalizadas[['id_edificio', 'timestamp', 'consumo_sintetico_normalizado', 'periodo']],
                 on=['id_edificio', 'timestamp'], how='inner')

media_treino_por_edificio = dados_previsao[dados_previsao["periodo"] == "treino"].groupby("id_edificio")["consumo_kwh"].mean()

identificadores_edificios = dados_previsao["id_edificio"].unique()
registos = []
registos_sensibilidade = [] if executar_sensibilidade else None


def treinar_e_avaliar(treino, teste, changepoint_prior_scale=None):
    treino = treino.set_index('ds').asfreq('h').reset_index()
    teste = teste.set_index('ds').asfreq('h').reset_index()

    treino = treino.dropna(subset=["y"]).reset_index(drop=True)
    teste = teste.dropna(subset=["y"]).reset_index(drop=True)

    if treino.empty or teste.empty:
        return np.nan, np.nan, np.nan, 0

    usar_temperatura = "temperatura_ar_exterior" in treino.columns

    if usar_temperatura:
        # imputação apenas com a média do treino, para não usar dados do teste
        media_temperatura_treino = treino["temperatura_ar_exterior"].mean()

        if np.isnan(media_temperatura_treino):
            return np.nan, np.nan, np.nan, treino.dropna(subset=["y"]).shape[0]

        treino["temperatura_ar_exterior"] = treino["temperatura_ar_exterior"].interpolate(method="linear", limit=6).fillna(media_temperatura_treino)
        teste["temperatura_ar_exterior"] = teste["temperatura_ar_exterior"].interpolate(method="linear", limit=6).fillna(media_temperatura_treino)

        if treino["temperatura_ar_exterior"].isna().any() or teste["temperatura_ar_exterior"].isna().any():
            return np.nan, np.nan, np.nan, treino.dropna(subset=["y"]).shape[0]

    # sazonalidade anual desligada devido ao histórico curto nos cenários escassos
    modelo_kwargs = dict(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    if changepoint_prior_scale is not None:
        modelo_kwargs["changepoint_prior_scale"] = changepoint_prior_scale

    modelo_prophet = Prophet(**modelo_kwargs)

    if usar_temperatura:
        modelo_prophet.add_regressor("temperatura_ar_exterior")

    modelo_prophet.fit(treino)

    colunas_previsao = ["ds", "temperatura_ar_exterior"] if usar_temperatura else ["ds"]
    previsao_prophet = modelo_prophet.predict(teste[colunas_previsao])

    comparacao_previsao = pd.merge(teste[["ds", "y"]], previsao_prophet[["ds", "yhat"]], on="ds", how="inner").dropna()

    if comparacao_previsao.empty:
        return np.nan, np.nan, np.nan, treino.dropna(subset=["y"]).shape[0]

    valores_reais = comparacao_previsao["y"].values
    valores_previstos = comparacao_previsao["yhat"].values

    mae = mean_absolute_error(valores_reais, valores_previstos)
    rmse = np.sqrt(mean_squared_error(valores_reais, valores_previstos))
    mape = np.mean(np.abs((valores_reais - valores_previstos) / np.maximum(valores_reais, 1e-6))) * 100

    n_observacoes_efetivas = treino.dropna(subset=["y"]).shape[0]

    return mae, rmse, mape, n_observacoes_efetivas


for indice, id_edificio in enumerate(identificadores_edificios):
    print("Treinar Prophet", indice + 1, "/", len(identificadores_edificios), ":", id_edificio)

    dados_edificio = dados_previsao[dados_previsao["id_edificio"] == id_edificio].copy().sort_values("timestamp")

    serie_real = dados_edificio[["timestamp", "consumo_kwh", "temperatura_ar_exterior", "periodo"]].rename(columns={"timestamp": "ds", "consumo_kwh": "y"})
    serie_sintetica = dados_edificio[["timestamp", "consumo_sintetico_normalizado", "temperatura_ar_exterior", "periodo"]].rename(columns={"timestamp": "ds", "consumo_sintetico_normalizado": "y_norm"})

    treino_100 = serie_real[serie_real["periodo"] == "treino"].copy().sort_values("ds")
    teste_real = serie_real[serie_real["periodo"] == "teste"].copy()
    treino_sintetico = serie_sintetica[serie_sintetica["periodo"] == "treino"].copy().sort_values("ds")

    temperatura_treino = dados_edificio[dados_edificio["periodo"] == "treino"][["timestamp", "temperatura_ar_exterior"]].rename(columns={"timestamp": "ds"})

    media_edificio = media_treino_por_edificio.get(id_edificio, np.nan)

    if np.isnan(media_edificio) or media_edificio == 0:
        print("  [Aviso] média de treino inválida para", id_edificio, "- usar mediana")
        media_edificio = dados_previsao[dados_previsao["id_edificio"] == id_edificio]["consumo_kwh"].median()

        if np.isnan(media_edificio) or media_edificio == 0:
            media_edificio = 1.0

    mae, rmse, mape, n_observacoes_efetivas = treinar_e_avaliar(treino_100[colunas_modelo], teste_real[colunas_modelo])
    nrmse = (rmse / media_edificio) * 100
    registos.append([id_edificio, "100_real", 100, "Apenas real", mae, rmse, nrmse, mape, n_observacoes_efetivas, None, None])

    if executar_sensibilidade:
        mae_s, rmse_s, mape_s, n_observacoes_efetivas_sensibilidade = treinar_e_avaliar(treino_100[colunas_modelo], teste_real[colunas_modelo], changepoint_prior_scale=cps_sensibilidade)
        nrmse_s = (rmse_s / media_edificio) * 100
        registos_sensibilidade.append([id_edificio, "100_real", 100, "Apenas real", mae_s, rmse_s, nrmse_s, mape_s, n_observacoes_efetivas_sensibilidade, None, None])

    for fracao_treino in fracoes_treino_real:
        percentagem_treino = int(fracao_treino * 100)
        n_observacoes_reais = max(int(len(treino_100) * fracao_treino), minimo_observacoes_treino)

        treino_reduzido = treino_100.iloc[:n_observacoes_reais].copy()

        meses_treino = treino_reduzido["ds"].dt.month.unique().tolist()
        cobertura = round(len(meses_treino) / 12.0, 4)

        # a escala vem só do real disponível para manter o cenário de escassez
        if len(treino_reduzido) >= 24:
            escala = float(treino_reduzido.assign(_h=treino_reduzido["ds"].dt.hour).groupby("_h")["y"].median().median())
        else:
            escala = float(treino_reduzido["y"].median())

        if np.isnan(escala) or escala <= 0:
            media_alternativa = treino_reduzido["y"].mean()
            escala = float(media_alternativa) if media_alternativa > 0 else 1.0

        cenario_real = str(percentagem_treino) + "_real"
        cenario_sintetico = str(percentagem_treino) + "_real_sintetico"
        cenario_ruido = str(percentagem_treino) + "_real_ruido"

        # # cenário de base: mede o erro quando o Prophet usa apenas a fração real disponível
        mae, rmse, mape, n_observacoes_efetivas = treinar_e_avaliar(treino_reduzido[colunas_modelo], teste_real[colunas_modelo])
        nrmse = (rmse / media_edificio) * 100
        registos.append([id_edificio, cenario_real, percentagem_treino, "Apenas real", mae, rmse, nrmse, mape, n_observacoes_efetivas, str(meses_treino), cobertura])

        if executar_sensibilidade:
            mae_s, rmse_s, mape_s, n_observacoes_efetivas_sensibilidade = treinar_e_avaliar(treino_reduzido[colunas_modelo], teste_real[colunas_modelo], changepoint_prior_scale=cps_sensibilidade)
            nrmse_s = (rmse_s / media_edificio) * 100
            registos_sensibilidade.append([id_edificio, cenario_real, percentagem_treino, "Apenas real", mae_s, rmse_s, nrmse_s, mape_s, n_observacoes_efetivas_sensibilidade, str(meses_treino), cobertura])

        # o sintético completa a janela de treino sem usar consumo real adicional
        dados_sinteticos_para_complemento = treino_sintetico[~treino_sintetico["ds"].isin(treino_reduzido["ds"])].copy()
        dados_sinteticos_para_complemento["y"] = dados_sinteticos_para_complemento["y_norm"] * escala

        dados_sinteticos_para_complemento = dados_sinteticos_para_complemento.merge(temperatura_treino, on="ds", how="left", suffixes=("_old", ""))
        if "temperatura_ar_exterior_old" in dados_sinteticos_para_complemento.columns:
            dados_sinteticos_para_complemento.drop(columns=["temperatura_ar_exterior_old"], inplace=True)

        treino_real_com_complemento_sintetico = pd.concat([treino_reduzido[colunas_modelo], dados_sinteticos_para_complemento[colunas_modelo]]).sort_values("ds")

        mae, rmse, mape, n_observacoes_efetivas = treinar_e_avaliar(treino_real_com_complemento_sintetico, teste_real[colunas_modelo])
        nrmse = (rmse / media_edificio) * 100
        registos.append([id_edificio, cenario_sintetico, percentagem_treino, "Com sintético", mae, rmse, nrmse, mape, n_observacoes_efetivas, str(meses_treino), cobertura])

        if executar_sensibilidade:
            mae_s, rmse_s, mape_s, n_observacoes_efetivas_sensibilidade = treinar_e_avaliar(treino_real_com_complemento_sintetico, teste_real[colunas_modelo], changepoint_prior_scale=cps_sensibilidade)
            nrmse_s = (rmse_s / media_edificio) * 100
            registos_sensibilidade.append([id_edificio, cenario_sintetico, percentagem_treino, "Com sintético", mae_s, rmse_s, nrmse_s, mape_s, n_observacoes_efetivas_sensibilidade, str(meses_treino), cobertura])

        # controlo: real reduzido + ruído branco em vez do sintético, para isolar o efeito da estrutura
        gerador_ruido = np.random.default_rng(semente_previsao + 20000 + percentagem_treino)
        ruido = gerador_ruido.normal(loc=escala, scale=escala * 0.3, size=len(dados_sinteticos_para_complemento))
        ruido = np.maximum(ruido, 0)

        dados_ruido_para_complemento = dados_sinteticos_para_complemento[["ds", "temperatura_ar_exterior"]].copy()
        dados_ruido_para_complemento["y"] = ruido

        treino_real_com_complemento_ruido = pd.concat([treino_reduzido[colunas_modelo], dados_ruido_para_complemento[colunas_modelo]]).sort_values("ds")

        mae_r, rmse_r, mape_r, n_observacoes_efetivas_ruido = treinar_e_avaliar(treino_real_com_complemento_ruido, teste_real[colunas_modelo])
        nrmse_r = (rmse_r / media_edificio) * 100
        registos.append([id_edificio, cenario_ruido, percentagem_treino, "Com ruído", mae_r, rmse_r, nrmse_r, mape_r, n_observacoes_efetivas_ruido, str(meses_treino), cobertura])


colunas_resultado = ["id_edificio", "cenario_treino", "percentagem_treino_real", "abordagem_treino", "MAE", "RMSE", "NRMSE", "MAPE", "n_observacoes_treino", "meses_treino_real", "cobertura_anual_treino_real"]

resultados_por_edificio = pd.DataFrame(registos, columns=colunas_resultado)
guardar_csv(resultados_por_edificio, os.path.join(pasta_resultados, "06_previsao_prophet_por_edificio_divisao_atual.csv"), index=False, sep=";", decimal=",")

resumo = (
    resultados_por_edificio
    .groupby(["percentagem_treino_real", "abordagem_treino"])
    .agg(MAE_medio=("MAE", "mean"),
         RMSE_medio=("RMSE", "mean"),
         NRMSE_medio=("NRMSE", "mean"),
         NRMSE_mediano=("NRMSE", "median"),
         MAPE_medio=("MAPE", "mean"))
    .reset_index()
)
resumo = resumo.sort_values(by=["percentagem_treino_real", "abordagem_treino"]).reset_index(drop=True)

nrmse_ref = valor_ref_100(resumo, "NRMSE_medio")
resumo["melhoria_nrmse_vs_100_real_pct"] = (nrmse_ref - resumo["NRMSE_medio"]) / nrmse_ref * 100

caminho_resumo = os.path.join(pasta_resultados, "06_1_resumo_previsao_prophet_por_cenario_divisao_atual.csv")
guardar_csv(resumo, caminho_resumo, sep=";", decimal=",", index=False)
print("Resumo guardado:", caminho_resumo)

# figuras intermédias, as versões definitivas saem do script 11
resumo_para_grafico = resumo[resumo["percentagem_treino_real"] < 100].copy()
resumo_para_grafico["percentagem_str"] = resumo_para_grafico["percentagem_treino_real"].astype(int).astype(str) + "%"

ordem_percentagens = ["5%", "10%", "20%"]

rotulos_grafico = {
    "Apenas real": "Real escasso",
    "Com sintético": "Real escasso + sintético",
    "Com ruído": "Real escasso + ruído",
}


def desenhar_figura(metrica, ylabel, valor_referencia, titulo, nome_ficheiro):
    figura, (eixo_superior, eixo_zoom) = plt.subplots(2, 1, figsize=(10.5, 9.4), gridspec_kw={"height_ratios": [1, 1.15]})

    abordagens = ["Apenas real", "Com sintético", "Com ruído"]
    largura = 0.24
    posicoes = range(len(ordem_percentagens))

    for indice, abordagem in enumerate(abordagens):
        valores = []

        for percentagem_treino in ordem_percentagens:
            valores_metrica = resumo_para_grafico[(resumo_para_grafico["percentagem_str"] == percentagem_treino) & (resumo_para_grafico["abordagem_treino"] == abordagem)][metrica]
            valores.append(valores_metrica.iloc[0] if not valores_metrica.empty else float("nan"))

        deslocamento = (indice - 1) * largura

        eixo_superior.bar([posicao + deslocamento for posicao in posicoes], valores, width=largura, color=CORES[abordagem], label=rotulos_grafico[abordagem], alpha=0.90)

    eixo_superior.axhline(y=valor_referencia, color="gray", linestyle="--", linewidth=1.5, label="Prophet com 100% real")

    eixo_superior.set_title("A. Erro nos cenários de treino avaliados")
    eixo_superior.set_xticks(list(posicoes))
    eixo_superior.set_xticklabels(ordem_percentagens)
    eixo_superior.set_xlabel("Histórico real disponível no treino (%)")
    eixo_superior.set_ylabel(ylabel)
    formatar_painel(eixo_superior)

    abordagens_zoom = ["Com sintético", "Com ruído"]
    largura_barras_zoom = 0.28

    for indice, abordagem in enumerate(abordagens_zoom):
        valores = []

        for percentagem_treino in ordem_percentagens:
            valores_metrica = resumo_para_grafico[(resumo_para_grafico["percentagem_str"] == percentagem_treino) & (resumo_para_grafico["abordagem_treino"] == abordagem)][metrica]
            valores.append(valores_metrica.iloc[0] if not valores_metrica.empty else float("nan"))

        deslocamento = (indice - 0.5) * largura_barras_zoom

        eixo_zoom.bar([posicao + deslocamento for posicao in posicoes], valores, width=largura_barras_zoom, color=CORES[abordagem], label=rotulos_grafico[abordagem], alpha=0.90)

    eixo_zoom.axhline(y=valor_referencia, color="gray", linestyle="--", linewidth=1.5, label="Prophet com 100% real")

    resumo_zoom = resumo_para_grafico[resumo_para_grafico["abordagem_treino"].isin(abordagens_zoom)]

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


nrmse_referencia_grafico = valor_ref_100(resumo, "NRMSE_medio")
desenhar_figura(metrica="NRMSE_medio", ylabel="NRMSE médio (%)", valor_referencia=nrmse_referencia_grafico, titulo="A. Erro de previsão do Prophet por cenário de treino (NRMSE)", nome_ficheiro="figura_06_1_nrmse_prophet_por_cenario_divisao_atual.png")

rmse_ref = valor_ref_100(resumo, "RMSE_medio")
desenhar_figura(metrica="RMSE_medio", ylabel="RMSE médio (kWh)", valor_referencia=rmse_ref, titulo="B. Erro de previsão do Prophet por cenário de treino (RMSE)", nome_ficheiro="figura_06_2_rmse_prophet_por_cenario_divisao_atual.png")

print("Gráficos provisórios guardados:", pasta_figuras)

if executar_sensibilidade and registos_sensibilidade:
    resultados_sensibilidade = pd.DataFrame(registos_sensibilidade, columns=colunas_resultado)
    guardar_csv(resultados_sensibilidade, os.path.join(pasta_resultados, "06_2_sensibilidade_prophet_cps_0_5_por_edificio.csv"), index=False, sep=";", decimal=",")

    resumo_sens = (
        resultados_sensibilidade
        .groupby(["percentagem_treino_real", "abordagem_treino"])
        .agg(MAE_medio=("MAE", "mean"),
             RMSE_medio=("RMSE", "mean"),
             NRMSE_medio=("NRMSE", "mean"),
             NRMSE_mediano=("NRMSE", "median"),
             MAPE_medio=("MAPE", "mean"))
        .reset_index()
        .sort_values(["percentagem_treino_real", "abordagem_treino"])
        .reset_index(drop=True)
    )

    referencia_sensibilidade = valor_ref_100(resumo_sens, "NRMSE_medio")
    resumo_sens["melhoria_nrmse_vs_100_real_pct"] = (referencia_sensibilidade - resumo_sens["NRMSE_medio"]) / referencia_sensibilidade * 100

    guardar_csv(resumo_sens, os.path.join(pasta_resultados, "06_2_1_resumo_sensibilidade_prophet_cps_0_5.csv"), sep=";", decimal=",", index=False)
    print("Sensibilidade changepoint_prior_scale=0.5 guardada")
