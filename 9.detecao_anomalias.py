import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


semente_anomalias = int(os.environ.get("ANOMALY_SEED", os.environ.get("EXPERIMENT_SEED", 42)))
np.random.seed(semente_anomalias)

pasta_intermedia = "ficheiros_intermedios"
pasta_resultados = "resultados"
os.makedirs(pasta_resultados, exist_ok=True)

caminho_base_modelacao = os.path.join(pasta_intermedia, "02_base_modelacao_consumo_e_temperatura.csv")
caminho_series_sinteticas = os.path.join(pasta_intermedia, "04_series_sinteticas_consumo_normalizado_por_edificio.csv")
caminho_resultados = os.path.join(pasta_resultados, "09_isolation_forest_anomalias_por_edificio_divisao_atual.csv")
caminho_resumo = os.path.join(pasta_resultados, "09_1_resumo_isolation_forest_anomalias_por_cenario_divisao_atual.csv")

percentagens_treino = [1.0, 0.05, 0.10, 0.20]
proporcao_anomalias = 0.01
lag_maximo = 24

print("Deteção de anomalias")

base_modelacao = pd.read_csv(caminho_base_modelacao, sep=";", decimal=",", parse_dates=["timestamp"])
base_modelacao["timestamp"] = pd.to_datetime(base_modelacao["timestamp"])

series_sinteticas = pd.read_csv(caminho_series_sinteticas, sep=";", decimal=",", parse_dates=["timestamp"])
series_sinteticas["timestamp"] = pd.to_datetime(series_sinteticas["timestamp"])

edificios = series_sinteticas["id_edificio"].unique()

registos = []


def injetar_anomalias(serie_consumo, proporcao=0.01, semente_anomalias=42, desvio_padrao_treino_referencia=None):
    # injeta picos positivos e negativos em cerca de 1% dos pontos para criar a referência
    serie_consumo = serie_consumo.astype(float).copy()
    n_observacoes = len(serie_consumo)
    n_anomalias = max(1, int(n_observacoes * proporcao))

    gerador_aleatorio = np.random.default_rng(semente_anomalias)
    indices_anomalia = gerador_aleatorio.choice(n_observacoes, n_anomalias, replace=False)

    desvio = desvio_padrao_treino_referencia if desvio_padrao_treino_referencia is not None else serie_consumo.std()
    if desvio == 0 or np.isnan(desvio):
        media = serie_consumo.mean()
        desvio = media * 0.1 if media > 0 and not np.isnan(media) else 1.0

    for indice_anomalia in indices_anomalia:
        fator_anomalia = 3 if gerador_aleatorio.random() > 0.5 else -3
        valor_anomalo = serie_consumo.iloc[indice_anomalia] + fator_anomalia * desvio
        serie_consumo.iloc[indice_anomalia] = max(0, valor_anomalo)

    rotulos_anomalia = np.zeros(n_observacoes, dtype=int)
    rotulos_anomalia[indices_anomalia] = 1
    return serie_consumo, rotulos_anomalia


def construir_variaveis_anomalias(consumo, timestamps, rotulos_anomalia=None, lag_maximo=24):
    dados = {
        "timestamp": pd.to_datetime(timestamps),
        "consumo": np.asarray(consumo, dtype=float),
    }

    if rotulos_anomalia is not None:
        rotulos_array = np.asarray(rotulos_anomalia)

        if len(rotulos_array) != len(dados["consumo"]):
            raise ValueError("Número de rótulos não bate com o número de observações.")

        dados["label"] = rotulos_array

    dados_variaveis = pd.DataFrame(dados).sort_values("timestamp").reset_index(drop=True)

    dados_variaveis["hora"] = dados_variaveis["timestamp"].dt.hour / 23.0
    dados_variaveis["dia_semana"] = dados_variaveis["timestamp"].dt.dayofweek / 6.0
    dados_variaveis["lag_1"] = dados_variaveis["consumo"].shift(1)
    dados_variaveis["lag_24"] = dados_variaveis["consumo"].shift(lag_maximo)

    dados_variaveis = dados_variaveis.dropna().reset_index(drop=True)
    return dados_variaveis


def treinar_avaliar_if(consumo_treino, timestamps_treino, consumo_teste, timestamps_teste, anomalias_reais, *, contaminacao=None, lag_maximo=24, proporcao=0.01, estado_aleatorio=42):
    dados_treino_variaveis = construir_variaveis_anomalias(consumo_treino, timestamps_treino, lag_maximo=lag_maximo)
    dados_teste_variaveis = construir_variaveis_anomalias(consumo_teste, timestamps_teste, rotulos_anomalia=anomalias_reais, lag_maximo=lag_maximo)

    if dados_treino_variaveis.empty or dados_teste_variaveis.empty:
        return None

    # normalização ajustada apenas no treino para evitar usar dados do teste
    normalizador = StandardScaler()

    variaveis_treino_escaladas = normalizador.fit_transform(dados_treino_variaveis[["consumo", "lag_1", "lag_24"]])
    variaveis_teste_escaladas = normalizador.transform(dados_teste_variaveis[["consumo", "lag_1", "lag_24"]])

    matriz_variaveis_treino = np.column_stack([variaveis_treino_escaladas[:, 0], dados_treino_variaveis["hora"].values, dados_treino_variaveis["dia_semana"].values, variaveis_treino_escaladas[:, 1], variaveis_treino_escaladas[:, 2]])

    matriz_variaveis_teste = np.column_stack([variaveis_teste_escaladas[:, 0], dados_teste_variaveis["hora"].values, dados_teste_variaveis["dia_semana"].values, variaveis_teste_escaladas[:, 1], variaveis_teste_escaladas[:, 2]])

    rotulos_teste = dados_teste_variaveis["label"].values

    contaminacao_modelo = contaminacao if contaminacao is not None else proporcao

    modelo_isolation_forest = IsolationForest(contamination=contaminacao_modelo, random_state=estado_aleatorio)
    modelo_isolation_forest.fit(matriz_variaveis_treino)

    anomalias_previstas = (modelo_isolation_forest.predict(matriz_variaveis_teste) == -1).astype(int)

    return (
        precision_score(rotulos_teste, anomalias_previstas, zero_division=0),
        recall_score(rotulos_teste, anomalias_previstas, zero_division=0),
        f1_score(rotulos_teste, anomalias_previstas, zero_division=0),
    )


# cada edifício é avaliado separadamente para não misturar padrões de consumo distintos
for indice_edificio, id_edificio in enumerate(edificios):
    print("Edifício:", id_edificio)

    dados_reais_edificio = base_modelacao[base_modelacao["id_edificio"] == id_edificio].sort_values("timestamp").reset_index(drop=True)
    dados_sinteticos_edificio = series_sinteticas[series_sinteticas["id_edificio"] == id_edificio].sort_values("timestamp").reset_index(drop=True)

    if dados_reais_edificio.empty or dados_sinteticos_edificio.empty:
        print(" Sem dados para", id_edificio, "- a ignorar")
        continue

    corte = int(len(dados_reais_edificio) * 0.7)

    treino_real_completo = dados_reais_edificio.iloc[:corte].copy().reset_index(drop=True)
    teste_real = dados_reais_edificio.iloc[corte:].copy().reset_index(drop=True)

    treino_sintetico = dados_sinteticos_edificio[dados_sinteticos_edificio["periodo"] == "treino"].copy().reset_index(drop=True)

    if treino_real_completo.empty or teste_real.empty:
        print(id_edificio, "sem treino ou teste suficiente")
        continue

    desvio_padrao_treino_referencia = treino_real_completo["consumo_kwh"].std()

    # semente específica das anomalias para não interferir com a do gerador
    semente_injecao_anomalias = semente_anomalias + 10000 + indice_edificio

    consumo_teste_com_anomalias, anomalias_reais = injetar_anomalias(teste_real["consumo_kwh"], proporcao=proporcao_anomalias, semente_anomalias=semente_injecao_anomalias, desvio_padrao_treino_referencia=desvio_padrao_treino_referencia)

    timestamps_teste = teste_real["timestamp"].values

    for percentagem_treino in percentagens_treino:
        if percentagem_treino == 1.0:
            n_treino_efetivo = len(treino_real_completo)
            nome_cenario_base = "100_real"
        else:
            n_treino_efetivo = max(int(len(treino_real_completo) * percentagem_treino), lag_maximo + 1)
            nome_cenario_base = str(int(percentagem_treino * 100)) + "_real"

        treino_real_reduzido = treino_real_completo.iloc[:n_treino_efetivo].copy()

        consumo_treino = treino_real_reduzido["consumo_kwh"].values.astype(float)
        timestamps_treino = treino_real_reduzido["timestamp"].values

        # cenário de referência do detetor sem reforço sintético
        metricas_contaminacao_base = treinar_avaliar_if(consumo_treino, timestamps_treino, consumo_teste_com_anomalias.values, timestamps_teste, anomalias_reais, lag_maximo=lag_maximo, proporcao=proporcao_anomalias, estado_aleatorio=semente_anomalias)
        metricas_contaminacao_baixa = treinar_avaliar_if(consumo_treino, timestamps_treino, consumo_teste_com_anomalias.values, timestamps_teste, anomalias_reais, contaminacao=proporcao_anomalias * 0.8, lag_maximo=lag_maximo, proporcao=proporcao_anomalias, estado_aleatorio=semente_anomalias)
        metricas_contaminacao_alta = treinar_avaliar_if(consumo_treino, timestamps_treino, consumo_teste_com_anomalias.values, timestamps_teste, anomalias_reais, contaminacao=proporcao_anomalias * 1.2, lag_maximo=lag_maximo, proporcao=proporcao_anomalias, estado_aleatorio=semente_anomalias)

        if metricas_contaminacao_base:
            f1_baixa = metricas_contaminacao_baixa[2] if metricas_contaminacao_baixa else np.nan
            f1_alta = metricas_contaminacao_alta[2] if metricas_contaminacao_alta else np.nan
            registos.append([id_edificio, nome_cenario_base, *metricas_contaminacao_base, f1_baixa, f1_alta])

        # testa se o complemento sintético altera o comportamento aprendido como normal pelo Isolation Forest
        if percentagem_treino < 1.0:
            if len(treino_real_reduzido) >= 24:
                escala = float(treino_real_reduzido.assign(_h=treino_real_reduzido["timestamp"].dt.hour).groupby("_h")["consumo_kwh"].median().median())
            else:
                escala = float(treino_real_reduzido["consumo_kwh"].median())

            if np.isnan(escala) or escala <= 0:
                media = treino_real_reduzido["consumo_kwh"].mean()
                escala = float(media) if media > 0 else 1.0

            dados_sinteticos_para_complemento = treino_sintetico[~treino_sintetico["timestamp"].isin(treino_real_reduzido["timestamp"])].copy()

            if dados_sinteticos_para_complemento.empty:
                print("  [Aviso] sem complemento sintético para", id_edificio, "no cenário", nome_cenario_base)
                continue

            consumo_sintetico_kwh = dados_sinteticos_para_complemento["consumo_sintetico_normalizado"].values.astype(float) * escala

            consumo_treino_aumentado = np.concatenate([consumo_treino, consumo_sintetico_kwh])
            timestamps_treino_aumentado = np.concatenate([treino_real_reduzido["timestamp"].values, dados_sinteticos_para_complemento["timestamp"].values])

            metricas_contaminacao_base = treinar_avaliar_if(consumo_treino_aumentado, timestamps_treino_aumentado, consumo_teste_com_anomalias.values, timestamps_teste, anomalias_reais, lag_maximo=lag_maximo, proporcao=proporcao_anomalias, estado_aleatorio=semente_anomalias)
            metricas_contaminacao_baixa = treinar_avaliar_if(consumo_treino_aumentado, timestamps_treino_aumentado, consumo_teste_com_anomalias.values, timestamps_teste, anomalias_reais, contaminacao=proporcao_anomalias * 0.8, lag_maximo=lag_maximo, proporcao=proporcao_anomalias, estado_aleatorio=semente_anomalias)
            metricas_contaminacao_alta = treinar_avaliar_if(consumo_treino_aumentado, timestamps_treino_aumentado, consumo_teste_com_anomalias.values, timestamps_teste, anomalias_reais, contaminacao=proporcao_anomalias * 1.2, lag_maximo=lag_maximo, proporcao=proporcao_anomalias, estado_aleatorio=semente_anomalias)

            if metricas_contaminacao_base:
                f1_baixa = metricas_contaminacao_baixa[2] if metricas_contaminacao_baixa else np.nan
                f1_alta = metricas_contaminacao_alta[2] if metricas_contaminacao_alta else np.nan
                registos.append([id_edificio, str(int(percentagem_treino * 100)) + "_real_sintetico", *metricas_contaminacao_base, f1_baixa, f1_alta])

if not registos:
    raise RuntimeError("Não foram gerados resultados de deteção de anomalias.")

resultados_por_edificio = pd.DataFrame(registos, columns=["id_edificio", "cenario_treino", "precisao", "sensibilidade", "f1", "f1_contaminacao_baixa", "f1_contaminacao_alta"])
resultados_por_edificio.to_csv(caminho_resultados, index=False, sep=";", decimal=",")

resumo = resultados_por_edificio.groupby("cenario_treino")[["precisao", "sensibilidade", "f1", "f1_contaminacao_baixa", "f1_contaminacao_alta"]].mean().reset_index()

resumo["amplitude_f1_contaminacao"] = resumo[["f1_contaminacao_baixa", "f1_contaminacao_alta"]].sub(resumo["f1"], axis=0).abs().max(axis=1)
resumo.to_csv(caminho_resumo, sep=";", decimal=",", index=False)

print("\n" + "-" * 60)
print(resumo.to_string(index=False))
print("-" * 60)
print("Resultados guardados em:", pasta_resultados)
