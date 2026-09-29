import os
import numpy as np
import pandas as pd


semente_gerador = int(os.environ.get("GENERATOR_SEED", os.environ.get("EXPERIMENT_SEED", 42)))
np.random.seed(semente_gerador)

pasta_intermedia = "ficheiros_intermedios"
os.makedirs(pasta_intermedia, exist_ok=True)

caminho_base_modelacao = os.path.join(pasta_intermedia, "02_base_modelacao_consumo_e_temperatura.csv")
caminho_calibracao = os.path.join(pasta_intermedia, "01_1_edificios_calibracao.csv")
caminho_teste = os.path.join(pasta_intermedia, "01_2_edificios_teste.csv")
caminho_series_sinteticas = os.path.join(pasta_intermedia, "04_series_sinteticas_consumo_normalizado_por_edificio.csv")
caminho_parametros = os.path.join(pasta_intermedia, "04_1_parametros_gerador_sintetico.csv")

print("Gerar dados sintéticos")
print("Modelo: perfil hora/semana + temperatura + AR(1)")

base_modelacao = pd.read_csv(caminho_base_modelacao, sep=";", decimal=",", parse_dates=["timestamp"])
edificios_calibracao = pd.read_csv(caminho_calibracao, sep=";")["id_edificio"].tolist()
edificios_teste = pd.read_csv(caminho_teste, sep=";")["id_edificio"].tolist()

dados_calibracao = base_modelacao[base_modelacao["id_edificio"].isin(edificios_calibracao)].copy()
dados_teste = base_modelacao[base_modelacao["id_edificio"].isin(edificios_teste)].copy()

dados_calibracao = dados_calibracao.sort_values(by=["id_edificio", "timestamp"]).reset_index(drop=True)
dados_teste = dados_teste.sort_values(by=["id_edificio", "timestamp"]).reset_index(drop=True)


# 70% treino, 30% teste, igual aos restantes scripts
def definir_periodo(serie_edificio):
    n_observacoes_treino = int(len(serie_edificio) * 0.7)
    periodos = ["treino"] * n_observacoes_treino + ["teste"] * (len(serie_edificio) - n_observacoes_treino)
    return pd.Series(periodos, index=serie_edificio.index)


dados_calibracao["periodo"] = dados_calibracao.groupby("id_edificio", group_keys=False).apply(definir_periodo)
dados_teste["periodo"] = dados_teste.groupby("id_edificio", group_keys=False).apply(definir_periodo)

treino_calibracao = dados_calibracao[dados_calibracao["periodo"] == "treino"].copy()
treino_calibracao["hora_do_dia"] = treino_calibracao["timestamp"].dt.hour
treino_calibracao["dia_da_semana"] = treino_calibracao["timestamp"].dt.dayofweek

media_consumo = treino_calibracao.groupby("id_edificio")["consumo_kwh"].transform("mean")
treino_calibracao["consumo_real_normalizado"] = treino_calibracao["consumo_kwh"] / media_consumo

# perfil médio por hora e dia, suavizado para evitar variações bruscas
perfil_hora_semana = treino_calibracao.groupby(["hora_do_dia", "dia_da_semana"])["consumo_real_normalizado"].mean().unstack()
perfil_hora_semana = perfil_hora_semana.rolling(3, min_periods=1, center=True).mean()
mapa_perfil_horario_semanal = perfil_hora_semana.stack().to_dict()

pares_hora_dia_semana = pd.MultiIndex.from_arrays([treino_calibracao["hora_do_dia"], treino_calibracao["dia_da_semana"]])
componente_temporal = pares_hora_dia_semana.map(mapa_perfil_horario_semanal).fillna(1.0).astype(float)
componente_temporal = np.maximum(componente_temporal, 0.1)

residuo_apos_temporal = treino_calibracao["consumo_real_normalizado"] / componente_temporal
coeficiente_temperatura, intercepto_temperatura = np.polyfit(treino_calibracao["temperatura_ar_exterior"], residuo_apos_temporal, 1)

# limite no coeficiente térmico para evitar ajustes exagerados
coeficiente_temperatura = np.clip(coeficiente_temperatura, -0.05, 0.05)

componente_temperatura_base = np.maximum((treino_calibracao["temperatura_ar_exterior"] * coeficiente_temperatura) + intercepto_temperatura, 0.1)
consumo_estimado_sem_residuo = componente_temporal * componente_temperatura_base
residuo_multiplicativo = (treino_calibracao["consumo_real_normalizado"] / consumo_estimado_sem_residuo) - 1
desvio_padrao_residuo = residuo_multiplicativo.std()

percentil_01_consumo_normalizado = treino_calibracao["consumo_real_normalizado"].quantile(0.01)
percentil_999_consumo_normalizado = treino_calibracao["consumo_real_normalizado"].quantile(0.999)


# usa-se um coeficiente AR(1) fixo para introduzir persistência temporal sem ajustar demasiado o gerador aos edifícios de calibração
# AR1_PHI permite a análise de sensibilidade; o valor por defeito é o das execuções finais
coeficiente_autocorrelacao_ar1 = float(os.environ.get("AR1_PHI", 0.85))
registos = []

for id_edificio in edificios_teste:
    serie_edificio = dados_teste[dados_teste["id_edificio"] == id_edificio].copy()
    serie_edificio["hora_do_dia"] = serie_edificio["timestamp"].dt.hour
    serie_edificio["dia_da_semana"] = serie_edificio["timestamp"].dt.dayofweek

    pares_hora_dia_semana_edificio = pd.MultiIndex.from_arrays([serie_edificio["hora_do_dia"], serie_edificio["dia_da_semana"]])
    componente_temporal_edificio = pares_hora_dia_semana_edificio.map(mapa_perfil_horario_semanal).fillna(1.0).astype(float)

    componente_temperatura_edificio = np.maximum(serie_edificio["temperatura_ar_exterior"] * coeficiente_temperatura + intercepto_temperatura, 0.1)

    consumo_sintetico_sem_residuo = componente_temporal_edificio * componente_temperatura_edificio

    n_observacoes = len(serie_edificio)
    ruido_inovacao_ar1 = np.random.normal(0, desvio_padrao_residuo * np.sqrt(1 - coeficiente_autocorrelacao_ar1**2), n_observacoes)
    residuo_ar1 = np.zeros(n_observacoes)
    residuo_ar1[0] = ruido_inovacao_ar1[0]
    for posicao_temporal in range(1, n_observacoes):
        residuo_ar1[posicao_temporal] = coeficiente_autocorrelacao_ar1 * residuo_ar1[posicao_temporal - 1] + ruido_inovacao_ar1[posicao_temporal]

    componente_residual_ar1 = np.maximum(1 + residuo_ar1, 0.1)

    serie_edificio["consumo_sintetico_normalizado"] = consumo_sintetico_sem_residuo * componente_residual_ar1

    registos.append(serie_edificio)

if not registos:
    raise RuntimeError("Não foram geradas séries sintéticas.")

series_sinteticas = pd.concat(registos, ignore_index=True)

parametros = pd.DataFrame([{
    "coeficiente_efeito_temperatura": round(coeficiente_temperatura, 5),
    "intercepto_temperatura": round(intercepto_temperatura, 5),
    "desvio_residuo_multiplicativo_calibracao": round(desvio_padrao_residuo, 4),
    "consumo_norm_minimo_calibracao": round(percentil_01_consumo_normalizado, 4),
    "consumo_norm_maximo_calibracao": round(percentil_999_consumo_normalizado, 4),
    "coeficiente_autocorrelacao_ar1": coeficiente_autocorrelacao_ar1,
}])

parametros.to_csv(caminho_parametros, index=False, sep=";", decimal=",")

series_sinteticas[["id_edificio", "timestamp", "temperatura_ar_exterior", "consumo_sintetico_normalizado", "periodo"]].to_csv(caminho_series_sinteticas, index=False, sep=";", decimal=",")

print("Dados sintéticos guardados em:", caminho_series_sinteticas)
