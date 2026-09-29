import os
import pandas as pd


pasta_dados = "data"
pasta_intermedia = "ficheiros_intermedios"
os.makedirs(pasta_intermedia, exist_ok=True)

caminho_meteorologia = os.path.join(pasta_dados, "weather.csv")
caminho_consumo = os.path.join(pasta_intermedia, "01_consumo_horario_edificios_educacao_selecionados.csv")
caminho_calibracao = os.path.join(pasta_intermedia, "01_1_edificios_calibracao.csv")
caminho_teste = os.path.join(pasta_intermedia, "01_2_edificios_teste.csv")
caminho_saida = os.path.join(pasta_intermedia, "02_base_modelacao_consumo_e_temperatura.csv")
caminho_diagnostico_juncao = os.path.join(pasta_intermedia, "02_1_diagnostico_juncao_consumo_temperatura.csv")
caminho_diagnostico_edificios = os.path.join(pasta_intermedia, "02_2_diagnostico_qualidade_por_edificio.csv")

print("Preparar base de modelação")

edificios_calibracao = pd.read_csv(caminho_calibracao, sep=";")
edificios_teste = pd.read_csv(caminho_teste, sep=";")
edificios_estudo = pd.concat([edificios_calibracao, edificios_teste], ignore_index=True)
ids_edificios_estudo = edificios_estudo["id_edificio"].tolist()
print("Edifícios:", len(ids_edificios_estudo))

edificios_estudo[["site_bdg2", "tipo_edificio", "nome_original_edificio"]] = edificios_estudo["id_edificio"].str.split("_", n=2, expand=True)
sites_usados = edificios_estudo["site_bdg2"].unique().tolist()

consumo_largo = pd.read_csv(caminho_consumo, sep=";", decimal=",", parse_dates=["timestamp"])
consumo_largo = consumo_largo[["timestamp"] + ids_edificios_estudo]
print("Dimensão consumo:", consumo_largo.shape)

# passar para formato longo porque a meteorologia está indexada por site
consumo_longo = consumo_largo.melt(id_vars="timestamp", var_name="id_edificio", value_name="consumo_kwh")
consumo_longo = pd.merge(
    consumo_longo,
    edificios_estudo[["id_edificio", "site_bdg2", "tipo_edificio", "nome_original_edificio"]],
    on="id_edificio",
    how="left",
)
print("Dimensão após formato longo:", consumo_longo.shape)

meteorologia = pd.read_csv(caminho_meteorologia, parse_dates=["timestamp"])
meteorologia = meteorologia[meteorologia["site_id"].isin(sites_usados)].copy()
meteorologia = meteorologia[["timestamp", "site_id", "airTemperature"]]
meteorologia.rename(columns={"site_id": "site_bdg2", "airTemperature": "temperatura_ar_exterior"}, inplace=True)
print("Dimensão meteorologia:", meteorologia.shape)

base_modelacao = pd.merge(consumo_longo, meteorologia, on=["timestamp", "site_bdg2"], how="left")

cobertura_meteorologica = 1 - base_modelacao["temperatura_ar_exterior"].isna().mean()
print("Cobertura meteorológica:", round(cobertura_meteorologica * 100, 2), "%")
linhas_antes_filtragem = len(base_modelacao)

diagnostico_edificios = (
    base_modelacao
    .groupby("id_edificio")
    .agg(n_linhas=("timestamp", "count"),
         faltas_consumo=("consumo_kwh", lambda serie_coluna: serie_coluna.isna().sum()),
         faltas_temperatura=("temperatura_ar_exterior", lambda serie_coluna: serie_coluna.isna().sum()),
         inicio=("timestamp", "min"),
         fim=("timestamp", "max"))
    .reset_index()
)

diagnostico_edificios["percentagem_faltas_consumo"] = diagnostico_edificios["faltas_consumo"] / diagnostico_edificios["n_linhas"] * 100
diagnostico_edificios["percentagem_faltas_temperatura"] = diagnostico_edificios["faltas_temperatura"] / diagnostico_edificios["n_linhas"] * 100
diagnostico_edificios.to_csv(caminho_diagnostico_edificios, index=False, sep=";", decimal=",")

base_modelacao.dropna(subset=["consumo_kwh", "temperatura_ar_exterior"], inplace=True)

linhas_apos_filtragem = len(base_modelacao)
linhas_removidas_por_faltas = linhas_antes_filtragem - linhas_apos_filtragem
percentagem_linhas_removidas = (linhas_removidas_por_faltas / linhas_antes_filtragem) * 100 if linhas_antes_filtragem > 0 else 0.0

print("Linhas removidas:", linhas_removidas_por_faltas, "(", round(percentagem_linhas_removidas, 2), "%)")

diagnostico_juncao = pd.DataFrame({
    "metrica": ["cobertura_meteorologica_percentagem", "linhas_removidas_percentagem"],
    "valor": [cobertura_meteorologica * 100, percentagem_linhas_removidas],
})
diagnostico_juncao.to_csv(caminho_diagnostico_juncao, index=False, sep=";", decimal=",")
print("Diagnóstico guardado em:", caminho_diagnostico_juncao)

base_modelacao.sort_values(["id_edificio", "timestamp"], inplace=True)
base_modelacao.reset_index(drop=True, inplace=True)
print("Dimensão final:", base_modelacao.shape)

base_modelacao.to_csv(caminho_saida, index=False, sep=";", decimal=",")
print("Dataset guardado em:", caminho_saida)
