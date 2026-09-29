import os
import numpy as np
import pandas as pd


semente_divisao = int(os.environ.get("SPLIT_SEED", os.environ.get("EXPERIMENT_SEED", 42)))
np.random.seed(semente_divisao)

ficheiro_consumo = "data/electricity_cleaned.csv"
pasta_intermedia = "ficheiros_intermedios"
os.makedirs(pasta_intermedia, exist_ok=True)

limite_faltas_pre = 0.05
limite_faltas_pos = 0.02

total_edificios_calibracao = 7
total_edificios_teste = 5
total_edificios_estudo = total_edificios_calibracao + total_edificios_teste

print("Carregar o dataset de eletricidade")

consumo_original = pd.read_csv(ficheiro_consumo)
consumo_original["timestamp"] = pd.to_datetime(consumo_original["timestamp"])
consumo_original.set_index("timestamp", inplace=True)

print("Dimensão do dataset:", consumo_original.shape)

faltas_por_edificio = consumo_original.isna().mean()
edificios_com_faltas_aceitaveis = faltas_por_edificio[faltas_por_edificio < limite_faltas_pre].index.tolist()

print("Edifícios com menos de 5% de faltas:", len(edificios_com_faltas_aceitaveis))

# o id do BDG2 vem no formato site_tipo_nome
edificios_selecionaveis = pd.DataFrame({"id_edificio": edificios_com_faltas_aceitaveis})
edificios_selecionaveis[["site_bdg2", "tipo_edificio", "nome_original_edificio"]] = edificios_selecionaveis["id_edificio"].str.split("_", n=2, expand=True)

edificios_educacao = edificios_selecionaveis[edificios_selecionaveis["tipo_edificio"] == "education"].reset_index(drop=True)
print("Edifícios de educação:", len(edificios_educacao))

# privilegia-se a diversidade dos sites antes de completar a amostra
amostra = edificios_educacao.groupby("site_bdg2").sample(n=1, random_state=semente_divisao)

if len(amostra) < total_edificios_estudo:
    edificios_restantes = edificios_educacao[~edificios_educacao["id_edificio"].isin(amostra["id_edificio"])]
    numero_edificios_em_falta = total_edificios_estudo - len(amostra)

    if len(edificios_restantes) < numero_edificios_em_falta:
        raise ValueError("Edifícios de educação insuficientes.")

    amostra = pd.concat([amostra, edificios_restantes.sample(n=numero_edificios_em_falta, random_state=semente_divisao)], ignore_index=True)

amostra = amostra.sample(frac=1, random_state=semente_divisao).reset_index(drop=True)

print("Total selecionado:", len(amostra))
print("Número de sites:", amostra["site_bdg2"].nunique())

edificios_calibracao = amostra.head(total_edificios_calibracao)["id_edificio"].tolist()
edificios_teste = amostra.iloc[total_edificios_calibracao:total_edificios_calibracao + total_edificios_teste]["id_edificio"].tolist()

print("Calibração:", len(edificios_calibracao))
print("Teste:", len(edificios_teste))

consumo_edificios_selecionados = consumo_original[edificios_calibracao + edificios_teste].copy()
consumo_edificios_selecionados = consumo_edificios_selecionados.sort_index()

# interpolação só para lacunas curtas (até 6 horas)
consumo_edificios_selecionados = consumo_edificios_selecionados.interpolate(method="time", limit=6)

percentagem_faltas_apos_interpolacao = consumo_edificios_selecionados.isna().mean().mean()
print(f"Percentagem média de faltas após interpolação: {percentagem_faltas_apos_interpolacao:.4%}")

faltas_por_edificio_pos = consumo_edificios_selecionados.isna().mean()
edificios_validos_apos_interpolacao = consumo_edificios_selecionados.columns[faltas_por_edificio_pos < limite_faltas_pos]
consumo_edificios_selecionados = consumo_edificios_selecionados[edificios_validos_apos_interpolacao]

edificios_calibracao = [id_edificio for id_edificio in edificios_calibracao if id_edificio in consumo_edificios_selecionados.columns]
edificios_teste = [id_edificio for id_edificio in edificios_teste if id_edificio in consumo_edificios_selecionados.columns]

n_edificios_removidos = total_edificios_estudo - len(edificios_calibracao) - len(edificios_teste)

print(f"Edifícios removidos por qualidade insuficiente: {n_edificios_removidos}")
print(f"Resultado final da divisão (Calibração): {len(edificios_calibracao)} | Teste: {len(edificios_teste)}")

if len(edificios_calibracao) < total_edificios_calibracao or len(edificios_teste) < total_edificios_teste:
    raise ValueError("Número insuficiente de edifícios após filtragem.")

consumo_edificios_selecionados.to_csv(os.path.join(pasta_intermedia, "01_consumo_horario_edificios_educacao_selecionados.csv"), sep=";", decimal=",")

pd.DataFrame({"id_edificio": edificios_calibracao}).to_csv(os.path.join(pasta_intermedia, "01_1_edificios_calibracao.csv"), index=False, sep=";", decimal=",")

pd.DataFrame({"id_edificio": edificios_teste}).to_csv(os.path.join(pasta_intermedia, "01_2_edificios_teste.csv"), index=False, sep=";", decimal=",")

print("Ficheiros guardados:", pasta_intermedia)
