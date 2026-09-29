import os
import shutil
import subprocess
import sys
import time
import pandas as pd


SEMENTES_DIVISAO = [42, 100, 200, 400, 500]

pasta_resultados = "resultados"
pasta_scripts = "scripts"
pasta_intermedia = "ficheiros_intermedios"

os.makedirs(pasta_resultados, exist_ok=True)


def extrair_sementes(tabela):
    if "semente_divisao" not in tabela.columns:
        return []

    sementes = pd.to_numeric(tabela["semente_divisao"], errors="coerce").dropna().astype(int).unique()
    return sorted(sementes.tolist())


def validar_sementes(pasta_resultados, sementes_esperadas):
    sementes_esperadas_ordenadas = sorted(int(semente) for semente in sementes_esperadas)
    ficheiros_consolidados = {
        "previsao": "10_1_resumo_previsao_prophet_por_divisao.csv",
        "anomalias": "10_2_resumo_isolation_forest_anomalias_por_divisao.csv",
        "previsao_por_edificio": "10_5_previsao_prophet_por_edificio_todas_divisoes.csv",
    }

    inconsistencias = []
    for rotulo_resultado, nome_ficheiro in ficheiros_consolidados.items():
        caminho = os.path.join(pasta_resultados, nome_ficheiro)
        tabela_resultados_consolidados = pd.read_csv(caminho, sep=";", decimal=",")
        sementes_obtidas = extrair_sementes(tabela_resultados_consolidados)

        if sementes_obtidas != sementes_esperadas_ordenadas:
            inconsistencias.append(nome_ficheiro + ": esperado " + str(sementes_esperadas_ordenadas) + ", obtido " + str(sementes_obtidas))

    if inconsistencias:
        raise RuntimeError("Sementes inconsistentes:\n" + "\n".join(inconsistencias))


def executar(comando, maximo_tentativas=3, espera=5):
    # tenta de novo em caso de falha transitória do subprocesso
    for tentativa in range(maximo_tentativas):
        try:
            subprocess.run(comando, check=True)
            return
        except subprocess.CalledProcessError as erro:
            if tentativa < maximo_tentativas - 1:
                nome_script = os.path.basename(erro.cmd[-1])
                print("  Tentativa", tentativa + 1, "/", maximo_tentativas, ":", nome_script, "falhou - a esperar", espera, "s")
                time.sleep(espera)
            else:
                raise


def copiar_se_existir(origem, destino):
    if os.path.exists(origem):
        shutil.copy(origem, destino)


resumos_previsao = []
resumos_anomalias = []
previsoes_por_edificio_todas_divisoes = []

print("Splits a executar:", SEMENTES_DIVISAO)
print("  Pipeline: 1 > 2 > 4 > 6 > 9\n")

hora_inicio_execucao = time.time()

for indice_divisao, semente_divisao in enumerate(SEMENTES_DIVISAO):
    os.environ["EXPERIMENT_SEED"] = str(semente_divisao)
    os.environ["SPLIT_SEED"] = str(semente_divisao)
    os.environ["GENERATOR_SEED"] = str(10000 + semente_divisao)
    os.environ["FORECAST_SEED"] = str(20000 + semente_divisao)
    os.environ["ANOMALY_SEED"] = str(30000 + semente_divisao)

    print("=" * 60)
    print("Split", indice_divisao + 1, "/", len(SEMENTES_DIVISAO), "[semente=", semente_divisao, "]")
    print("=" * 60)

    try:
        executar([sys.executable, os.path.join(pasta_scripts, "1.selecao_edificios.py")])
        executar([sys.executable, os.path.join(pasta_scripts, "2.data_modelacao.py")])
        executar([sys.executable, os.path.join(pasta_scripts, "4.gerar_dados.py")])
        executar([sys.executable, os.path.join(pasta_scripts, "6.experiencias_previsao.py")])
        executar([sys.executable, os.path.join(pasta_scripts, "9.detecao_anomalias.py")])

        # arquiva as listas de edifícios e os parâmetros do gerador de cada split para auditoria
        ficheiros_intermedios_a_arquivar = ["01_1_edificios_calibracao.csv", "01_2_edificios_teste.csv", "04_1_parametros_gerador_sintetico.csv"]
        for nome_ficheiro in ficheiros_intermedios_a_arquivar:
            origem = os.path.join(pasta_intermedia, nome_ficheiro)
            destino = os.path.join(pasta_resultados, nome_ficheiro.replace(".csv", "_divisao" + str(semente_divisao) + ".csv"))
            copiar_se_existir(origem, destino)

        shutil.copy(os.path.join(pasta_resultados, "06_previsao_prophet_por_edificio_divisao_atual.csv"),
                    os.path.join(pasta_resultados, "06_previsao_prophet_por_edificio_divisao" + str(semente_divisao) + ".csv"))
        shutil.copy(os.path.join(pasta_resultados, "09_isolation_forest_anomalias_por_edificio_divisao_atual.csv"),
                    os.path.join(pasta_resultados, "09_isolation_forest_anomalias_por_edificio_divisao" + str(semente_divisao) + ".csv"))

        copiar_se_existir(os.path.join(pasta_resultados, "06_1_resumo_previsao_prophet_por_cenario_divisao_atual.csv"),
                          os.path.join(pasta_resultados, "06_1_resumo_previsao_prophet_por_cenario_divisao" + str(semente_divisao) + ".csv"))

        copiar_se_existir(os.path.join(pasta_resultados, "09_1_resumo_isolation_forest_anomalias_por_cenario_divisao_atual.csv"),
                          os.path.join(pasta_resultados, "09_1_resumo_isolation_forest_anomalias_por_cenario_divisao" + str(semente_divisao) + ".csv"))

        resumo_previsao_divisao = pd.read_csv(os.path.join(pasta_resultados, "06_1_resumo_previsao_prophet_por_cenario_divisao_atual.csv"), sep=";", decimal=",")
        resumo_previsao_divisao["semente_divisao"] = semente_divisao
        resumos_previsao.append(resumo_previsao_divisao)

        resumo_anomalias_divisao = pd.read_csv(os.path.join(pasta_resultados, "09_1_resumo_isolation_forest_anomalias_por_cenario_divisao_atual.csv"), sep=";", decimal=",")
        resumo_anomalias_divisao["semente_divisao"] = semente_divisao
        resumos_anomalias.append(resumo_anomalias_divisao)

        previsao_edificio_divisao = pd.read_csv(os.path.join(pasta_resultados, "06_previsao_prophet_por_edificio_divisao_atual.csv"), sep=";", decimal=",")
        previsao_edificio_divisao["semente_divisao"] = semente_divisao
        previsoes_por_edificio_todas_divisoes.append(previsao_edificio_divisao)

        print("  Split", semente_divisao, "concluída.\n")

    except Exception as erro:
        raise RuntimeError("O split " + str(semente_divisao) + " falhou.") from erro


print("\nConsolidar resultados de todos os splits...")

if not resumos_previsao:
    print("Nenhum split completou com sucesso.")
    sys.exit(1)

n_divisoes_validas = len(resumos_previsao)
print("  Splits concluídos:", n_divisoes_validas, "/", len(SEMENTES_DIVISAO))

resumo_previsao_todas_divisoes = pd.concat(resumos_previsao, ignore_index=True)
resumo_anomalias_todas_divisoes = pd.concat(resumos_anomalias, ignore_index=True)
previsao_edificio_todas_divisoes = pd.concat(previsoes_por_edificio_todas_divisoes, ignore_index=True)

resumo_previsao_todas_divisoes.to_csv(os.path.join(pasta_resultados, "10_1_resumo_previsao_prophet_por_divisao.csv"), index=False, sep=";", decimal=",")
resumo_anomalias_todas_divisoes.to_csv(os.path.join(pasta_resultados, "10_2_resumo_isolation_forest_anomalias_por_divisao.csv"), index=False, sep=";", decimal=",")
previsao_edificio_todas_divisoes.to_csv(os.path.join(pasta_resultados, "10_5_previsao_prophet_por_edificio_todas_divisoes.csv"), index=False, sep=";", decimal=",")

validar_sementes(pasta_resultados, SEMENTES_DIVISAO)
print("  Sementes validadas em 10_1, 10_2 e 10_5.")

if {"percentagem_treino_real", "abordagem_treino"}.issubset(resumo_previsao_todas_divisoes.columns):
    colunas_agregacao = resumo_previsao_todas_divisoes.columns.tolist()
    agregado_previsao = {"RMSE_medio": ["mean", "std"]}

    if "NRMSE_medio" in colunas_agregacao:
        agregado_previsao["NRMSE_medio"] = ["mean", "std"]
    if "NRMSE_mediano" in colunas_agregacao:
        agregado_previsao["NRMSE_mediano"] = ["mean", "std"]

    estabilidade_previsao = resumo_previsao_todas_divisoes.groupby(["percentagem_treino_real", "abordagem_treino"]).agg(agregado_previsao).reset_index()
    estabilidade_previsao.columns = ["_".join(coluna_metrica).strip("_") if isinstance(coluna_metrica, tuple) else coluna_metrica for coluna_metrica in estabilidade_previsao.columns]

    mapa_metricas_previsao = {
        "RMSE_medio_mean": "RMSE_media_global",
        "RMSE_medio_std": "RMSE_desvio_global",
        "NRMSE_medio_mean": "NRMSE_media_global",
        "NRMSE_medio_std": "NRMSE_desvio_global",
        "NRMSE_mediano_mean": "NRMSE_mediana_global",
        "NRMSE_mediano_std": "NRMSE_mediana_desvio_global",
    }
    estabilidade_previsao = estabilidade_previsao.rename(columns=mapa_metricas_previsao)
    estabilidade_previsao.to_csv(os.path.join(pasta_resultados, "10_3_estabilidade_previsao_prophet_entre_divisoes.csv"), index=False, sep=";", decimal=",")

if "cenario_treino" in resumo_anomalias_todas_divisoes.columns:
    colunas_agregacao_anomalias = ["precisao", "sensibilidade", "f1"]
    for coluna_metrica in ["f1_contaminacao_baixa", "f1_contaminacao_alta", "amplitude_f1_contaminacao"]:
        if coluna_metrica in resumo_anomalias_todas_divisoes.columns:
            colunas_agregacao_anomalias.append(coluna_metrica)

    estabilidade_anomalias = resumo_anomalias_todas_divisoes.groupby("cenario_treino")[colunas_agregacao_anomalias].agg(["mean", "std"]).reset_index()
    estabilidade_anomalias.columns = ["_".join(coluna_metrica).strip("_") if isinstance(coluna_metrica, tuple) else coluna_metrica for coluna_metrica in estabilidade_anomalias.columns]

    mapa_metricas_anomalias = {
        "precisao_mean": "precisao_media_global",
        "precisao_std": "precisao_desvio_global",
        "sensibilidade_mean": "sensibilidade_media_global",
        "sensibilidade_std": "sensibilidade_desvio_global",
        "f1_mean": "f1_media_global",
        "f1_std": "f1_desvio_global",
        "f1_contaminacao_baixa_mean": "f1_contaminacao_baixa_media_global",
        "f1_contaminacao_baixa_std": "f1_contaminacao_baixa_desvio_global",
        "f1_contaminacao_alta_mean": "f1_contaminacao_alta_media_global",
        "f1_contaminacao_alta_std": "f1_contaminacao_alta_desvio_global",
        "amplitude_f1_contaminacao_mean": "amplitude_f1_contaminacao_media_global",
        "amplitude_f1_contaminacao_std": "amplitude_f1_contaminacao_desvio_global",
    }
    estabilidade_anomalias = estabilidade_anomalias.rename(columns=mapa_metricas_anomalias)
    estabilidade_anomalias.to_csv(os.path.join(pasta_resultados, "10_4_estabilidade_isolation_forest_anomalias_entre_divisoes.csv"), index=False, sep=";", decimal=",")


duracao_total_execucao = (time.time() - hora_inicio_execucao) / 60
print("\nConcluído em", round(duracao_total_execucao, 1), "minutos.")
print(" ", n_divisoes_validas, "/", len(SEMENTES_DIVISAO), "splits com sucesso.")
print("  Resultados em:", pasta_resultados)
