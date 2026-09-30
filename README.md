
This repository contains the experimental pipeline, code, and data for the MSc thesis *"Geração de Dados Sintéticos para Gestão Energética de Edifícios: Uma Abordagem Estatístico-Temporal para Data Augmentation"* by Tiago Esteves, presented at the Universidade Politécnica de Santarém.

The research evaluates the impact of complementing limited historical training data—specifically 5%, 10%, and 20% availability scenarios—with synthetic time series. The experiments utilize education buildings from the Building Data Genome Project 2 (BDG2) dataset and enforce a strict separation between the buildings used to calibrate the generator and those used for final evaluation.

<p align="center"><img src="Código/figuras/fluxograma_metodologia.png" width="90%"/></p>

**Methodology and Models**

* **Synthetic Generator:** A statistical-temporal approach that combines average hourly and weekly profiles, outdoor temperature effects, and an AR(1) residual component with temporal dependence.


* **Energy Forecasting:** Uses the Prophet model to evaluate prediction error (NRMSE) across real data, synthetic reinforcement, and a white noise control scenario.


* **Anomaly Detection:** Uses the Isolation Forest algorithm to assess how synthetic training data alters precision, recall, and F1-score when identifying artificially injected anomalies.



**Main Findings**

The effect of synthetic data augmentation is task-dependent. In forecasting, the synthetic reinforcement successfully reduced the prediction error compared to using only scarce real data, although it did not show a statistically significant advantage over the white noise control scenario. In anomaly detection, adding synthetic data degraded the model's F1-score primarily due to a drop in recall, indicating the model missed more true anomalies.

**Repository Contents**

All files are under [`Código/`](Código):
| Folder | Contents |
|---|---|
| `scripts/` | the eleven pipeline scripts |
| `data/` | `weather.csv`; `electricity_cleaned.csv` must be downloaded from BDG2 |
| `resultados/` | outputs of the run behind the tables and figures in the thesis |
| `resultados_verificacao/` | outputs of the verification run and the sensitivity analyses |
| `figuras/` | figures produced by the scripts |

Installation, execution and reproduction instructions are in the [README inside `Código/`](Código/README.md).
