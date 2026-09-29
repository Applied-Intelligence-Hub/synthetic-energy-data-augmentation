# Dados de entrada

O pipeline usa dois ficheiros do BDG2:

* `weather.csv` — incluído nesta pasta;
* `electricity_cleaned.csv` — **não incluído**.

O `electricity_cleaned.csv` tem 166,9 MiB e excede o limite de 100 MiB por
ficheiro imposto pelo GitHub. Tem de ser descarregado e colocado nesta pasta
antes de executar o pipeline.

O BDG2 é público e está disponível em
https://github.com/buds-lab/building-data-genome-project-2

Os identificadores dos edifícios têm de manter o formato `site_tipologia_nome`
usado no BDG2, porque o código extrai deles o *site* e a tipologia do edifício.
A seleção é restringida aos edifícios da tipologia `education`.
