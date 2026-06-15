# Agrupamento Inteligente de Pacientes (Insuficiência Cardíaca)

Sistema **não supervisionado** que recebe os dados de um paciente desconhecido e indica a qual
**grupo** (cluster) ele pertence / é mais similar, com um **grau de similaridade**. Base:
`heart_failure_clinical_records_dataset.csv` (299 pacientes, 13 variáveis).

Segue o material de clustering do curso (`iris_cluster_treinamento.py`, `descrever_centroides.py`,
`cluster_iris_inferencia.py`) e a arquitetura dos projetos anteriores: `dados`, `treinamento`,
`descrever_grupos`, `inferencia`, `main`.

## Como rodar

```bash
cd coracao
python main.py             # pré-processa, escolhe k (cotovelo), treina o KMeans e descreve os grupos
python descrever_grupos.py # mostra o perfil de cada grupo em unidades reais
python inferencia.py       # indica o grupo de pacientes novos, com grau de similaridade
```

## a) Justificativa do metaestimador (KMeans)

O problema é **agrupar por similaridade** sem rótulos de grupo pré-definidos → tarefa de
**clustering**. Escolhi o **KMeans** porque:

- É o metaestimador de agrupamento particional padrão e eficiente para esta escala (299×11).
- Produz **centroides interpretáveis**: cada grupo vira um "paciente-protótipo" que podemos
  descrever em unidades reais (idade, fração de ejeção, etc.) — essencial para caracterizar os
  grupos e justificar clinicamente.
- Permite **atribuir um paciente novo** diretamente (`predict`) e medir a **distância aos
  centroides** (`transform`), de onde sai o grau de similaridade.
- O número de grupos não é conhecido a priori → combina naturalmente com o **método do cotovelo**.

Alternativas consideradas: clustering **hierárquico** (não exige k, mas é O(n²) e não tem um
`predict` direto para novos pacientes); **DBSCAN** (acha formas arbitrárias e ruído, mas não gera
centroides e é sensível aos parâmetros em dados mistos/altas dimensões); e **k-prototypes**
(teoricamente o mais correto para dados mistos contínuo+binário). O KMeans foi preferido por
unir interpretabilidade (centroides), atribuição simples de novos casos e aderência ao método
do cotovelo do material de aula. A ressalva do k-prototypes está discutida abaixo.

## b) Pré-processamento (todos os procedimentos)

1. **Identificação dos tipos de variável:**
   - **Contínuas (6):** `age`, `creatinine_phosphokinase`, `ejection_fraction`, `platelets`,
     `serum_creatinine`, `serum_sodium`.
   - **Binárias 0/1 (5):** `anaemia`, `diabetes`, `high_blood_pressure`, `sex`, `smoking`.
   - **Excluídas (2):** `time` (tempo de acompanhamento do estudo) e `DEATH_EVENT` (desfecho:
     morte). Nenhuma é conhecida para um paciente novo na admissão; `DEATH_EVENT` é usada só
     *depois*, para caracterizar a mortalidade de cada grupo.

2. **Tratamento das variáveis binárias (o ponto de atenção):** o KMeans usa distância
   euclidiana, então a **escala** de cada atributo importa. As contínuas têm escalas
   absurdamente diferentes (`platelets` ~10⁵ contra `ejection_fraction` 14–80); se não fossem
   normalizadas, dominariam a distância e as binárias seriam irrelevantes. Já as **binárias
   não devem ser escalonadas** como magnitudes — elas são categorias e já estão em 0/1. Então:
   - **MinMaxScaler aplicado SÓ nas contínuas** → todas vão para [0,1].
   - **Binárias mantidas em 0/1.**
   - Tudo é unido numa única matriz em [0,1], onde cada atributo contribui de forma comparável
     (exatamente o padrão do `iris_cluster_treinamento.py`: normalizar o numérico, manter o
     categórico 0/1, e juntar).
   - O **normalizador é salvo** (`pickle`) e reaplicado, idêntico, na inferência.

   > Consequência honesta: como as 5 binárias assumem os valores extremos 0/1 e as 6 contínuas
   > ficam comprimidas no meio de [0,1], **as binárias acabam dominando a formação dos grupos**
   > (os grupos saíram organizados por anemia, diabetes, sexo, fumo — ver perfis abaixo). Para
   > um agrupamento dirigido pelas variáveis clínicas contínuas, o caminho seria ponderar os
   > atributos ou usar **k-prototypes** (k-means + k-modes), que trata categóricas de forma
   > apropriada. Mantive KMeans + MinMax por aderência ao material de aula.

3. **Escolha de k:** método do cotovelo (distância máxima à reta que liga o primeiro ao último
   ponto da curva de distorções) → **k = 6**, confirmado pela silhueta (melhor faixa em 6–7).
   Gráfico em `plots/Coracao_cotovelo.png`.

## Perfil dos grupos (k = 6, centroides em unidades reais)

Contínuas desnormalizadas (`inverse_transform`); binárias = proporção de pacientes com valor 1.

| grupo | n | age | ejection_fraction | serum_creatinine | anaemia | diabetes | sex | smoking | mortalidade % |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 47 | 62 | 38 | 1.41 | 0.00 | 0.00 | 0.68 | 0.00 | 36.2 |
| 1 | 58 | 63 | 39 | 1.49 | 1.00 | 0.29 | 0.78 | 0.00 | 31.0 |
| 2 | 57 | 57 | 40 | 1.39 | 0.26 | 1.00 | 0.44 | 0.00 | 28.1 |
| 3 | 43 | 61 | 38 | 1.33 | 0.51 | 0.51 | 0.00 | 0.05 | 37.2 |
| 4 | 60 | 60 | 36 | 1.28 | 0.00 | 0.33 | 1.00 | 1.00 | 28.3 |
| 5 | 34 | 63 | 38 | 1.51 | 1.00 | 0.26 | 0.94 | 1.00 | 35.3 |

Os grupos se diferenciam sobretudo pelas comorbidades (anemia, diabetes), sexo e tabagismo; os
indicadores contínuos são parecidos entre grupos, e a mortalidade varia pouco (28–37%, em torno
dos ~32% gerais da base).

## c) Inferência funcionando (grau de similaridade)

`inferencia.py` aplica o mesmo pré-processamento (normalizador salvo) a um paciente novo e indica
o grupo via `predict`; o grau de similaridade vem das distâncias aos centroides (`transform`),
convertidas em proximidade normalizada.

```
=== Paciente 1 (idoso, anêmico, fração de ejeção baixa) ===
  Grupo indicado: 1  (similaridade 20.1% | distância ao centroide 0.940)
  2o grupo mais próximo: 5 (similaridade 17.1%)

=== Paciente 2 (mais jovem, diabético) ===
  Grupo indicado: 2  (similaridade 19.2% ...)
```

O Paciente 1 (anêmico) cai no grupo 1 (grupo dos anêmicos) e o Paciente 2 (diabético) no grupo 2
(diabéticos) — coerente. As similaridades modestas (~20% entre 6 grupos) refletem que os grupos
não são muito separados (silhueta 0,25), o que é típico de dados clínicos.

## Licença

MIT.
