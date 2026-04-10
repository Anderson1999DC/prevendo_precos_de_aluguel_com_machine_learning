# Prevendo Preços de Aluguel de Apartamentos com Machine Learning

### EDA · Regressão · Cross Validation · GridSearchCV · Random Forest

&nbsp;

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Status](https://img.shields.io/badge/Status-Concluído-28a745?style=for-the-badge)](https://github.com/Anderson1999DC/prevendo_precos_de_aluguel_com_machine_learning/blob/main)

&nbsp;
> Pipeline completo de Machine Learning para previsão do preço de aluguel de apartamentos
> em São Paulo, desde a EDA até a otimização do modelo final com GridSearchCV.

---

## Índice

- [Contexto](#contexto)
- [Objetivos](#objetivos)
- [Pipeline do Projeto](#pipeline-do-projeto)
- [Tecnologias](#tecnologias-utilizadas)
- [Dataset](#dataset)
- [Etapas Detalhadas](#etapas-detalhadas)
- [Modelos Avaliados](#modelos-avaliados)
- [Principais Resultados](#principais-resultados)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Autor](#autor)

---

## Contexto

Projeto de Machine Learning aplicado ao mercado imobiliário de São Paulo, utilizando dados reais de anúncios de apartamentos para aluguel. O objetivo é construir um modelo preditivo capaz de estimar o valor de aluguel com base nas características do imóvel.

| Etapa | Descrição |
|---|---|
| **EDA** | Análise exploratória, distribuições e correlações |
| **Limpeza** | Remoção de outliers e coordenadas inválidas |
| **Feature Engineering** | Encoding de variáveis categóricas (distritos) |
| **Modelagem** | Comparação entre 3 algoritmos de regressão |
| **Otimização** | GridSearchCV para tuning do modelo final |

---

## Objetivos

- Desenvolver um modelo de regressão supervisionada para prever o valor de aluguel
- Comparar o desempenho de Regressão Linear, Decision Tree e Random Forest
- Validar os modelos com Cross Validation (10 folds)
- Otimizar hiperparâmetros do melhor modelo via GridSearchCV
- Avaliar o modelo final com métricas completas (RMSE, MAE, R²) e análise de resíduos
- Exportar o modelo treinado para deploy via API

---

## Pipeline do Projeto

```mermaid
flowchart TD
    A([Dataset\nAnúncios SP\nAbril 2019]) --> B[EDA\nDistribuições · Correlações]
    B --> C[Limpeza\nOutliers · Coordenadas inválidas]
    C --> D[Feature Engineering\nOneHotEncoder · Distritos]
    D --> E[Modelagem\n3 algoritmos comparados]
    E --> F[Otimização\nGridSearchCV · 5 folds]
    F --> G([Modelo Final\nRandom Forest\nR² ≈ 0.80])

    style A fill:#4A90D9,color:#fff,stroke:none
    style G fill:#28a745,color:#fff,stroke:none
    style B fill:#6C757D,color:#fff,stroke:none
    style C fill:#6C757D,color:#fff,stroke:none
    style D fill:#6C757D,color:#fff,stroke:none
    style E fill:#6C757D,color:#fff,stroke:none
    style F fill:#6C757D,color:#fff,stroke:none
```

---

## Tecnologias Utilizadas

| Tecnologia | Uso no Projeto |
|---|---|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Linguagem principal |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Manipulação e limpeza dos dados |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Operações numéricas |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) | Modelos, Cross Validation e GridSearchCV |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white) | Visualizações e gráficos de avaliação |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square&logo=python&logoColor=white) | Visualizações estatísticas |
| ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white) | Mapa interativo de preços por localização |

---

## Dataset

**Fonte:** [Imóveis em São Paulo Venda / Aluguel Abril 2019](https://www.kaggle.com/datasets/argonalyst/sao-paulo-real-estate-sale-rent-april-2019) Kaggle  
**Uso:** Exclusivamente educacional

| Característica | Detalhe |
|---|---|
| Volume total | ~13.000 anúncios |
| Volume para aluguel (após limpeza) | **6.645 apartamentos** |
| Colunas originais | 16 |
| Outliers removidos | 583 registros (coordenadas inválidas e preços extremos) |
| Período | Abril de 2019 |

**Estatísticas descritivas Preço de Aluguel (após limpeza):**

| Métrica | Valor |
|---|---|
| Média | R$ 2.808 |
| Mediana | R$ 2.000 |
| Máximo considerado | R$ 15.000 |
| Área média | 89,5 m² |

---

## Etapas Detalhadas

**Análise Exploratória de Dados (EDA)**
- Visualização da distribuição geográfica dos preços via mapa interativo Plotly
- Análise de correlação entre todas as variáveis numéricas e `Price`
- Identificação de 483 imóveis com coordenadas inválidas (lat/lon = 0)
- Identificação de 103 imóveis com preços acima de R$ 15.000

**Top correlações com o preço de aluguel:**

| Feature | Correlação com Price |
|---|---|
| Size (área m²) | **0.732** |
| Condo (condomínio) | 0.700 |
| Parking (vagas) | 0.641 |
| Suites | 0.588 |
| Toilets (banheiros) | 0.583 |
| Rooms (quartos) | 0.391 |

**Preparação dos Dados**
- Remoção de imóveis com coordenadas geográficas inválidas
- Remoção de outliers de preço (acima de R$ 15.000 1,4% do dataset)
- Remoção de colunas irrelevantes: `New`, `Property Type`, `Negotiation Type`
- Encoding de `District` via `get_dummies`
- Split treino/teste: **80% / 20%** com `random_state=42`

---

## Modelos Avaliados

### Comparação de RMSE Cross Validation (10 folds)

![Comparação dos Modelos](assets/comparacao_modelos.png)

| Modelo | RMSE Médio (CV) | MAE | R² |
|---|---|---|---|
| Regressão Linear | R$ 1.171 | R$ 736 | 0.751 |
| Decision Tree | R$ 1.354 | R$ 719 | 0.667 |
| **Random Forest** | **R$ 1.027** | **R$ 552** | **0.808** |

> O **Random Forest** apresentou o menor RMSE e o maior R², capturando melhor as relações não lineares entre localização, tamanho e preço.

### Features Mais Importantes

![Feature Importance](assets/feature_importance_aluguel.png)

> Tamanho (m²) e condomínio são os fatores que mais influenciam o preço do aluguel, seguidos pelo número de suítes e a localização geográfica (longitude) que separa bairros nobres dos periféricos em SP.

---

## Principais Resultados

### Otimização com GridSearchCV

Parâmetros testados no Random Forest:

| Parâmetro | Valores Testados |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | None, 10, 20 |
| `min_samples_split` | 2, 5, 10 |
| `max_features` | sqrt, log2 |

### Performance do Modelo Final

| Métrica | Valor |
|---|---|
| **R²** | **0.797** |
| **MAE** | **R$ 577** |
| **RMSE** | **R$ 1.056** |
| Resíduo médio | R$ 24 (sem viés sistemático) |

> O modelo explica **79,7% da variação nos preços** com erro médio de R$ 577 por previsão.

### Análise de Resíduos

![Resíduos do Modelo Final](assets/residuos.png)

> Resíduos distribuídos de forma aproximadamente simétrica em torno de zero, sem viés sistemático. A maior dispersão em imóveis de alto valor é esperada imóveis de luxo têm precificação mais heterogênea.

### Aplicações do Modelo

- Apoio à precificação em imobiliárias
- Auxílio a proprietários na definição do valor de aluguel
- Geração de inteligência de mercado imobiliário
- Base para deploy via API REST (próxima etapa)

---

## Estrutura do Repositório

```
prevendo_precos_de_aluguel_com_machine_learning/
│
├──  assets/                                  # Gráficos gerados na análise
│   ├── comparacao_modelos.png
│   ├── feature_importance_aluguel.png
│   └── residuos.png
│
├──  base_de_dados/
│   └── sao-paulo-properties-april-2019.csv     # Dataset original do Kaggle
│
├──  prevendo_preco_de_alugueis_com_ML.ipynb  # Notebook completo
├──  modelo_aluguel_sp.pkl                    # Modelo Random Forest treinado
├──  colunas_modelo.pkl                       # Features esperadas pela API
├──  requirements.txt                         # Dependências do projeto
└──  README.md                                # Documentação do projeto
```

## Autor

<div align="center">

<img src="https://github.com/Anderson1999DC.png" width="100px" style="border-radius:50%"/>

**Anderson Coelho**
*Cientista de Dados*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anderson-coelho-42671634a/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Anderson1999DC)

</div>

---

<div align="center">

</div>
