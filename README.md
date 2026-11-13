# Customer Churn — Telco

[![CI](https://github.com/LorenzoNMachado/Customer-Churn/actions/workflows/ci.yml/badge.svg)](https://github.com/LorenzoNMachado/Customer-Churn/actions/workflows/ci.yml)

Análise exploratória, testes estatísticos e modelos de previsão para entender quais clientes de uma operadora de telecom cancelam o serviço (churn).

**Dataset:** [Telco Customer Churn (Kaggle, blastchar)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7.043 clientes e 21 colunas. A taxa de churn é de 26,5%.

## Estrutura

```
data/        WA_Fn-UseC_-Telco-Customer-Churn.csv
notebooks/   analise_churn.ipynb (análise completa, com saídas)
src/         estatistica.py (tabelas de frequência, medidas resumo, testes)
tests/       testes do src/ (pytest)
```

## Como rodar

Requer Python 3.12 ou mais novo.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/analise_churn.ipynb
```

Testes: `pip install pytest` e depois `pytest`. A cada push, o GitHub Actions roda os testes e executa o notebook inteiro.

## O que o notebook faz

1. **Tabelas de distribuição de frequência.** `tenure` (meses) usa faixas de 6 meses; `MonthlyCharges` e `TotalCharges` usam a regra de Sturges, com histograma. As categóricas aparecem numa tabela única. Cada tabela traz fi, fr, Fr %, Fi e Fr acumulada %.
2. **Medidas de posição e dispersão.** Média, mediana, moda pela fórmula de Czuber, desvio padrão, coeficiente de variação e quartis, com boxplots de quem saiu contra quem ficou.
3. **Taxa de churn por variável.** Contrato, tipo de internet, forma de pagamento e faixa de tempo de contrato.
4. **Testes de associação.** Qui-quadrado com V de Cramér para as categóricas e Mann-Whitney com correlação rank-biserial para as contínuas.
5. **Modelos.** Regressão logística, Random Forest e Gradient Boosting, comparados por validação cruzada estratificada. Depois vem a escolha do limiar de decisão pelo custo de retenção contra o custo de perda.

## Principais resultados

| Variável | Maior churn | Menor churn |
|---|---|---|
| Contrato | Mensal: 42,7% | Dois anos: 2,8% |
| Internet | Fibra óptica: 41,9% | Sem internet: 7,4% |
| Pagamento | Boleto eletrônico (electronic check): 45,3% | Cartão de crédito automático: 15,2% |
| Tempo de contrato | 0 a 6 meses: 54,3% | 66 a 72 meses: 6,1% |

- **Associações mais fortes** (V de Cramér): tipo de contrato (0,41), segurança online (0,35), suporte técnico (0,34) e tipo de internet (0,32). Gênero e serviço de telefone não têm associação com churn (p > 0,3).
- **Tempo de contrato** é a variável contínua com maior efeito (rank-biserial de -0,48): a mediana é de 10 meses para quem sai e de 38 para quem fica.
- **Modelos:** os três empatam na prática (ROC-AUC de 0,835 a 0,842 na validação cruzada). A regressão logística foi escolhida por ser a mais simples de interpretar e chega a **ROC-AUC 0,844 no teste**.
- **Limiar de decisão:** com custos hipotéticos (ação de retenção = 1, perda do cliente = 5), o limiar ótimo é 0,40 em vez de 0,50. Isso aumenta o recall de 79% para 86% e reduz o custo total no teste.
- `TotalCharges` ficou de fora dos modelos porque é praticamente `tenure × MonthlyCharges` (correlação de 1,000), o que deixava os coeficientes instáveis.
