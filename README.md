# Customer Churn — Telco

Análise exploratória e modelo baseline para entender quais clientes de uma operadora de telecom cancelam o serviço (churn).

**Dataset:** [Telco Customer Churn (Kaggle, blastchar)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7.043 clientes e 21 colunas. A taxa de churn é de 26,5%.

## Estrutura

```
data/        WA_Fn-UseC_-Telco-Customer-Churn.csv
notebooks/   analise_churn.ipynb (análise completa, com saídas)
src/         estatistica.py (tabelas de frequência, medidas resumo, taxa de churn)
```

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/analise_churn.ipynb
```

## O que o notebook faz

1. **Tabelas de distribuição de frequência**: intervalos de classe pela regra de Sturges para as variáveis contínuas (com histograma) e frequência simples para as categóricas. Cada tabela traz fi, fr, Fr %, Fi e Fr acumulada %.
2. **Medidas de posição e dispersão**: média, mediana, moda, desvio padrão, coeficiente de variação e quartis, mais boxplots comparando quem saiu e quem ficou.
3. **Taxa de churn por variável**: contrato, tipo de internet, forma de pagamento e faixa de tempo de contrato (tenure).
4. **Modelo baseline**: regressão logística com split estratificado e `class_weight='balanced'`, avaliada por recall, F1 e ROC-AUC.

## Principais resultados

| Variável | Maior churn | Menor churn |
|---|---|---|
| Contrato | Mensal: 42,7% | Dois anos: 2,8% |
| Internet | Fibra óptica: 41,9% | Sem internet: 7,4% |
| Pagamento | Boleto eletrônico (electronic check): 45,3% | Cartão de crédito automático: 15,2% |
| Tempo de contrato | 0 a 5 meses: 54,3% | 67 a 72 meses: 5,3% |

- O risco se concentra nos **primeiros meses**: mais da metade dos clientes com até 5 meses de casa cancela.
- Quem sai paga **mensalidade mais alta** (mediana de cerca de 80, contra 65 de quem fica).
- O modelo baseline chega a **ROC-AUC 0,846** e identifica **80% dos clientes que saem** (recall), com precisão de 52%. Para comparar: sempre prever "não cancela" acerta 73,5%, o que mostra por que a acurácia sozinha engana nessa base.

## Próximos passos

- Testar modelos baseados em árvores (Random Forest, Gradient Boosting) e ajustar o limiar de decisão conforme o custo de reter um cliente.
- Validação cruzada em vez de um único split.
