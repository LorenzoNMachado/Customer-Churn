# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Análise de Churn — Telco Customer Churn
#
# Análise da base de clientes de uma operadora de telecom (7.043 clientes, 21 colunas) para entender quem cancela o serviço (`Churn = Yes`) e quando.
#
# 1. Tabelas de distribuição de frequência (contínuas e categóricas)
# 2. Medidas de posição e dispersão
# 3. Taxa de churn por variável, com intervalo de confiança
# 4. Análise de sobrevivência (Kaplan-Meier)
# 5. Testes de associação com o churn
# 6. Modelos de previsão: variáveis derivadas, comparação, calibração, limiar por custo e explicação por cliente

# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

sys.path.append(os.path.abspath('../src'))
from estatistica import (calcular_classes, classificar, criar_tabela_frequencia,
                         formatar_tabela, kaplan_meier, mediana_sobrevivencia, medidas_resumo,
                         moda_czuber, resumo_categoricas, taxa_churn, teste_log_rank,
                         teste_mann_whitney, teste_qui_quadrado)
from modelagem import contribuicoes, criar_features, principais_motivos

df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# TotalCharges vem como texto: 11 clientes com tenure 0 têm o campo vazio
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# SeniorCitizen é binária (0/1), não contínua
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

# tenure é discreta (meses): faixas de 6 meses no lugar de Sturges
LARGURA = {'tenure': 6}
taxa_geral = (df['Churn'] == 'Yes').mean() * 100

# Gráficos usados no README
FIGURAS = '../reports/figures'
os.makedirs(FIGURAS, exist_ok=True)

# Estilo dos gráficos: cores fixas por série, grade discreta
AZUL, LARANJA, VERDE, CINZA = '#2a78d6', '#eb6834', '#1baf7a', '#8a8a85'
TEXTO_SECUNDARIO = '#55554f'
plt.rcParams.update({
    'figure.dpi': 110,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': '#e4e4e0',
    'grid.linewidth': 0.8,
})

# %% [markdown]
# ## Variáveis disponíveis

# %%
continuas = df.select_dtypes(include='number').columns.tolist()
categoricas = [c for c in df.columns if c not in continuas and c != 'customerID']

print('VARIÁVEIS NUMÉRICAS (tabela com intervalos de classe):')
for col in continuas:
    print(f'   -> {col}')
print('\nVARIÁVEIS CATEGÓRICAS (tabela de frequência simples):')
for col in categoricas:
    print(f'   -> {col}')

# %% [markdown]
# ## 1. Distribuição de frequência — variáveis contínuas
#
# `tenure` usa faixas de 6 meses, porque é contada em meses inteiros. `MonthlyCharges` e `TotalCharges` usam a regra de Sturges. O histograma é desenhado sobre as mesmas classes da tabela.

# %%
for coluna in continuas:
    largura = LARGURA.get(coluna)
    display(formatar_tabela(criar_tabela_frequencia(df, coluna, largura)))

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(df[coluna].dropna(), bins=calcular_classes(df[coluna], largura), color=AZUL,
            edgecolor='white', linewidth=2)
    ax.set_title(f'Distribuição de {coluna}', loc='left')
    ax.set_xlabel(coluna)
    ax.set_ylabel('Clientes (fi)')
    ax.grid(axis='x', visible=False)
    plt.show()

# %% [markdown]
# ## 2. Distribuição de frequência — variáveis categóricas

# %%
display(formatar_tabela(resumo_categoricas(df, categoricas)))

# %% [markdown]
# ## 3. Medidas de posição e dispersão
#
# A moda das variáveis contínuas é estimada pela fórmula de Czuber sobre a classe modal da tabela de frequência, já que quase não há valores repetidos exatos.

# %%
resumo = pd.DataFrame({c: medidas_resumo(df[c], LARGURA.get(c)) for c in continuas})
display(resumo.style.format('{:,.2f}'))

for coluna in continuas:
    classe, _ = moda_czuber(df[coluna], LARGURA.get(coluna))
    print(f'Classe modal de {coluna}: {classe.left:,.2f} |--- {classe.right:,.2f}')

# %%
fig, axes = plt.subplots(1, len(continuas), figsize=(10, 3.6))
for ax, coluna in zip(axes, continuas):
    grupos = [df.loc[df['Churn'] == g, coluna].dropna() for g in ('No', 'Yes')]
    caixas = ax.boxplot(grupos, patch_artist=True, widths=0.5,
                        medianprops={'color': 'black', 'linewidth': 1.5},
                        flierprops={'markersize': 3, 'markeredgecolor': CINZA})
    for caixa, cor in zip(caixas['boxes'], (AZUL, LARANJA)):
        caixa.set(facecolor=cor, edgecolor=cor, alpha=0.85)
    ax.set_xticks([1, 2], ['Ficou (No)', 'Saiu (Yes)'])
    ax.set_title(coluna, loc='left')
    ax.grid(axis='x', visible=False)
fig.suptitle('Clientes que saíram x clientes que ficaram', x=0.01, ha='left')
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Taxa de churn por variável
#
# As barras de erro são intervalos de confiança de 95% (Wilson). Quando os intervalos de dois grupos não se sobrepõem, a diferença entre eles é sólida. A linha tracejada marca a taxa geral da base.

# %%
print(f'Taxa geral de churn: {taxa_geral:.1f}%')
df_faixas = df.assign(**{'Faixa de tenure (meses)': classificar(df['tenure'], LARGURA['tenure'])})

for coluna in ['Contract', 'InternetService', 'PaymentMethod', 'Faixa de tenure (meses)']:
    tabela = taxa_churn(df_faixas, coluna)
    if coluna.startswith('Faixa'):
        tabela = tabela.sort_index()  # faixas seguem a ordem natural
    display(tabela.style.format({c: '{:.1f}%' for c in tabela.columns if '%' in c}))

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(tabela) + 1.2))
    erro = [tabela['Churn (%)'] - tabela['IC 95% inf (%)'], tabela['IC 95% sup (%)'] - tabela['Churn (%)']]
    ax.barh(tabela.index.astype(str), tabela['Churn (%)'], color=AZUL, height=0.6,
            xerr=erro, error_kw={'ecolor': TEXTO_SECUNDARIO, 'capsize': 3, 'elinewidth': 1})
    ax.invert_yaxis()
    ax.axvline(taxa_geral, color=CINZA, linestyle='--', linewidth=1)
    ax.annotate(f'taxa geral {taxa_geral:.1f}%', xy=(taxa_geral, 1), xycoords=('data', 'axes fraction'),
                xytext=(4, 2), textcoords='offset points', fontsize=8, color=TEXTO_SECUNDARIO, va='bottom')
    for i, (v, sup) in enumerate(zip(tabela['Churn (%)'], tabela['IC 95% sup (%)'])):
        ax.text(sup + 0.8, i, f'{v:.1f}%', va='center', fontsize=9)
    ax.set_xlim(0, tabela['IC 95% sup (%)'].max() * 1.15)
    ax.set_xlabel('Churn (%), com IC 95%')
    ax.set_title(f'Taxa de churn por {coluna}', loc='left', pad=14)
    ax.grid(axis='y', visible=False)
    if coluna == 'Contract':
        fig.savefig(f'{FIGURAS}/churn_por_contrato.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 5. Análise de sobrevivência
#
# `tenure` é o tempo até o cliente sair. Para quem ainda não saiu, é só o tempo observado até agora: um dado **censurado**, porque o cliente pode sair depois. A curva de **Kaplan-Meier** estima, para cada mês, a proporção de clientes que continuam na base, aproveitando também esses dados censurados.
#
# O teste **log-rank** compara as curvas entre grupos.

# %%
tempo, evento = df['tenure'], df['Churn'] == 'Yes'
cores = {'Month-to-month': AZUL, 'One year': LARANJA, 'Two year': VERDE}

fig, ax = plt.subplots(figsize=(8, 4.2))
resumo_sobrevivencia = {}
for contrato, cor in cores.items():
    m = df['Contract'] == contrato
    curva = kaplan_meier(tempo[m], evento[m])
    s = curva['Sobrevivência']
    ax.step(s.index, s.values * 100, where='post', color=cor, linewidth=2, label=contrato)
    ax.annotate(f'{contrato} ({s.iloc[-1]:.0%})', (72, s.iloc[-1] * 100), xytext=(6, 0),
                textcoords='offset points', va='center', fontsize=9)
    resumo_sobrevivencia[contrato] = {
        'Clientes': int(m.sum()),
        'Mediana (meses)': mediana_sobrevivencia(curva),
        'Permanece 12 meses': s.asof(12),
        'Permanece 24 meses': s.asof(24),
        'Permanece 72 meses': s.asof(72),
    }
ax.axhline(50, color=CINZA, linestyle='--', linewidth=1)
ax.set_xlim(0, 72)
ax.set_ylim(0, 102)
ax.set_xticks(range(0, 73, 12))
ax.set_xlabel('Meses de contrato (tenure)')
ax.set_ylabel('Clientes que permanecem (%)')
ax.set_title('Curva de sobrevivência (Kaplan-Meier) por tipo de contrato', loc='left')
ax.legend(loc='lower left', frameon=False)
fig.savefig(f'{FIGURAS}/sobrevivencia_por_contrato.png', dpi=150, bbox_inches='tight')
plt.show()

fmt_pct = {c: '{:.1%}' for c in ['Permanece 12 meses', 'Permanece 24 meses', 'Permanece 72 meses']}
display(pd.DataFrame(resumo_sobrevivencia).T.style.format({**fmt_pct, 'Clientes': '{:,.0f}', 'Mediana (meses)': '{:.0f}'},
                                                            na_rep='não atinge 50%'))

# %%
log_rank = pd.DataFrame({c: teste_log_rank(tempo, evento, df[c])
                         for c in ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport', 'gender']}).T
display(log_rank.sort_values('Qui-quadrado', ascending=False)
                .style.format({'Qui-quadrado': '{:,.1f}', 'gl': '{:.0f}', 'p-valor': '{:.2e}'}))

# %% [markdown]
# ## 6. Testes de associação com o churn
#
# - **Categóricas:** qui-quadrado de independência. O tamanho do efeito é medido pelo **V de Cramér** (cerca de 0,1 é pequeno, 0,3 é médio e 0,5 é grande).
# - **Contínuas:** Mann-Whitney comparando quem saiu e quem ficou, porque as distribuições não são normais. O efeito é medido pela **correlação rank-biserial**, que vai de -1 a 1 e é positiva quando quem saiu tem valores maiores.
#
# Com 7 mil clientes, quase qualquer diferença dá p-valor < 0,05. Por isso as tabelas estão ordenadas pelo tamanho do efeito, que é o que mostra a força da associação.

# %%
categoricas_X = [c for c in categoricas if c != 'Churn']

qui2 = pd.DataFrame({c: teste_qui_quadrado(df, c) for c in categoricas_X}).T
qui2 = qui2.sort_values('V de Cramér', ascending=False)
display(qui2.style.format({'Qui-quadrado': '{:,.1f}', 'gl': '{:.0f}', 'p-valor': '{:.2e}',
                           'V de Cramér': '{:.3f}'}))

mw = pd.DataFrame({c: teste_mann_whitney(df, c) for c in continuas}).T
mw = mw.sort_values('Rank-biserial', key=abs, ascending=False)
display(mw.style.format({'Mediana (saiu)': '{:,.2f}', 'Mediana (ficou)': '{:,.2f}',
                         'U': '{:,.0f}', 'p-valor': '{:.2e}', 'Rank-biserial': '{:+.3f}'}))

# %% [markdown]
# ## 7. Modelos de previsão
#
# **Colinearidade:** `TotalCharges` é praticamente `tenure × MonthlyCharges`. Com as três variáveis juntas, os coeficientes da regressão logística ficam instáveis e trocam de sinal (por exemplo, `TotalCharges` parecia aumentar o churn). Por isso `TotalCharges` fica de fora dos modelos.

# %%
estimado = df['tenure'] * df['MonthlyCharges']
print(f"Correlação entre TotalCharges e tenure × MonthlyCharges: {df['TotalCharges'].corr(estimado):.3f}")

# %% [markdown]
# ### 7.1 Variáveis derivadas
#
# - `NumServicos`: quantos dos 8 serviços adicionais o cliente contrata;
# - `SemProtecao`: tem internet, mas nem segurança online nem suporte técnico.
#
# A regressão logística é comparada com e sem essas variáveis, por validação cruzada estratificada de 5 partes no conjunto de treino. O conjunto de teste (25%) só é usado no final.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df_modelo = criar_features(df)
y = (df_modelo['Churn'] == 'Yes').astype(int)
treino, teste = train_test_split(df_modelo.index, test_size=0.25, stratify=y, random_state=42)
y_treino, y_teste = y[treino], y[teste]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def pipeline(classificador, numericas, categoricas):
    return make_pipeline(
        ColumnTransformer([
            ('num', StandardScaler(), numericas),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categoricas),
        ]),
        classificador,
    )

conjuntos = {
    'Originais': (['tenure', 'MonthlyCharges'], categoricas_X),
    'Originais + derivadas': (['tenure', 'MonthlyCharges', 'NumServicos'], categoricas_X + ['SemProtecao']),
}
auc_conjuntos = {}
for nome, (num, cat) in conjuntos.items():
    modelo = pipeline(LogisticRegression(max_iter=1000, class_weight='balanced'), num, cat)
    notas = cross_val_score(modelo, df_modelo.loc[treino, num + cat], y_treino, cv=cv, scoring='roc_auc')
    auc_conjuntos[nome] = {'roc_auc (média)': notas.mean(), 'roc_auc (desvio)': notas.std()}
display(pd.DataFrame(auc_conjuntos).T.style.format('{:.4f}'))

escolhido = max(auc_conjuntos, key=lambda k: auc_conjuntos[k]['roc_auc (média)'])
numericas_X, categoricas_modelo = conjuntos[escolhido]
X_treino = df_modelo.loc[treino, numericas_X + categoricas_modelo]
X_teste = df_modelo.loc[teste, numericas_X + categoricas_modelo]
print(f'Conjunto usado daqui em diante: {escolhido}')

# %% [markdown]
# ### 7.2 Comparação de modelos
#
# Três modelos, todos com `class_weight='balanced'`, porque só 26,5% dos clientes saem.

# %%
modelos = {
    'Regressão Logística': pipeline(LogisticRegression(max_iter=1000, class_weight='balanced'),
                                    numericas_X, categoricas_modelo),
    'Random Forest': pipeline(RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight='balanced',
                                                     random_state=42, n_jobs=-1), numericas_X, categoricas_modelo),
    'Gradient Boosting': pipeline(HistGradientBoostingClassifier(class_weight='balanced', random_state=42),
                                  numericas_X, categoricas_modelo),
}

resultados = {}
for nome, modelo in modelos.items():
    r = cross_validate(modelo, X_treino, y_treino, cv=cv, scoring=['roc_auc', 'recall', 'precision', 'f1'])
    resultados[nome] = {f'{m} (média)': r[f'test_{m}'].mean() for m in ['roc_auc', 'recall', 'precision', 'f1']}
    resultados[nome]['roc_auc (desvio)'] = r['test_roc_auc'].std()

comparacao = pd.DataFrame(resultados).T.sort_values('roc_auc (média)', ascending=False)
display(comparacao.style.format('{:.3f}'))
melhor = comparacao.index[0]
print(f'Melhor modelo na validação cruzada: {melhor}')

# %% [markdown]
# ### 7.3 Calibração das probabilidades
#
# Com `class_weight='balanced'`, o modelo dá o mesmo peso total às duas classes, como se metade dos clientes saísse. As "probabilidades" ficam infladas: um 0,40 não quer dizer 40% de chance real de cancelar. Na regressão logística, essa distorção é basicamente um deslocamento do intercepto, que a **calibração sigmoide** (Platt, via `CalibratedClassifierCV`) corrige diretamente. A calibração isotônica também foi testada, mas dava riscos de 100% nas pontas da escala, um excesso de confiança.
#
# No gráfico, um modelo bem calibrado fica sobre a diagonal: entre os clientes com risco previsto de 30%, cerca de 30% de fato saem. O **Brier score** mede o erro das probabilidades (quanto menor, melhor).

# %%
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_predict

sem_calibracao = modelos[melhor].fit(X_treino, y_treino)
calibrado = CalibratedClassifierCV(modelos[melhor], method='sigmoid', cv=5).fit(X_treino, y_treino)
prob = {
    'Sem calibração': sem_calibracao.predict_proba(X_teste)[:, 1],
    'Calibrado': calibrado.predict_proba(X_teste)[:, 1],
}

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot([0, 1], [0, 1], color=CINZA, linestyle='--', linewidth=1)
ax.annotate('calibração perfeita', (0.80, 0.74), rotation=42, fontsize=8, color=TEXTO_SECUNDARIO)
for (nome, p), cor in zip(prob.items(), (LARANJA, AZUL)):
    real, previsto = calibration_curve(y_teste, p, n_bins=10, strategy='quantile')
    ax.plot(previsto, real, marker='o', markersize=6, color=cor, linewidth=2,
            markeredgecolor='white', markeredgewidth=1.5,
            label=f'{nome} (Brier {brier_score_loss(y_teste, p):.3f})')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Risco previsto')
ax.set_ylabel('Proporção real de churn')
ax.set_title(f'Calibração — {melhor} (teste)', loc='left')
ax.legend(loc='upper left', frameon=False)
plt.show()

# %% [markdown]
# ### 7.4 Limiar de decisão pelo custo
#
# Os custos abaixo são **hipotéticos**; ajuste para a realidade do negócio:
#
# - `CUSTO_ACAO`: custo de uma ação de retenção (contato e desconto), aplicada a todo cliente marcado como risco;
# - `CUSTO_PERDA`: custo de perder um cliente que cancelaria e não foi abordado.
#
# Com probabilidades calibradas, a teoria diz que vale agir quando `risco × CUSTO_PERDA > CUSTO_ACAO`, ou seja, quando o risco passa de `CUSTO_ACAO / CUSTO_PERDA`. O limiar também é buscado empiricamente, com as probabilidades calibradas da validação cruzada no treino, para conferir.

# %%
CUSTO_ACAO = 1
CUSTO_PERDA = 5

def custo_total(y_real, prob, limiar):
    tn, fp, fn, tp = confusion_matrix(y_real, prob >= limiar).ravel()
    return (tp + fp) * CUSTO_ACAO + fn * CUSTO_PERDA

prob_cv = cross_val_predict(CalibratedClassifierCV(modelos[melhor], method='sigmoid', cv=5),
                            X_treino, y_treino, cv=cv, method='predict_proba')[:, 1]
limiares = np.round(np.arange(0.05, 0.96, 0.01), 2)
custos = np.array([custo_total(y_treino, prob_cv, l) for l in limiares])
limiar = limiares[custos.argmin()]
limiar_teorico = CUSTO_ACAO / CUSTO_PERDA
print(f'Limiar teórico: {limiar_teorico:.2f} | limiar empírico: {limiar:.2f}')

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(limiares, custos, color=AZUL, linewidth=2)
ax.plot(limiar, custos.min(), 'o', color=AZUL, markersize=8, markeredgecolor='white', markeredgewidth=2)
ax.annotate(f'mínimo em {limiar:.2f}', (limiar, custos.min()), xytext=(0, 12),
            textcoords='offset points', fontsize=9, ha='center')
ax.axvline(limiar_teorico, color=CINZA, linestyle='--', linewidth=1)
ax.annotate(f'teórico {limiar_teorico:.2f}', xy=(limiar_teorico, 1), xycoords=('data', 'axes fraction'),
            xytext=(4, -12), textcoords='offset points', fontsize=8, color=TEXTO_SECUNDARIO)
ax.set_xlabel('Limiar de decisão (risco calibrado)')
ax.set_ylabel('Custo total (treino)')
ax.set_title(f'Custo por limiar — {melhor} (ação = {CUSTO_ACAO}, perda = {CUSTO_PERDA})', loc='left')
plt.show()

# %% [markdown]
# ### 7.5 Avaliação final no teste

# %%
prob_teste = prob['Calibrado']
print(f'Modelo: {melhor} calibrado | ROC-AUC no teste: {roc_auc_score(y_teste, prob_teste):.3f}')
print(f'Acurácia de sempre prever "No": {1 - y_teste.mean():.3f}\n')
for nome_limiar, l in [('0,50', 0.5), (f'por custo {limiar:.2f}', limiar)]:
    print(f'--- Limiar {nome_limiar} | custo no teste: {custo_total(y_teste, prob_teste, l):,}')
    print(classification_report(y_teste, prob_teste >= l, target_names=['Ficou (No)', 'Saiu (Yes)'], digits=3))

# %% [markdown]
# ### 7.6 Por que cada cliente está em risco
#
# A explicação usa a regressão logística, um modelo linear. Nele, a contribuição de cada variável para a previsão de um cliente é `coeficiente × (valor do cliente − média)`, em log-odds. Esse é exatamente o valor SHAP de um modelo linear, então não é preciso nenhuma biblioteca extra. As colunas one-hot de uma mesma variável são somadas, e a calibração não muda a ordem dos clientes nem os motivos.
#
# Primeiro os coeficientes globais, depois os 10 clientes do teste com maior risco e os 3 fatores que mais pesam para cada um.
#
# `MonthlyCharges` aparece com sinal negativo, apesar de quem sai pagar mais. Isso acontece porque a mensalidade é quase toda explicada pelos serviços contratados (fibra, streaming etc.), que já estão no modelo. O coeficiente mede o efeito de pagar mais **pelos mesmos serviços**, e não o efeito de ter uma mensalidade alta em geral.

# %%
logistica = modelos['Regressão Logística'].fit(X_treino, y_treino)
nomes = logistica[0].get_feature_names_out()
coeficientes = pd.Series(logistica[-1].coef_[0], index=nomes).sort_values()
display(pd.concat([coeficientes.head(5), coeficientes.tail(5)])
          .rename('Coeficiente').to_frame()
          .style.format('{:+.3f}'))

# %%
risco = pd.Series(prob_teste, index=X_teste.index, name='Risco calibrado')
top = risco.sort_values(ascending=False).head(10).index
motivos = principais_motivos(contribuicoes(logistica, X_treino, X_teste.loc[top]), X_teste)

ranking = pd.concat([df.loc[top, ['customerID']], risco[top], motivos], axis=1)
display(ranking.style.format({'Risco calibrado': '{:.0%}'}).hide(axis='index'))
