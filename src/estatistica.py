"""Funções de estatística descritiva usadas no notebook de churn."""
import math

import numpy as np
import pandas as pd


def calcular_classes(dados):
    """Bordas dos intervalos de classe pela regra de Sturges."""
    dados = dados.dropna()
    n = len(dados)
    k = math.ceil(1 + 3.322 * math.log10(n))
    h = (dados.max() - dados.min()) / k
    bins = [dados.min() + i * h for i in range(k + 1)]
    # Com right=False o último intervalo é aberto à direita; estende a borda
    # para incluir o valor máximo
    bins[-1] = np.nextafter(dados.max(), np.inf)
    return bins


def _completar_frequencias(tabela, n):
    tabela['Frequência Relativa (fr)'] = tabela['Frequência Absoluta (fi)'] / n
    tabela['Fr (%)'] = tabela['Frequência Relativa (fr)'] * 100
    tabela['Freq. Acumulada (Fi)'] = tabela['Frequência Absoluta (fi)'].cumsum()
    tabela['Fr Acumulada (%)'] = tabela['Fr (%)'].cumsum()
    assert tabela['Frequência Absoluta (fi)'].sum() == n
    return tabela


def criar_tabela_frequencia(df, coluna):
    """Distribuição de frequência com intervalos de classe (variável numérica)."""
    dados = df[coluna].dropna()
    classes = pd.cut(dados, bins=calcular_classes(dados), right=False)
    fi = classes.value_counts().sort_index()

    tabela = pd.DataFrame({
        'Ponto Médio (xi)': [(c.left + c.right) / 2 for c in fi.index],
        'Frequência Absoluta (fi)': fi.values,
    }, index=pd.Index([f"{c.left:,.2f} |--- {c.right:,.2f}" for c in fi.index],
                      name=f'Classes ({coluna})'))
    return _completar_frequencias(tabela, len(dados))


def criar_tabela_frequencia_categorica(df, coluna):
    """Distribuição de frequência simples (variável categórica)."""
    dados = df[coluna].dropna()
    fi = dados.value_counts()
    tabela = pd.DataFrame({'Frequência Absoluta (fi)': fi.values},
                          index=pd.Index(fi.index, name=coluna))
    return _completar_frequencias(tabela, len(dados))


def formatar_tabela(tabela):
    """Formata para exibição sem perder os valores numéricos da tabela."""
    formatos = {
        'Ponto Médio (xi)': '{:,.2f}',
        'Frequência Relativa (fr)': '{:.4f}',
        'Fr (%)': '{:.2f}%',
        'Fr Acumulada (%)': '{:.2f}%',
    }
    return tabela.style.format({c: f for c, f in formatos.items() if c in tabela.columns})


def medidas_resumo(dados):
    """Medidas de posição e dispersão de uma variável numérica."""
    dados = dados.dropna()
    q1, q3 = dados.quantile([0.25, 0.75])
    return pd.Series({
        'Média': dados.mean(),
        'Mediana': dados.median(),
        'Moda': dados.mode().iloc[0],
        'Desvio Padrão': dados.std(),
        'Coef. Variação (%)': dados.std() / dados.mean() * 100,
        'Mínimo': dados.min(),
        'Q1': q1,
        'Q3': q3,
        'Máximo': dados.max(),
        'Amplitude Interquartil': q3 - q1,
    })


def taxa_churn(df, coluna, alvo='Churn'):
    """Quantidade de clientes e % de churn em cada categoria de `coluna`."""
    tabela = pd.DataFrame({
        'Clientes': df.groupby(coluna, observed=True)[alvo].size(),
        'Churn (%)': df.groupby(coluna, observed=True)[alvo].apply(lambda s: (s == 'Yes').mean() * 100),
    })
    return tabela.sort_values('Churn (%)', ascending=False)
