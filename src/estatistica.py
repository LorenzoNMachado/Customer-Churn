"""Funções de estatística descritiva e inferencial usadas no notebook de churn."""
import math

import numpy as np
import pandas as pd
from scipy import stats


def calcular_classes(dados, largura=None):
    """Bordas dos intervalos de classe.

    Sem `largura`, usa a regra de Sturges. Com `largura`, cria classes de
    tamanho fixo a partir do mínimo (útil para variáveis discretas, como meses).
    """
    dados = dados.dropna()
    if largura is None:
        k = math.ceil(1 + 3.322 * math.log10(len(dados)))
        largura = (dados.max() - dados.min()) / k
    else:
        k = math.ceil((dados.max() - dados.min()) / largura)
    bins = [dados.min() + i * largura for i in range(k + 1)]
    # Com right=False o último intervalo é aberto à direita; estende a borda
    # para incluir o valor máximo
    if bins[-1] <= dados.max():
        bins[-1] = np.nextafter(dados.max(), np.inf)
    return bins


def _rotulo(intervalo):
    # Bordas inteiras dispensam casas decimais (ex.: "0 |--- 6")
    fmt = '{:,.0f}' if all(round(b, 6) == round(b) for b in (intervalo.left, intervalo.right)) else '{:,.2f}'
    return f"{fmt.format(intervalo.left)} |--- {fmt.format(intervalo.right)}"


def classificar(dados, largura=None):
    """Classe de cada valor, com rótulos no formato "a |--- b"."""
    classes = pd.cut(dados, bins=calcular_classes(dados, largura), right=False)
    return classes.cat.rename_categories([_rotulo(c) for c in classes.cat.categories])


def _completar_frequencias(tabela, n):
    tabela['Frequência Relativa (fr)'] = tabela['Frequência Absoluta (fi)'] / n
    tabela['Fr (%)'] = tabela['Frequência Relativa (fr)'] * 100
    tabela['Freq. Acumulada (Fi)'] = tabela['Frequência Absoluta (fi)'].cumsum()
    tabela['Fr Acumulada (%)'] = tabela['Fr (%)'].cumsum()
    assert tabela['Frequência Absoluta (fi)'].sum() == n
    return tabela


def criar_tabela_frequencia(df, coluna, largura=None):
    """Distribuição de frequência com intervalos de classe (variável numérica)."""
    dados = df[coluna].dropna()
    classes = pd.cut(dados, bins=calcular_classes(dados, largura), right=False)
    fi = classes.value_counts().sort_index()

    tabela = pd.DataFrame({
        'Ponto Médio (xi)': [(c.left + c.right) / 2 for c in fi.index],
        'Frequência Absoluta (fi)': fi.values,
    }, index=pd.Index([_rotulo(c) for c in fi.index], name=f'Classes ({coluna})'))
    return _completar_frequencias(tabela, len(dados))


def criar_tabela_frequencia_categorica(df, coluna):
    """Distribuição de frequência simples (variável categórica)."""
    dados = df[coluna].dropna()
    fi = dados.value_counts()
    tabela = pd.DataFrame({'Frequência Absoluta (fi)': fi.values},
                          index=pd.Index(fi.index, name=coluna))
    return _completar_frequencias(tabela, len(dados))


def resumo_categoricas(df, colunas):
    """Uma única tabela com fi e Fr (%) de cada categoria de várias variáveis."""
    partes = {c: criar_tabela_frequencia_categorica(df, c)[['Frequência Absoluta (fi)', 'Fr (%)']]
              for c in colunas}
    tabela = pd.concat(partes, names=['Variável', 'Categoria'])
    return tabela


def formatar_tabela(tabela):
    """Formata para exibição sem perder os valores numéricos da tabela."""
    formatos = {
        'Ponto Médio (xi)': '{:,.2f}',
        'Frequência Relativa (fr)': '{:.4f}',
        'Fr (%)': '{:.2f}%',
        'Fr Acumulada (%)': '{:.2f}%',
    }
    return tabela.style.format({c: f for c, f in formatos.items() if c in tabela.columns})


def moda_czuber(dados, largura=None):
    """Classe modal e moda pela fórmula de Czuber: Mo = li + d1 / (d1 + d2) * h."""
    dados = dados.dropna()
    bins = calcular_classes(dados, largura)
    fi = pd.cut(dados, bins=bins, right=False).value_counts().sort_index().to_numpy()
    i = int(fi.argmax())
    anterior = fi[i - 1] if i > 0 else 0
    posterior = fi[i + 1] if i < len(fi) - 1 else 0
    d1, d2 = fi[i] - anterior, fi[i] - posterior
    h = bins[i + 1] - bins[i]
    moda = bins[i] + d1 / (d1 + d2) * h if d1 + d2 else bins[i] + h / 2
    return pd.Interval(bins[i], bins[i + 1], closed='left'), moda


def medidas_resumo(dados, largura=None):
    """Medidas de posição e dispersão de uma variável numérica."""
    dados = dados.dropna()
    q1, q3 = dados.quantile([0.25, 0.75])
    _, moda = moda_czuber(dados, largura)
    return pd.Series({
        'Média': dados.mean(),
        'Mediana': dados.median(),
        'Moda (Czuber)': moda,
        'Desvio Padrão': dados.std(),
        'Coef. Variação (%)': dados.std() / dados.mean() * 100,
        'Mínimo': dados.min(),
        'Q1': q1,
        'Q3': q3,
        'Máximo': dados.max(),
        'Amplitude Interquartil': q3 - q1,
    })


def intervalo_wilson(sucessos, n, confianca=0.95):
    """Intervalo de confiança de Wilson para uma proporção."""
    ic = stats.binomtest(int(sucessos), int(n)).proportion_ci(confianca, method='wilson')
    return ic.low, ic.high


def taxa_churn(df, coluna, alvo='Churn'):
    """Clientes, % de churn e IC 95% (Wilson) em cada categoria de `coluna`."""
    grupos = df.groupby(coluna, observed=True)[alvo]
    clientes = grupos.size()
    saidas = grupos.apply(lambda s: (s == 'Yes').sum())
    ic = [intervalo_wilson(k, n) for k, n in zip(saidas, clientes)]
    tabela = pd.DataFrame({
        'Clientes': clientes,
        'Churn (%)': saidas / clientes * 100,
        'IC 95% inf (%)': [baixo * 100 for baixo, _ in ic],
        'IC 95% sup (%)': [alto * 100 for _, alto in ic],
    })
    return tabela.sort_values('Churn (%)', ascending=False)


def teste_qui_quadrado(df, coluna, alvo='Churn'):
    """Qui-quadrado de independência entre `coluna` e o alvo, com V de Cramér."""
    contingencia = pd.crosstab(df[coluna], df[alvo])
    qui2, p, gl, _ = stats.chi2_contingency(contingencia)
    n = contingencia.to_numpy().sum()
    v_cramer = math.sqrt(qui2 / (n * (min(contingencia.shape) - 1)))
    return pd.Series({'Qui-quadrado': qui2, 'gl': gl, 'p-valor': p, 'V de Cramér': v_cramer})


def teste_mann_whitney(df, coluna, alvo='Churn'):
    """Mann-Whitney entre quem saiu e quem ficou, com correlação rank-biserial."""
    saiu = df.loc[df[alvo] == 'Yes', coluna].dropna()
    ficou = df.loc[df[alvo] == 'No', coluna].dropna()
    u, p = stats.mannwhitneyu(saiu, ficou, alternative='two-sided')
    # Positivo: quem saiu tende a ter valores maiores
    rank_biserial = 2 * u / (len(saiu) * len(ficou)) - 1
    return pd.Series({'Mediana (saiu)': saiu.median(), 'Mediana (ficou)': ficou.median(),
                      'U': u, 'p-valor': p, 'Rank-biserial': rank_biserial})


def kaplan_meier(tempo, evento):
    """Curva de sobrevivência de Kaplan-Meier.

    `tempo` é o tempo observado e `evento` indica se o evento (churn) ocorreu;
    quando não ocorreu, a observação é censurada naquele tempo.
    """
    tempo = np.asarray(tempo)
    evento = np.asarray(evento, dtype=bool)
    linhas = [(0, len(tempo), 0, 1.0)]
    s = 1.0
    for t in np.unique(tempo[evento]):
        em_risco = (tempo >= t).sum()
        eventos = (evento & (tempo == t)).sum()
        s *= 1 - eventos / em_risco
        linhas.append((t, em_risco, eventos, s))
    curva = pd.DataFrame(linhas, columns=['Tempo', 'Em risco', 'Eventos', 'Sobrevivência']).set_index('Tempo')
    return curva[~curva.index.duplicated(keep='last')]


def mediana_sobrevivencia(curva):
    """Primeiro tempo em que a sobrevivência chega a 50% (NaN se não chegar)."""
    abaixo = curva.index[curva['Sobrevivência'] <= 0.5]
    return abaixo[0] if len(abaixo) else np.nan


def teste_log_rank(tempo, evento, grupo):
    """Teste log-rank para comparar curvas de sobrevivência de k grupos."""
    tempo, grupo = np.asarray(tempo), np.asarray(grupo)
    evento = np.asarray(evento, dtype=bool)
    grupos = pd.unique(grupo)
    k = len(grupos)
    observados, esperados, variancia = np.zeros(k), np.zeros(k), np.zeros((k, k))
    for t in np.unique(tempo[evento]):
        em_risco = tempo >= t
        n_j = np.array([(em_risco & (grupo == g)).sum() for g in grupos])
        d_j = np.array([(evento & (tempo == t) & (grupo == g)).sum() for g in grupos])
        n, d = n_j.sum(), d_j.sum()
        observados += d_j
        esperados += d * n_j / n
        if n > 1:
            variancia += d * (n - d) / (n - 1) * (np.diag(n_j / n) - np.outer(n_j, n_j) / n ** 2)
    # Um grupo é redundante: os desvios somam zero
    desvio = (observados - esperados)[:-1]
    qui2 = float(desvio @ np.linalg.solve(variancia[:-1, :-1], desvio))
    return pd.Series({'Qui-quadrado': qui2, 'gl': k - 1, 'p-valor': stats.chi2.sf(qui2, k - 1)})
