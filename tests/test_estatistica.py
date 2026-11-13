import math

import pandas as pd
import pytest

from estatistica import (calcular_classes, classificar, criar_tabela_frequencia,
                         criar_tabela_frequencia_categorica, medidas_resumo, moda_czuber,
                         resumo_categoricas, taxa_churn, teste_mann_whitney,
                         teste_qui_quadrado)


@pytest.fixture
def df():
    return pd.DataFrame({
        'tenure': [0, 1, 2, 5, 6, 12, 30, 72, 72, 72],
        'valor': [10.5, 20.0, 20.0, 35.2, 50.0, 61.1, 70.0, 80.3, 99.9, None],
        'plano': ['A', 'A', 'B', 'A', 'C', 'B', 'A', 'C', 'C', 'C'],
        'Churn': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'No', 'No'],
    })


@pytest.mark.parametrize('largura', [None, 6])
def test_tabela_inclui_valor_maximo(df, largura):
    tabela = criar_tabela_frequencia(df, 'tenure', largura)
    assert tabela['Frequência Absoluta (fi)'].sum() == len(df)
    assert tabela['Freq. Acumulada (Fi)'].iloc[-1] == len(df)
    assert tabela['Fr Acumulada (%)'].iloc[-1] == pytest.approx(100)


def test_tabela_ignora_valores_ausentes(df):
    tabela = criar_tabela_frequencia(df, 'valor')
    assert tabela['Frequência Absoluta (fi)'].sum() == df['valor'].notna().sum()


def test_classes_por_sturges(df):
    k = math.ceil(1 + 3.322 * math.log10(len(df)))
    assert len(calcular_classes(df['tenure'])) == k + 1


def test_classes_de_largura_fixa():
    bins = calcular_classes(pd.Series(range(73)), largura=6)
    assert bins[:-1] == list(range(0, 72, 6))
    assert 72 < bins[-1] < 72.001
    assert len(bins) == 13


def test_rotulos_inteiros_sem_decimais():
    rotulos = classificar(pd.Series([0, 5, 6, 12]), largura=6).cat.categories
    assert rotulos[0] == '0 |--- 6'


def test_tabela_categorica(df):
    tabela = criar_tabela_frequencia_categorica(df, 'plano')
    assert tabela.loc['C', 'Frequência Absoluta (fi)'] == 4
    assert tabela['Fr (%)'].sum() == pytest.approx(100)


def test_resumo_categoricas(df):
    tabela = resumo_categoricas(df, ['plano', 'Churn'])
    assert tabela.loc[('plano', 'A'), 'Frequência Absoluta (fi)'] == 4
    assert tabela.loc[('Churn', 'Yes'), 'Fr (%)'] == pytest.approx(40)


def test_moda_czuber():
    # Classes [0,10) [10,20) [20,30): fi = 2, 5, 3 -> Mo = 10 + 3 / (3 + 2) * 10 = 16
    dados = pd.Series([0, 2] + [11] * 5 + [21, 22, 29.999])
    classe, moda = moda_czuber(dados, largura=10)
    assert classe.left == 10
    assert moda == pytest.approx(16)


def test_moda_czuber_classe_modal_na_ponta():
    dados = pd.Series([1] * 5 + [11, 12])
    classe, moda = moda_czuber(dados, largura=10)
    assert classe.left == 1
    assert classe.left <= moda < classe.right


def test_medidas_resumo(df):
    medidas = medidas_resumo(df['tenure'])
    assert medidas['Mediana'] == pytest.approx(9)
    assert medidas['Amplitude Interquartil'] == medidas['Q3'] - medidas['Q1']


def test_taxa_churn(df):
    tabela = taxa_churn(df, 'plano')
    assert tabela.loc['A', 'Churn (%)'] == pytest.approx(50)
    assert tabela['Clientes'].sum() == len(df)


def test_qui_quadrado_independencia_perfeita():
    df = pd.DataFrame({'x': ['a', 'a', 'b', 'b'] * 25, 'Churn': ['Yes', 'No'] * 50})
    resultado = teste_qui_quadrado(df, 'x')
    assert resultado['V de Cramér'] == pytest.approx(0)
    assert resultado['p-valor'] > 0.05


def test_mann_whitney_sinal(df):
    # Quem saiu tem tenure menor -> rank-biserial negativo
    assert teste_mann_whitney(df, 'tenure')['Rank-biserial'] < 0
