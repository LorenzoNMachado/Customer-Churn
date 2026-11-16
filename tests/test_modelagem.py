import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modelagem import SERVICOS, contribuicoes, criar_features, principais_motivos


@pytest.fixture
def clientes():
    base = {s: ['No', 'No', 'No'] for s in SERVICOS}
    base['PhoneService'] = ['Yes', 'Yes', 'No']
    base['StreamingTV'] = ['Yes', 'No', 'No']
    base['InternetService'] = ['Fiber optic', 'DSL', 'No']
    base['TechSupport'] = ['No', 'Yes', 'No']
    return pd.DataFrame(base)


def test_num_servicos(clientes):
    assert criar_features(clientes)['NumServicos'].tolist() == [2, 2, 0]


def test_sem_protecao_so_vale_para_quem_tem_internet(clientes):
    assert criar_features(clientes)['SemProtecao'].tolist() == ['Yes', 'No', 'No']


@pytest.fixture
def modelo_e_dados():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({'idade': rng.normal(40, 10, 200),
                      'plano': rng.choice(['A', 'B', 'C'], 200)})
    y = ((X['plano'] == 'C') | (X['idade'] > 50)).astype(int)
    modelo = make_pipeline(
        ColumnTransformer([('num', StandardScaler(), ['idade']),
                           ('cat', OneHotEncoder(sparse_output=False), ['plano'])]),
        LogisticRegression(),
    ).fit(X, y)
    return modelo, X


def test_contribuicoes_somam_o_log_odds(modelo_e_dados):
    modelo, X = modelo_e_dados
    contrib = contribuicoes(modelo, X, X)
    assert list(contrib.columns) == ['idade', 'plano']
    # log-odds de cada cliente = log-odds médio + soma das contribuições
    log_odds = modelo.decision_function(X)
    assert np.allclose(contrib.sum(axis=1) + log_odds.mean(), log_odds)


def test_principais_motivos(modelo_e_dados):
    modelo, X = modelo_e_dados
    contrib = contribuicoes(modelo, X, X)
    motivos = principais_motivos(contrib, X, n=2)
    cliente_c = X.index[X['plano'] == 'C'][0]
    assert motivos.shape == (len(X), 2)
    assert any(m.startswith('plano = C') for m in motivos.loc[cliente_c])
