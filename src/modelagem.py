"""Variáveis derivadas e explicação das previsões dos modelos de churn."""
import numpy as np
import pandas as pd

SERVICOS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']


def criar_features(df):
    """Acrescenta variáveis derivadas dos serviços contratados."""
    novo = df.copy()
    novo['NumServicos'] = (df[SERVICOS] == 'Yes').sum(axis=1)
    # Tem internet, mas nem segurança online nem suporte técnico
    sem_protecao = (df['InternetService'] != 'No') & (df['OnlineSecurity'] == 'No') & (df['TechSupport'] == 'No')
    novo['SemProtecao'] = np.where(sem_protecao, 'Yes', 'No')
    return novo


def _variavel_de_origem(preprocessador):
    # Nome da coluna original de cada coluna gerada pelo ColumnTransformer
    origem = []
    for nome, transformador, colunas in preprocessador.transformers_:
        if nome == 'remainder':
            continue
        if hasattr(transformador, 'categories_'):
            for coluna, categorias in zip(colunas, transformador.categories_):
                origem += [coluna] * len(categorias)
        else:
            origem += list(colunas)
    return origem


def contribuicoes(pipeline, X_referencia, X):
    """Contribuição de cada variável (em log-odds) para a previsão de cada cliente.

    Para uma regressão logística, coef * (x - média de referência) é o valor
    SHAP exato sob independência das variáveis. As colunas one-hot de uma mesma
    variável são somadas.
    """
    preprocessador, classificador = pipeline[0], pipeline[-1]
    Z = preprocessador.transform(X)
    media = preprocessador.transform(X_referencia).mean(axis=0)
    valores = (Z - media) * classificador.coef_[0]
    tabela = pd.DataFrame(valores, index=X.index, columns=_variavel_de_origem(preprocessador))
    return tabela.T.groupby(level=0, sort=False).sum().T


def principais_motivos(contrib, X, n=3):
    """Os `n` fatores que mais aumentam o risco de cada cliente, como texto."""
    motivos = {}
    for cliente, linha in contrib.iterrows():
        maiores = linha[linha > 0].sort_values(ascending=False).head(n)
        textos = [f'{var} = {X.at[cliente, var]} (+{valor:.2f})' for var, valor in maiores.items()]
        motivos[cliente] = textos + [''] * (n - len(textos))
    return pd.DataFrame.from_dict(motivos, orient='index',
                                  columns=[f'Motivo {i + 1}' for i in range(n)])
