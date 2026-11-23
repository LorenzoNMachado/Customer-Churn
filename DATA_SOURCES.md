# Dados e licenças de artefatos

## Telco Customer Churn

O arquivo `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` está versionado neste
repositório desde o commit inicial `4767b84` (2026-03-03). O histórico do
projeto não registra de qual endereço a cópia foi obtida, então a fonte de
aquisição original não pode ser afirmada com certeza.

### Proveniência e comparação

O arquivo local tem 7.043 registros, 21 colunas e 970.457 bytes. Seu SHA-256
é `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`, igual
ao arquivo `data/Telco-Customer-Churn.csv` no [repositório IBM de referência](https://github.com/IBM/telco-customer-churn-on-icp4d).
Essa correspondência demonstra igualdade binária com o espelho IBM consultado,
mas, isoladamente, não prova a origem da cópia local nem concede direitos de
redistribuição.

Também baixamos o CSV do [dataset publicado por BlastChar no Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
pela [API pública do Kaggle](https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn)
em 2026-11-23. O arquivo extraído tem SHA-256
`88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a` e
977.501 bytes. Os dois arquivos têm o mesmo cabeçalho, os mesmos 7.043
registros e os mesmos valores em cada célula quando lidos como CSV. Os hashes
binários diferem por causa da representação das linhas (CRLF no arquivo
baixado do Kaggle e LF no arquivo do espelho IBM).

### Licença e decisão

Os metadados do Kaggle atualmente identificam a licença como **“Data files ©
Original Authors”**. A [documentação de datasets do Kaggle](https://www.kaggle.com/docs/datasets)
lista esse rótulo entre as opções de licença e também descreve a licença como
um campo próprio do dataset; não encontrei uma concessão verificável de
redistribuição para este arquivo. O [README do repositório IBM](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/README.md#license)
declara que o *code pattern* é Apache-2.0 e descreve o CSV como dataset usado
no exemplo, mas não declara separadamente que essa licença cobre os dados nem
identifica quem tem autoridade para licenciá-los. A [publicação da IBM sobre
seus dados de churn](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)
descreve amostras de uma empresa fictícia, mas não concede direitos sobre este
CSV. Portanto, não é possível concluir que o CSV esteja sob Apache-2.0 ou que
sua redistribuição pública esteja autorizada.

A licença MIT deste repositório cobre somente código e documentação originais
para os quais o autor pode conceder esses direitos. Ela não se estende ao CSV,
às análises que incorporam dados de origem nem aos arquivos derivados. O CSV
foi mantido nesta versão porque sua retirada exige autorização do proprietário
do repositório. Recomenda-se removê-lo da árvore pública atual após essa
autorização. Removê-lo em um commit futuro não apagará as cópias já presentes
no histórico Git; reescrever esse histórico é uma decisão separada e não foi
feito.

### Reprodução obtendo os dados na fonte

Após a retirada autorizada do arquivo versionado, baixe a versão fixada da
fonte Kaggle para o caminho usado pelos notebooks:

```bash
python scripts/download_dataset.py
```

O utilitário usa apenas a biblioteca padrão, verifica o hash, o cabeçalho e a
quantidade de registros antes de gravar o arquivo e se recusa a sobrescrever
um arquivo existente sem `--force`. Se o Kaggle atualizar os dados, o hash
fixado fará o download falhar até que a mudança seja revisada. Confira os
termos e as permissões atuais da fonte antes de obter ou reutilizar o dataset.

Os gráficos e métricas são derivados desse dataset e não recebem uma licença
independente para os dados de origem. Os custos da política de retenção são
hipóteses didáticas, não valores observados de uma operadora.
