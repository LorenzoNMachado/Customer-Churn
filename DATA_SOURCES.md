# Dados e licenças de artefatos

## Telco Customer Churn

O dataset original é publicado na [página Telco Customer Churn de BlastChar no Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). O arquivo usado nos notebooks é `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

O CSV esteve versionado desde o commit inicial `4767b84` (2026-03-03) até sua remoção da árvore atual nesta branch. O histórico do projeto não registra de qual endereço a cópia original foi obtida, então a origem de aquisição daquela cópia não pode ser afirmada com certeza. A remoção não apaga cópias em commits históricos. O histórico não foi reescrito.

### Proveniência e comparação

O CSV histórico tinha 7.043 registros, 21 colunas e 970.457 bytes. Seu SHA-256
era `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`, igual
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

Os metadados do Kaggle identificam a licença como **“Data files ©
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

Não foi encontrada autorização explícita verificável para redistribuir o CSV.
A licença MIT deste repositório cobre somente código e documentação originais
para os quais o autor pode conceder esses direitos. Ela não se estende ao CSV,
às análises que incorporam dados de origem nem aos arquivos derivados. Por isso,
o CSV foi removido da árvore atual e seu caminho foi adicionado ao `.gitignore`.
As cópias em commits históricos permanecem. Não houve reescrita de histórico.

### Reprodução obtendo os dados na fonte

Para reproduzir as análises, obtenha o arquivo da fonte original e salve-o em
`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`. Use o script do projeto:

```bash
python scripts/download_dataset.py
```

O utilitário usa apenas a biblioteca padrão, baixa da API pública do Kaggle e
verifica o hash, o cabeçalho e a quantidade de registros antes de gravar o
arquivo. Ele não inclui credenciais e se recusa a sobrescrever um arquivo
existente sem `--force`. Se o Kaggle atualizar os dados, a verificação do hash
falhará até que a mudança seja revisada. O workflow de CI tenta baixar e verificar
o arquivo antes de executar o notebook e não o publica como artefato. Se o
endpoint estiver indisponível ou negar acesso, os testes independentes do CSV
continuam, a execução integral do notebook é marcada como ignorada e o workflow
emite um aviso. Erros de integridade falham o workflow. Nesse caso, obtenha o
arquivo manualmente pela página original, sem contornar controles de acesso.
Confira os termos atuais da fonte antes de obter ou reutilizar o dataset.

Os gráficos e métricas são derivados desse dataset e não recebem uma licença
independente para os dados de origem. Os custos da política de retenção são
hipóteses didáticas, não valores observados de uma operadora.
