# Política de segurança

## Versões mantidas

Somente a branch `main` recebe manutenção. Não há versões publicadas ou
branches de release com suporte de segurança separado. O ambiente validado
pelo CI é Python 3.12.

## Como reportar uma vulnerabilidade

Use **Security → Advisories → Report a vulnerability** na página deste
repositório para enviar um relato privado. Não abra uma issue pública nem
publique detalhes de exploração antes de conversar em privado com o mantenedor.
Inclua o arquivo e o contexto afetados, o impacto observado e passos mínimos
para reproduzir, sem anexar dados de clientes reais.

O projeto é mantido como portfólio e pesquisa; não há prazo de resposta ou
garantia de correção. Issues públicas podem ser usadas para problemas que não
sejam vulnerabilidades.

## Escopo e limites

Relatos relevantes incluem credenciais versionadas, execução de conteúdo não
confiável, falhas em dependências ou workflows e exposição indevida de dados.
O CSV de Telco Customer Churn é um artefato de terceiros e não é coberto pela
licença MIT do código; consulte [`DATA_SOURCES.md`](DATA_SOURCES.md). Os
resultados dos scanners não constituem garantia de segurança.
