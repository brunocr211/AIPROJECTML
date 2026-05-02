# `aipwML` -- Ajuste de regressão e estimativa IPW e AIPW

O pacote [`aipwML`](https://github.com/apoorvalal/aipwML) calcula
efeitos causais utilizando modelos de pscore e de desfecho estimados por meio de regressão linear /
logística, regressão regularizada (ajustada com `glmnet`), florestas aleatórias
(ajustadas com `ranger`) e árvores com reforço de gradiente (ajustadas com
`xgboost`). Ele foi escrito para ser o mais modular possível, de modo que os usuários
possam especificar diferentes opções para os modelos de resultado e de escore de propensão
em `mhatter` e `ehatter`.

## instalação

``` r
library(remotes)
remotes::install_github(“apoorvalal/aipwML”)
```

Este artigo demonstra as funções de estimativa utilizando o conjunto de dados observacionais de Lalonde,
onde os controles experimentais foram substituídos por
unidades de controle do PSID, e os estimadores padrão apresentam um viés significativo
para o efeito experimental de ≈ $1700.

Preparação dos dados
=========
