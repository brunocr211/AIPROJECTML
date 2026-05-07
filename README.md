# `aipwML` -- Ajuste de regressão e estimativa IPW e AIPW

O pacote [`aipwML`](https://github.com/apoorvalal/aipwML) calcula
efeitos causais utilizando modelos de pscore e de desfecho estimados por meio de regressão linear /
logística, regressão regularizada (ajustada com `glmnet`), florestas aleatórias
(ajustadas com `ranger`) e árvores com reforço de gradiente (ajustadas com
`xgboost`). Ele foi escrito para ser o mais modular possível, de modo que os usuários
possam especificar diferentes opções para os modelos de resultado e de escore de propensão
em `mhatter` e `ehatter`

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
``` r
data(lalonde.psid); df = lalonde.psid
y = ‘re78’; w = ‘treat’
x = setdiff(colnames(df), c(y, w))

# fórmula do modelo de resultado
fo = re78 ~ (idade + escolaridade + negro + hispânico + casado + sem diploma +
    re74 + re75 + u74 + u75)
# fórmula do pscore
fp = treat ~ (idade + escolaridade + negro + hispânico + casado + sem diploma +
    re74 + re75 + u74 + u75)
```

Temos os dados
{*y*<sub>*i*</sub>, *w*<sub>*i*</sub>, *x*<sub>*i*</sub>}<sub>*i* = 1</sub><sup>*N*</sup> ∈ ℝ × {0, 1} × ℝ<sup>*k*</sup>.
Sob suposições de seleção em observáveis, podemos calcular o ATE
imputando o resultado potencial ausente.

Ajuste de regressão
=====================

$$
\\hat{\\tau}\_{\\text{reg}}^{\\text{ATE}}  = \\frac{1}{N} \\sum\_{i=1}^N (
    \\hat{\\mu}\_1 (x\_i) - \\hat{\\mu}\_0 (x\_i)
)
$$

``` r
regadjusts = c(
  ate_reg(‘ols’,      w = w, y = y, df = df, fml = fo),
  ate_reg(‘lasso’,    w = w, y = y, df = df, fml = fo),
  ate_reg(‘ridge’,    w = w, y = y, df = df, fml = fo),
  ate_reg(‘rforest’,  w = w, y = y, df = df, fml = fo),
  ate_reg(‘xgboost’,  w = w, y = y, df = df, fml = fo)
)
regadjusts |> round(3)
```

    ## [1]  -8746 -13824 -12260  -9677 -11623

muito ruim.

Ponderação de Propensão Inversa (IPW)
==================================

$$
\\hat{\\tau}\_{\\text{ipw}}^{\\text{ATE}} = \\frac{1}{N} \\sum\_{i=1}^N
\\frac{y\_i (w\_i - \\hat{e}(x\_i)) }{\\hat{e}(x\_i) (1 - \\hat{e}(x\_i)) }
$$

``` r
ipws = c(
  ate_ipw(‘logit’,   w = w, y = y, df = df, fml = fp),
  ate_ipw(‘lasso’,   w = w, y = y, df = df, fml = fp),
  ate_ipw(‘ridge’,   w = w, y = y, df = df, fml = fp),
  ate_ipw(‘rforest’, w = w, y = y, df = df, fml = fp),
  ate_ipw(‘xgboost’, w = w, y = y, df = df, fml = fp)
)

ipws |> round(3)
```

    ## [1] -10454 -15260 -17703 -19568 -19442

Ainda está bem ruim. Agora, corte os pscores extremos.
