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
para o efeito experimental de ≈ $1700

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
``` r
psr = c(0.05, 0.95)
ipws2 = c(
  ate_ipw(‘logit’,   w = w, y = y, df = df, fml = fp, psrange = psr),
  ate_ipw(‘lasso’,   w = w, y = y, df = df, fml = fp, psrange = psr),
  ate_ipw(‘ridge’,   w = w, y = y, df = df, fml = fp, psrange = psr),
  ate_ipw(‘rforest’, w = w, y = y, df = df, fml = fp, psrange = psr),
  ate_ipw(‘xgboost’, w = w, y = y, df = df, fml = fp, psrange = psr)
)

ipws2 |> round(3)
```

    ## [1] -1356,4 -1144,8 -1623,8   361,6  2971,2

Melhor.

IPW Aumentado
=============

$$
\\hat{\\tau}\_{\\mathrm{AIPW}}^{\\text{ATE}} =
  \\frac{1}{N} \\sum\_{i=1}^{N}
  \\left\[\\left(
    \\hat{m}\_{1}\\left(x\_{i}\\right)+\\frac{w\_{i}}{\\hat{e}\\left(x\_{i}\\right)}
    \\left(y\_{i}-\\hat{m}\_{1}\\left(x\_{i}\\right)\\right)\\right) -
    \\left(\\hat{m}\_{0}\\left(x\_{i}\\right)+\\frac{1-w\_{i}}{1-\\hat{e}\\left(x\_{i}\\right)}
    \\left(y\_{i}-\\hat{m}\_{0}\\left(x\_{i}\\right)\\right)\\right)\\right\]
$$

Precisamos de escolher como estimar *e* e *m*, pelo que realizamos uma pesquisa exaustiva.
Para cada escolha de modelo de resultados, tento todos os outros ajustadores para
o p-score.

É necessário escolher como estimar *e* e *m*, pelo que realizamos uma pesquisa exaustiva.
Para cada escolha de modelo de resultado, testo todos os outros estimadores para
o pscore.

Modelo de resultado OLS
-----------------

``` r
ols_mean = c(
  ate_aipw(fit_me(meanfn = ‘ols’, pscorefn = ‘logit’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘ols’, pscorefn = ‘lasso’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘ols’, pscorefn = ‘ridge’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘ols’, pscorefn = ‘rforest’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘ols’, pscorefn = ‘xgboost’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr)
)
```
Modelo de resultados Ridge
-------------------

``` r
ridge_mean = c(
  ate_aipw(fit_me(meanfn = ‘ridge’, pscorefn = ‘logit’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘ridge’, pscorefn = ‘lasso’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘ridge’, pscorefn = ‘ridge’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘ridge’, pscorefn = ‘rforest’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘ridge’, pscorefn = ‘xgboost’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr)
)
```

Modelo de resultado LASSO
-------------------

``` r
lasso_mean = c(
  ate_aipw(fit_me(meanfn = ‘lasso’, pscorefn = ‘logit’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘lasso’, pscorefn = ‘lasso’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘lasso’, pscorefn = ‘ridge’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘lasso’, pscorefn = ‘rforest’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘lasso’, pscorefn = ‘xgboost’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr)
)
```

Modelo de resultados da Floresta Aleatória
---------------------------
``` r
rforest_mean = c(
  ate_aipw(fit_me(meanfn = ‘rforest’, pscorefn = ‘logit’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘rforest’, pscorefn = ‘lasso’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘rforest’, pscorefn = ‘ridge’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘rforest’, pscorefn = ‘rforest’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘rforest’, pscorefn = ‘xgboost’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr)
)
```

Modelo de resultado GBM
-----------------

``` r
xgboost_mean = c(
  ate_aipw(fit_me(meanfn = ‘xgboost’, pscorefn = ‘logit’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘xgboost’, pscorefn = ‘lasso’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘xgboost’, pscorefn = ‘ridge’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘xgboost’, pscorefn = ‘rforest’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr),
  ate_aipw(fit_me(meanfn = ‘xgboost’, pscorefn = ‘xgboost’,
    mean_fml = fo, psc_fml = fp, y = y, w = w, df = df), psrange = psr)
)
```
Tabela AIPW
----------

``` r
# empilhar estimativas
aipw_estimates = rbind(ols_mean, lasso_mean, ridge_mean, rforest_mean, xgboost_mean)
colnames(aipw_estimates) = c(‘PS: logit’, ‘PS: lasso’, ‘PS: ridge’,
  'PS: rforest', ‘PS: xgboost’)
rownames(aipw_estimates)= c(‘Resultado: ols’, ‘Resultado: lasso’, ‘Resultado: ridge’,
  ‘Resultado: rforest’, ‘Resultado: xgboost’)
aipw_estimates |> kbl() %>%
  kable_styling()
```

<table class="table" style="margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:right;">
PS: logit
</th>
<th style="text-align:right;">
PS: lasso
</th>
<th style="text-align:right;">
PS: ridge
</th>
<th style="text-align:right;">
PS: rforest
</th>
<th style="text-align:right;">
PS: xgboost
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Outcome: ols
</td>
<td style="text-align:right;">
56.49
</td>
<td style="text-align:right;">
1.199
</td>
<td style="text-align:right;">
-519.4
</td>
<td style="text-align:right;">
439.2
</td>
<td style="text-align:right;">
3227
</td>
</tr>
<tr>
<td style="text-align:left;">
Outcome: lasso
</td>
<td style="text-align:right;">
-236.96
</td>
<td style="text-align:right;">
-74.295
</td>
<td style="text-align:right;">
-909.8
</td>
<td style="text-align:right;">
558.7
</td>
<td style="text-align:right;">
3213
</td>
</tr>
<tr>
<td style="text-align:left;">
Outcome: ridge
</td>
<td style="text-align:right;">
-245.79
</td>
<td style="text-align:right;">
-478.324
</td>
<td style="text-align:right;">
-620.9
</td>
<td style="text-align:right;">
189.8
</td>
<td style="text-align:right;">
3247
</td>
</tr>
<tr>
<td style="text-align:left;">
Outcome: rforest
</td>
<td style="text-align:right;">
-197.46
</td>
<td style="text-align:right;">
110.549
</td>
<td style="text-align:right;">
-355.1
</td>
<td style="text-align:right;">
764.2
</td>
<td style="text-align:right;">
3202
</td>
</tr>
<tr>
<td style="text-align:left;">
Outcome: xgboost
</td>
<td style="text-align:right;">
-320.45
</td>
<td style="text-align:right;">
6.479
</td>
<td style="text-align:right;">
-879.7
</td>
<td style="text-align:right;">
307.1
</td>
<td style="text-align:right;">
3249
</td>
</tr>
</tbody>
</table>

Uma *linha ou coluna* relativamente estável na tabela acima sugere que
acertamos em uma das duas funções de interferência. Neste caso, parece
que a função pscore do GBM produz estimativas estáveis em todas as opções
de modelos de resultado.

Uso manual para inferência, outros estimandos
=========================================

A função `fit_me` ajusta as `m` funções e a função `e` para cada
observação e retorna um conjunto de dados que pode ser usado para
cálculos manuais.

``` r
library(data.table)
fit_mod = fit_me(meanfn = ‘xgboost’, pscorefn = ‘xgboost’,
    mean_fml = fo, psc_fml  = fp, y = y, w = w, df = df)
setDT(fit_mod)
fit_mod |> head()
```

    ##          y     w    m0      m1     eh
    ##      <num> <num> <num>   <num>  <num>
    ## 1:  9930,0     1  9017  9929,8 0,9657
    ## 2:  3595,9     1 11787  3593,9 1,0012
    ## 3: 24909,5     1  3748 24906,3 0,9887
    ## 4:  7506,1     1  8709  2685,8 1,0012
    ## 5:   289,8     1  2021   291,1 0,9965
    ## 6:  4056,5     1  6751  4060,6 0,9889

recortar pontuações P extremas antes do AIPW
---------------- ----------------

``` r
fit_mod |> ate_aipw(c(0.1, 0.9)) |> round(3)
```

    ## [1] 2893

bootstrap
---------

``` r
library(boot); library(MASS)
boot.fn <- function(data, ind){
  d = data[ind, ]
  fit_mod = fit_me(meanfn = ‘lasso’, pscorefn = ‘lasso’,
    mean_fml = fo, psc_fml  = fp, y = y, w = w, df = d) |>
    ate_aipw(c(0.1, 0.9))
}
out = boot(df, boot.fn, R = 100)
out |> print()
```
   ## 
    ## BOOTSTRAP NÃO PARAMÉTRICO ORDINÁRIO
    ## 
    ## 
    ## Chamada:
    ## boot(data = df, statistic = boot.fn, R = 100)
    ## 
    ## 
    ## Estatísticas do Bootstrap:
    ##     original  viés    erro padrão
    ## t1*      342    -101        1023

ATT
---

o ATT subdivide as unidades tratadas e calcula a média entre o
*Y* realizado e o *Y* imputado (0), o que pode ser feito facilmente com nossas
estimativas.

``` r
fit_mod[w == 1, mean(y - m0)] |> round(3)
```

    ## [1] 1796
