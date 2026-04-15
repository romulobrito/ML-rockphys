# Plano metodologico: melhorias nos scripts ML e sincronizacao com o relatorio LaTeX

Este documento organiza, por fases e dependencias, a implementacao das melhorias discutidas (itens 2.1 a 2.14), o protocolo de regeneracao de resultados e as alteracoes esperadas em `relatorio_porosidade_ml.tex` e em `figs/`. Alinha-se a `AI_AGENT_RULES.md` (metodologia clara, execucao ponta a ponta com dados reais, atualizacao de figuras no relatorio, registo em `agent.log`).

---

## 1. Objetivo e principios

**Objetivo:** fortalecer a defesa cientifica do estudo (protocolo de avaliacao, diagnostico de colinearidade, redundancia GR–Vc, reprodutibilidade) sem perder a linha narrativa do relatorio.

**Principios:**

1. **Separar protocolo de conclusao:** cada metrica reportada deve corresponder ao mesmo universo de dados (treino, teste ou ambos) explicitamente declarado.
2. **Intra-poco versus inter-poco:** nao misturar no mesmo paragrafo conclusoes de generalizacao interna ao poco com transferencia entre pocos.
3. **Mudanca minima necessaria no texto:** o LaTeX deve refletir o codigo; evitar reescrever secoes inteiras se apenas numeros e uma frase de metodo mudarem.
4. **Baseline reprodutivel:** antes de alterar codigo, fixar uma etiqueta Git (ou commit) do estado atual para comparacao regressiva se necessario.

---

## 2. Inventario: ligacao codigo — saidas — LaTeX

| Origem (script) | Saidas relevantes (padrao atual) | Figuras em `docs/figs/` citadas no `.tex` |
|-----------------|----------------------------------|-------------------------------------------|
| `porosity_baseline.py` | `outputs/*.csv`, PNG em `outputs/`, `outputs/models/*.joblib` | `porosity_F03-4_*.png`, `porosity_F06-1_com_Vp.png` (copiar para `figs/` apos geracao) |
| `feature_relationship_study.py` | `outputs/feature_study_<stem>/` | `study_F03-4_corr_pearson.png`, `study_F03-4_corr_spearman.png`, `study_F03-4_perm_importance.png`, `study_F03-4_shap_summary.png` |
| `gr_vc_rank_dependence.py` | `outputs/gr_vc_rank_<stem>/` | `gr_vc_F03-4_ranks_scatter.png`, `gr_vc_F03-4_rank_method_sensitivity.png` |
| `exploratory_analysis.py` | `eda_output/` | hoje nao referenciadas diretamente no relatorio principal; util para coerencia metodologica |

**Convencao recomendada:** o relatorio consome apenas ficheiros sob `ML-prediction/docs/figs/` com nomes estaveis; apos cada rodada de scripts, copiar ou exportar PNG finais para `figs/` (ou o orquestrador faz isso).

---

## 3. Fases de trabalho (ordem logica e dependencias)

### Fase A — Fundacoes compartilhadas (bloqueia varias fases seguintes)

| ID | Entrega | Ficheiros | Criterio de aceitacao |
|----|---------|-----------|------------------------|
| A1 | **Metadados comuns de experimento** | Novo modulo leve, por exemplo `ML-prediction/scripts/run_metadata.py`, ou funcao partilhada | Cada run grava JSON ou CSV com: timestamp ISO, seed, caminho do dataset, n linhas, lista de features, `test_fraction` ou equivalente, modo (`within_well` / `cross_well`), versao Git (opcional via subprocess) |
| A2 | **Pipeline GBDT sem escalonamento por defeito** | `porosity_baseline.py`: `build_pipeline(..., use_scaler=False)` ou flag CLI `--use-scaler` | Teste: mesmo `random_state` e dados, metricas de teste podem mudar ligeiramente face ao pipeline antigo; documentar no `run_meta` se scaler foi usado |
| A3 | **Refactor minimo** | `feature_relationship_study.py` importa o mesmo `build_pipeline` atualizado | Permutacao e SHAP usam o mesmo estimador que o baseline |

**LaTeX apos A:** atualizar paragrafo de metodologia onde se descreve pre-processamento (mencionar que o GBDT opera nas features originais, com opcao de escalonamento apenas para comparacao linear se existir).

---

### Fase B — Prioridade metodologica (ordem sugerida pelo utilizador)

| ID | Entrega | Ficheiros | Detalhe metodologico |
|----|---------|-----------|----------------------|
| B1 | **MI apenas no treino** (+ opcional coluna ou ficheiro MI no teste para estabilidade) | `feature_relationship_study.py` | Calcular `mutual_info_regression` em `x_train, y_train` apos `depth_block_split`; guardar `mutual_info_porosity_train.csv`; opcional `..._test.csv` para comparacao |
| B2 | **Modo inter-poco / leave-one-well-out (LOO por poco)** | `porosity_baseline.py` + possivel extensao de CLI para lista de ficheiros | Definir contrato: treino = uniao de N-1 pocos, teste = poco retido; metricas por fold e agregadas (media, desvio); exigir esquema 7logs alinhado |
| B3 | **Sensibilidade a multiplos cortes em profundidade** | `porosity_baseline.py` | Para modo within-well: loop sobre `test_fraction` em grelha (ex.: 0.15, 0.2, 0.25) ou cortes por quantis de profundidade; exportar tabela `r2_depth_sensitivity.csv` |
| B4 | **Quantificacao monotonica GR → Vc** | `gr_vc_rank_dependence.py` | IsotonicRegression de `Vc` em `GR` (ou ranks), RMSE, R2 ou variancia explicada; exportar linha em CSV e, se util, PNG de ajuste |
| B5 | **Hiperparametros: sensibilidade leve** | `porosity_baseline.py` | Grelha pequena (ex.: `max_depth`, `learning_rate`, `max_iter`) no mesmo split ou num subconjunto de folds; tabela resumo; nao substituir o relatorio principal por uma busca pesada |

**LaTeX apos B:** novas tabelas ou frases em Resultados/Discussao: protocolo LOO, robustez ao corte, metricas monotonicas, nota de que MI e calculada no treino; atualizar qualquer numero citado inline se mudar.

---

### Fase C — Diagnosticos e rigor de explicacao

| ID | Entrega | Ficheiros |
|----|---------|-----------|
| C1 | **Numero de condicao, autovalores, PCA** | `feature_relationship_study.py`: matriz de correlacao ou covariancia padronizada; `numpy.linalg.cond`; PCA com `explained_variance_ratio_`; CSV dedicado |
| C2 | **SHAP no mesmo espaco do modelo** | `feature_relationship_study.py`: passar a matriz transformada ao `summary_plot` com `feature_names` iguais, ou remover pre-processamento nao linear e usar so features brutas (coerente com A2) |
| C3 | **Coluna interpretativa na permutacao** | `permutation_importance_test.csv`: ex. `likely_redundant_or_noise` = `importance_mean <= 0` |
| C4 | **Nota interpretativa em correlações parciais** | `gr_vc_rank_dependence.py`: cabecalho comentado no CSV ou coluna `interpretation_note` constante; docstring no codigo com limite linear |

**LaTeX apos C:** Metodologia/VIF/PCA: uma frase sobre condicao e variancia explicada acumulada; SHAP: alinhar texto ao espaco usado; paragrafo curto sobre correlacao parcial linear.

---

### Fase D — Portabilidade e orquestracao

| ID | Entrega | Ficheiros |
|----|---------|-----------|
| D1 | **CLI `--data-dir` e `--out-dir`** | `exploratory_analysis.py` |
| D2 | **Duas correlacoes explicitas no EDA** | Ja existe Spearman e Pearson em `feature_relationship_study.py`; no `exploratory_analysis.py` garantir parametrizacao e dois ficheiros por dataset se ainda nao estiver claro |
| D3 | **Script orquestrador** | Novo `ML-prediction/scripts/run_report_pipeline.py` (nome a acordar): ordem EDA → baseline(s) → feature study → gr_vc → copia para `docs/figs/` → opcional `pdflatex` |

**LaTeX apos D:** opcionalmente mencionar comando unico no Anexo metodologico (sem poluir o corpo; ver regra 10 do `AI_AGENT_RULES.md` sobre caminhos no texto principal).

---

## 4. Protocolo de regeneracao e verificacao (apos implementacao)

1. **Smoke:** importar modulos, `--help` de cada CLI, uma execucao rapida com subamostra se existir flag (ou ficheiro pequeno de teste).
2. **Ponta a ponta (Regra 5):** com dados em `ML-prediction/data/`, correr o orquestrador ou a sequencia manual; registar comandos e hashes ou timestamps em `agent.log`.
3. **Diff de CSV:** comparar chaves (`R2`, importancias, MI) entre commit antigo e novo para identificar mudancas so por protocolo.
4. **Figuras:** substituir apenas PNG referenciados no `.tex`; verificar resolucao e legendas.
5. **`pdflatex`** duas vezes em `ML-prediction/docs/`; corrigir avisos críticos.
6. **Regra 6:** no relatorio ou no `agent.log`, nota critica: o que ficou mais forte, o que continua fragil (ex.: poucos pocos para LOO).

---

## 5. Sincronizacao com `relatorio_porosidade_ml.tex` (checklist)

- [ ] **Resumo:** protocolo (within-well / LOO) e uma linha sobre robustez ao corte se aplicavel.
- [ ] **Metodologia:** split por profundidade vs LOO; MI no treino; ausencia de scaler no GBDT; correlacao parcial linear; monotonia GR–Vc.
- [ ] **Resultados:** tabelas e numeros alinhados aos novos CSV; novas tabelas para sensibilidade de hiperparametros e multi-corte se entrarem no argumento.
- [ ] **Figuras:** `\includegraphics{...}` na lista da Secao 2 deste plano; copiar PNG para `figs/`.
- [ ] **Discussao e limitacoes:** LOO com poucos pocos; dependencia do corte em within-well; interpretacao de MI condicional ao split.
- [ ] **Referencias internas:** labels de tabelas/figuras coerentes apos insercoes.

---

## 6. Riscos e mitigacao

| Risco | Mitigacao |
|-------|-----------|
| Poucos ficheiros 7logs para LOO estavel | Reportar intervalo ou desvio entre folds; nao sobrevender "generalizacao entre pocos" |
| Metricas mudam e enfraquecem narrativa | Manter tabela antiga como "versao anterior" apenas se necessario para tese; preferir texto honesto |
| Repositório Git com dados sensiveis | Reavaliar `.gitignore` e LFS em passo separado (fora do escopo deste plano tecnico) |
| Tempo de SHAP/perm com muitos folds | Reduzir `n_repeats` ou amostra apenas para sensibilidade, declarando no meta |

---

## 7. Ordem de implementacao recomendada (para o agente ou desenvolvedor)

1. A1, A2, A3 (fundacao + pipeline)
2. B1 (MI treino)
3. B2 (LOO / cross-well) — pode exigir mais desenho de CLI
4. B3 (multi-corte)
5. Ajustar `feature_relationship_study` para C2, C3, C1 (SHAP + CSV perm + PCA/cond)
6. B4 (monotonia GR–Vc) + C4 (notas parciais)
7. B5 (hiperparametros leves)
8. D1, D2, D3 (EDA CLI + orquestrador)
9. Regeneracao completa + LaTeX + `agent.log`

---

## 8. Estado deste documento

Trata-se de um **plano de execucao**; a implementacao deve seguir commits incrementais (por fase ou por ID), com mensagens claras, e atualizacao do `agent.log` apos cada bloco substantivo concluido e testado.
