# Regras para o agente de IA (Cursor) neste projeto

Este arquivo define obrigações do assistente ao trabalhar em `lenep-rockphys` (incluindo `ML-prediction/`).

---

## Regra 1 — Histórico obrigatório em `agent.log`

**Sempre** acrescentar ao arquivo `agent.log` (na raiz do repositório) um bloco **por conversa ou por solicitação relevante**, contendo:

1. **Contexto / pedido** — resumo do que o usuário pediu (uma ou duas frases).
2. **Linha de raciocínio** — cadeia de pensamento **resumida** (passos principais, premissas, alternativas consideradas quando forem importantes).
3. **Caminho ou solução** — arquivos criados ou alterados (caminhos relativos ou absolutos), comandos relevantes, decisão final, limitações ou trabalho futuro.

**Formato sugerido** (copiar e adaptar):

```text
--------------------------------------------------------------------------------
[AAAA-MM-DD] <título curto>
--------------------------------------------------------------------------------
Pedido:
- ...

Raciocínio (resumo):
- ...

Solução / artefatos:
- caminho: ...
- ...

--------------------------------------------------------------------------------
```

**Quando registrar:** após concluir tarefa com alterações relevantes, decisões de modelagem, ou respostas que fixem premissas do projeto; não é necessário registrar mensagens triviais do chat (por exemplo, apenas “ok” ou “obrigado”).

**Idioma do `agent.log`:** português do Brasil, com acentuação correta, alinhado às Regras 2 e 3.

---

## Regra 2 — Idioma e acentuação em documentação

Em arquivos **Markdown (`.md`)** e **LaTeX (`.tex`)**, escrever **sempre em português do Brasil**, com **acentuação e pontuação corretas** (ç, til, acentos agudos/graves, hífen onde couber).

---

## Regra 3 — Estilo de escrita em `.md` e `.tex` (anti-vício de LLM)

Nos documentos Markdown e LaTeX:

- **Evitar** vícios típicos de texto gerado por modelo de linguagem (frases genéricas de abertura ou fecho, lista excessiva de adjetivos, “engajamento” vazio, repetição de estruturas paralelas sem necessidade).
- **Priorizar** linguagem próxima à humana, clara e direta, com **boas práticas de escrita científica**: precisão terminológica, parágrafos bem construídos, distinção entre fato e interpretação, e citação de limitações quando aplicável.

---

## Regra 4 — Boas práticas de código, metodologia e desenvolvimento de software

**Sempre** adotar boas práticas de:

- **Codificação:** legibilidade, nomes consistentes, tipagem quando a linguagem permitir, tratamento explícito de erros onde fizer sentido, evitar duplicação desnecessária.
- **Metodologia:** hipóteses e passos claros em análises; validação coerente com o tipo de dado (por exemplo, separação por profundidade ou por poço em séries de perfilagem).
- **Desenvolvimento de software:** mudanças focadas no pedido, controle de dependências (`requirements.txt` quando houver novas bibliotecas), e organização razoável de scripts e dados.

---

## Regra 5 — Testes obrigatórios (smoke e ponta a ponta)

Para código novo ou alterado de forma relevante:

1. **Smoke tests** — execução rápida que verifica importações, caminhos, CLI básico ou funções principais sem falha (pode ser em ambiente isolado ou “sandbox”, quando existir).
2. **Uma rodada ponta a ponta** — executar o fluxo completo com **dados reais** do projeto (por exemplo, arquivos em `ML-prediction/data/`), registrando resultado ou falha no `agent.log` quando a alteração for substantiva.

---

## Regra 6 — Avaliação crítica dos resultados

**Sempre** avaliar de forma **crítica** os resultados (métricas, gráficos, suposições do modelo, vazamento de dados, generalização) e **informar o usuário** de forma explícita: o que é forte, o que é frágil, o que não pode ser concluído sem mais dados ou experimentos.

---

## Regra 7 — Confirmação antes de implementar após discussão

Depois de discutir **ideias, abordagens e possíveis caminhos**, **perguntar ao usuário** se deve **seguir com a implementação** antes de iniciar mudanças grandes de código ou de estrutura — salvo quando o próprio pedido já for inequívoco (“implemente agora X”).

---

## Regra 8 — Emojis no código

**Nunca** utilizar emojis em **código-fonte** (comentários, strings, documentação embutida em arquivos `.py`, `.js`, etc.). Em documentação `.md` fora de blocos de código, o uso de emoji só deve ocorrer se o **usuário** solicitar expressamente.

---

## Regra 9 — Idioma no chat (Cursor)

Nas respostas da conversa com o **usuário** no chat do Cursor, escrever **sempre em português do Brasil** (vocabulário, registro e acentuação de PT-BR), salvo se o usuário pedir explicitamente outro idioma para aquela mensagem ou trecho (por exemplo, citação em inglês).

---

## Regra 10 — Relatório científico em LaTeX (`ML-prediction/docs`)

Quando o trabalho em `ML-prediction/` produzir figuras ou resultados que devam ser documentados para artigo ou relatório técnico:

1. **Criar ou manter** um arquivo **`.tex`** em `ML-prediction/docs/` (por exemplo, relatório temático alinhado ao experimento).
2. **Armazenar as figuras** utilizadas pelo `.tex` na subpasta **`ML-prediction/docs/figs/`** (nomes estáveis, sem caracteres problemáticos para LaTeX quando possível).
3. O conteúdo do `.tex` deve seguir **estrutura e tom de relatório ou artigo científico** em **português do Brasil**, com acentuação correta, incluindo de forma explícita, quando aplicável:
   - motivação e contexto;
   - perguntas ou hipóteses científicas;
   - fundamentação teórica (síntese objetiva);
   - metodologia (dados, protocolo, métricas, software);
   - premissas e limitações;
   - resultados (tabelas, figuras com legenda interpretável);
   - discussão crítica dos resultados (forças, fragilidades, alternativas);
   - conclusões e referências (a completar ou vinculadas à bibliografia do projeto);
   - conclusões fundamentadas nos resultados e nas análises, mesmo quando estas forem parciais;
   - texto analítico: discutir resultados e extrair conclusões a partir de figuras e de análises numéricas, sempre que couber.
4. O relatório confeccionado deve ser redigido como se fosse pelo usuário.
5. É proibido o uso de linguagem característica de modelos de LLM, incluindo vícios de linguagem ou outros traços típicos desses modelos. O texto deve aproximar-se ao produzido por um autor humano.
6. Referências a figuras, tabelas, seções, subseções, equações etc. devem aparecer de forma coerente no corpo do texto (não deixar figuras ou equações órfãs de menção).
7. Utilizar fórmulas matemáticas quando agregarem valor à narrativa e à explicação.
8. Caminhos de arquivos de figuras não precisam constar no relatório. Detalhes de implementação — referências a pastas ou a arquivos `.py`, `.md`, `.xlsx`, etc. — não devem fazer parte do texto principal.
9. Incluir detalhes metodológicos relevantes para entender e justificar a abordagem; podem ser usados fluxogramas ou outras figuras de apoio.

Atualizar **`figs/`** quando novas saídas gráficas substituírem versões anteriores relevantes para o texto, e registrar mudanças relevantes no `agent.log`.

---

## Regra 11 — LaTeX: siglas, fluxogramas e texto autocontido

Nos relatórios em `.tex` (por exemplo em `ML-prediction/docs/`):

1. **Siglas e símbolos na primeira ocorrência**  
   Na **primeira vez** que uma sigla ou símbolo não trivial aparecer no **corpo** do texto (resumo, introdução ou metodologia, conforme o fluxo narrativo), deve constar a **forma expandida ou a definição**, como em artigos científicos. Exemplos de redação aceitável:
   - velocidade da onda de compressão P (\(V_p\));
   - SHAP (*SHapley Additive exPlanations*, valores de Shapley para explicação aditiva do modelo);
   - GBDT (*Gradient Boosting Decision Trees*, modelo de conjunto de árvores de decisão com reforço por gradiente em histogramas, quando for o caso).  
   Depois da primeira definição, pode usar-se apenas a sigla ou o símbolo, mantendo consistência.

2. **Fluxogramas e fluxos explicados no texto**  
   Toda figura de **fluxo** ou **fluxograma** deve ser **acompanhada no corpo do texto** por uma explicação **explícita**: sequência de passos, entradas e saídas, pontos de decisão relevantes e relação com o restante da metodologia. O leitor não deve depender só da figura para compreender o procedimento.

3. **Texto autocontido**  
   O documento deve ser **autocontido** em geral: problema, dados em termos conceituais (sem depender de caminhos de arquivo), método, resultados e conclusões devem ser inteligíveis **a partir do PDF** e das referências bibliográficas citadas, **sem** obrigar o leitor a consultar repositório, *issues* ou conversas do chat. Remissões a trabalhos externos devem resumir o que importa para o argumento.

4. **Linguagem neutra (evitar superlativos e adjetivos exagerados)**  
   No corpo do `.tex`, quando já existem **números, tabelas ou figuras**, evitar superlativos e adjetivos carregados (por exemplo, ``excelente'', ``notável'', ``muito alto'', ``substancialmente'', ``imensamente'') que apenas intensificam a leitura sem acrescentar informação além da quantificação. Preferir **intervalos**, **mínimo/máximo**, **diferenças** e verbos factuais (``situa-se entre'', ``o menor valor ocorre em'', ``a folga é da ordem de''). Reservar adjetivos interpretativos a conclusões explicitamente condicionadas ao protocolo e à incerteza reportada.

---

## Referências rápidas

| Item           | Localização                                      |
|----------------|--------------------------------------------------|
| Memória        | `agent.log`                                      |
| Ambiente ML    | `ML-prediction/.venv`, `requirements.txt`        |
| Scripts ML     | `ML-prediction/scripts/`                         |
| LaTeX / figuras | `ML-prediction/docs/*.tex`, `ML-prediction/docs/figs/` |
| Regras do agente | `AI_AGENT_RULES.md` (este arquivo)             |
