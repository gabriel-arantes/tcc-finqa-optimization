# TCC — Otimização Automática de Prompts vs. Engenharia de Prompts Manual

Estudo comparativo utilizando o framework DSPy sobre o benchmark FinQA.

**Autor:** Gabriel Soares Arantes (ITA/CEDS)  
**Orientadora:** Profa. Dra. Lilian Berton

## Questão de Pesquisa

Em que medida a **otimização automática de prompts** (via DSPy) supera a **engenharia de prompts manual** na tarefa de raciocínio numérico sobre relatórios financeiros? Qual otimizador apresenta o melhor equilíbrio entre acurácia, custo computacional e reprodutibilidade?

## Estrutura do Projeto

```
tcc_finqa/
├── configs/
│   └── config.yaml          # Configuração centralizada
├── scripts/
│   ├── run_baseline.py       # Executa baseline manual
│   ├── run_optimizers.py     # Executa otimizadores DSPy
│   └── run_all.py            # Executa todos os experimentos
├── src/
│   ├── data_loader.py        # Carregamento do FinQA
│   ├── metrics.py            # Execution/program accuracy
│   ├── llm_client.py         # Cliente LLM com tracking de tokens
│   ├── baseline_manual.py    # Pipeline baseline (few-shot CoT)
│   ├── dspy_module.py        # Módulo DSPy (assinatura + metric)
│   ├── dspy_pipelines.py     # Otimizadores: Bootstrap, MIPROv2, GEPA
│   └── results_collector.py  # Persistência e exportação
├── results/                  # Resultados gerados (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

## Pipelines Implementados

| Pipeline | Tipo | Descrição |
|----------|------|-----------|
| `manual_baseline` | Engenharia de prompts manual | Few-shot CoT com 5 exemplos selecionados manualmente |
| `dspy_bootstrap_few_shot` | Otimização automática | Seleção de demos via rejection sampling |
| `dspy_miprov2` | Otimização automática | Otimização conjunta instrução+demos via busca bayesiana |
| `dspy_gepa` | Otimização automática | Evolução genética com reflexão em linguagem natural |

## Setup

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Configurar API key
cp .env.example .env
# Editar .env com sua chave OpenAI ou Anthropic

# 3. Exportar variável
export OPENAI_API_KEY=sk-...
```

## Execução

```bash
# Teste rápido (20 exemplos)
python scripts/run_all.py --eval_subset 20 --train_subset 100

# Apenas baseline
python scripts/run_baseline.py --subset 50

# Apenas um otimizador
python scripts/run_optimizers.py --optimizer bootstrap --eval_subset 50

# Experimento completo (dev set inteiro, 3 runs)
python scripts/run_all.py --num_runs 3

# Avaliação final no test set
python scripts/run_all.py --eval_split test --num_runs 3

# Pular otimizadores específicos
python scripts/run_all.py --skip gepa miprov2
```

## Métricas

- **Execution accuracy** (primária): compara resultado numérico com `exe_ans` do FinQA
- **Program accuracy** (secundária): compara programa de raciocínio gerado
- **Custo em tokens**: input + output tokens por exemplo
- **Latência**: tempo de resposta por exemplo

## Dataset FinQA

- ~8.300 pares pergunta-resposta sobre relatórios financeiros do S&P 500
- Contexto pré-extraído (tabelas + texto) — **não há etapa de retrieval**
- Tarefa: raciocínio numérico multi-etapas
- Operações: divide (69%), subtract (41%), add (15%), multiply (8%), ...

## Terminologia (fixa, sem sinônimos)

- **engenharia de prompts manual** → baseline
- **otimização automática de prompts** → abordagem DSPy
- **otimizadores** → BootstrapFewShot, MIPROv2, GEPA
