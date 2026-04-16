"""
Pipelines de otimização automática de prompts via DSPy.

Implementa cinco otimizadores:
  1. BootstrapFewShot — seleção automática de exemplos via rejection sampling
  2. MIPROv2 — otimização conjunta de instruções + exemplos via busca bayesiana
  3. GEPA — evolução genética de prompts com reflexão em linguagem natural
  4. KNNFewShot — seleção dinâmica de demos por similaridade (KNN) por exemplo
  5. SIMBA — otimização introspectiva com auto-reflexão sobre erros

Rate limiting: DSPy/LiteLLM fazem retry automático em 429.
Otimizadores rodam com num_threads=1 para serializar chamadas.
Avaliação usa checkpoints para retomar em caso de interrupção.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional

import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2, GEPA, KNNFewShot, SIMBA

from src.dspy_module import (
    FinQAModule,
    finqa_execution_accuracy,
    finqa_execution_accuracy_gepa,
    finqa_to_dspy_examples,
)
from src.metrics import (
    PredictionResult,
    EvaluationReport,
    execution_accuracy,
    program_accuracy,
)


# ── Configuração do LM DSPy ────────────────────────────────
def configure_dspy_lm(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 1024,
):
    """Configura o modelo de linguagem no DSPy."""
    if provider == "openai":
        lm = dspy.LM(
            model=f"openai/{model}",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.environ.get("OPENAI_API_KEY"),
            num_retries=10,
        )
    elif provider == "anthropic":
        lm = dspy.LM(
            model=f"anthropic/{model}",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            num_retries=10,
        )
    else:
        raise ValueError(f"Provider não suportado: {provider}")

    dspy.configure(lm=lm)
    print(f"  DSPy LM configurado: {provider}/{model} (retries=10)")
    return lm


# ── Otimização ──────────────────────────────────────────────
def optimize_bootstrap_few_shot(
    trainset: list[dspy.Example],
    max_bootstrapped_demos: int = 10,
    max_labeled_demos: int = 10,
    max_rounds: int = 3,
) -> FinQAModule:
    """
    Otimiza via BootstrapFewShot.

    Estratégia: rejection sampling — executa o módulo em exemplos de treino,
    seleciona automaticamente os que geram respostas corretas como demos.
    """
    optimizer = BootstrapFewShot(
        metric=finqa_execution_accuracy,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=max_rounds,
    )

    student = FinQAModule()
    optimized = optimizer.compile(student=student, trainset=trainset)
    return optimized


def optimize_miprov2(
    trainset: list[dspy.Example],
    valset: Optional[list[dspy.Example]] = None,
    auto_level: str = "medium",
    max_bootstrapped_demos: int = 10,
    max_labeled_demos: int = 10,
    seed: int = 42,
) -> FinQAModule:
    """
    Otimiza via MIPROv2.

    Estratégia: otimização conjunta de instruções + exemplos demonstrativos
    via busca bayesiana (Optuna).

    Args:
        auto_level: "light", "medium" ou "heavy" — controla número de
                    candidatos e trials automaticamente.
    """
    optimizer = MIPROv2(
        metric=finqa_execution_accuracy,
        auto=auto_level,
        num_threads=1,  # Serializar — essencial para free tier
        seed=seed,
    )

    student = FinQAModule()
    optimized = optimizer.compile(
        student=student,
        trainset=trainset,
        valset=valset,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        seed=seed,
    )
    return optimized


def optimize_gepa(
    trainset: list[dspy.Example],
    valset: Optional[list[dspy.Example]] = None,
    auto_level: str = "medium",
    seed: int = 42,
) -> FinQAModule:
    """
    Otimiza via GEPA.

    Estratégia: evolução genética de prompts com reflexão em linguagem natural.

    Args:
        auto_level: "light", "medium" ou "heavy" — controla intensidade
                    da otimização.
    """
    optimizer = GEPA(
        metric=finqa_execution_accuracy_gepa,
        auto=auto_level,
        num_threads=1,
        seed=seed,
        reflection_lm=dspy.LM("openai/gpt-4o-mini", temperature=1.0, max_tokens=4096),
    )

    student = FinQAModule()
    optimized = optimizer.compile(
        student=student,
        trainset=trainset,
        valset=valset,
    )
    return optimized


def optimize_knn_few_shot(
    trainset: list[dspy.Example],
    k: int = 10,
) -> FinQAModule:
    """
    Otimiza via KNNFewShot.

    Estratégia: para cada exemplo de teste, seleciona os k demos mais
    similares do trainset via KNN com embeddings. Diferente dos outros
    otimizadores que usam demos fixos, aqui os demos são DINÂMICOS —
    cada pergunta recebe demos diferentes baseados em similaridade.

    Usa embeddings da OpenAI (text-embedding-3-small) via LiteLLM.
    """
    embedder = dspy.Embedder("openai/text-embedding-3-small")

    optimizer = KNNFewShot(
        k=k,
        trainset=trainset,
        vectorizer=embedder,
    )

    student = FinQAModule()
    optimized = optimizer.compile(student=student)
    return optimized


def optimize_simba(
    trainset: list[dspy.Example],
    max_steps: int = 8,
    num_candidates: int = 6,
    max_demos: int = 8,
    seed: int = 42,
) -> FinQAModule:
    """
    Otimiza via SIMBA (Stochastic Introspective Mini-Batch Ascent).

    Estratégia: o LLM analisa seus próprios erros em mini-batches e
    gera regras de melhoria iterativamente. Combina auto-reflexão
    com seleção de demos bem-sucedidos.

    Args:
        trainset: Exemplos de treino
        max_steps: Número de passos de otimização
        num_candidates: Candidatos por iteração
        max_demos: Máximo de demos por predictor
        seed: Seed para reprodutibilidade
    """
    optimizer = SIMBA(
        metric=finqa_execution_accuracy,
        num_candidates=num_candidates,
        max_steps=max_steps,
        max_demos=max_demos,
        num_threads=1,  # Serializar — essencial para free tier
    )

    student = FinQAModule()
    optimized = optimizer.compile(
        student=student,
        trainset=trainset,
        seed=seed,
    )
    return optimized


# ── Avaliação de módulo otimizado (com checkpoint) ──────────
def evaluate_dspy_module(
    module: FinQAModule,
    examples: list[FinQAExample],
    pipeline_name: str,
    split: str = "dev",
    verbose: bool = True,
    checkpoint_dir: str = "results",
) -> EvaluationReport:
    """
    Avalia um módulo DSPy em exemplos FinQA com checkpointing.

    Salva progresso a cada exemplo. Se interrompido, retoma de onde parou.
    """
    checkpoint_path = f"{checkpoint_dir}/.checkpoint_{pipeline_name}.json"

    # Carregar checkpoint existente
    completed = {}
    if Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        print(f"  ✓ Checkpoint encontrado: {len(completed)} exemplos já processados")

    report = EvaluationReport(
        pipeline_name=pipeline_name,
        split=split,
        num_examples=len(examples),
    )

    for i, ex in enumerate(examples):
        # Pular exemplos já processados
        if ex.id in completed:
            c = completed[ex.id]
            result = PredictionResult(
                example_id=c["example_id"],
                question=c["question"],
                gold_exe_ans=c["gold_exe_ans"],
                gold_program=c.get("gold_program", ""),
                predicted_answer=c["predicted_answer"],
                predicted_program=c.get("predicted_program", ""),
                exec_acc=c["exec_acc"],
                prog_acc=c.get("prog_acc", False),
                input_tokens=c.get("input_tokens", 0),
                output_tokens=c.get("output_tokens", 0),
                latency_seconds=c.get("latency_seconds", 0),
                raw_response=c.get("raw_response", ""),
            )
            report.predictions.append(result)
            continue

        # Processar exemplo novo
        start = time.time()
        try:
            pred = module(context=ex.context, question=ex.question)
            latency = time.time() - start

            predicted_answer = getattr(pred, "answer", "")
            predicted_reasoning = getattr(pred, "reasoning", "")

            exec_acc = execution_accuracy(predicted_answer, ex.exe_ans)

            result = PredictionResult(
                example_id=ex.id,
                question=ex.question,
                gold_exe_ans=ex.exe_ans,
                gold_program=ex.program,
                predicted_answer=predicted_answer,
                predicted_program="",
                exec_acc=exec_acc,
                prog_acc=False,
                input_tokens=0,
                output_tokens=0,
                latency_seconds=latency,
                raw_response=predicted_reasoning,
            )

        except Exception as e:
            latency = time.time() - start
            if verbose:
                print(f"  [ERROR] Example {ex.id}: {e}")
            result = PredictionResult(
                example_id=ex.id,
                question=ex.question,
                gold_exe_ans=ex.exe_ans,
                gold_program=ex.program,
                predicted_answer="",
                exec_acc=False,
                prog_acc=False,
                latency_seconds=latency,
                raw_response=f"ERROR: {e}",
            )

        report.predictions.append(result)

        # Salvar checkpoint
        completed[ex.id] = {
            "example_id": result.example_id,
            "question": result.question,
            "gold_exe_ans": result.gold_exe_ans,
            "gold_program": result.gold_program,
            "predicted_answer": result.predicted_answer,
            "predicted_program": result.predicted_program,
            "exec_acc": result.exec_acc,
            "prog_acc": result.prog_acc,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_seconds": result.latency_seconds,
            "raw_response": result.raw_response,
        }
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(completed, f, ensure_ascii=False)

        # Log de progresso
        if verbose and (i + 1) % 10 == 0:
            done = len([p for p in report.predictions if ex.id not in completed or True])
            acc_so_far = (
                sum(1 for p in report.predictions if p.exec_acc)
                / len(report.predictions) * 100
            )
            print(f"  [{len(report.predictions)}/{len(examples)}] Exec Acc: {acc_so_far:.1f}%")

    # Limpar checkpoint ao concluir
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print("  ✓ Checkpoint removido (execução completa)")

    return report


# ── Salvar / Carregar módulo otimizado ──────────────────────
def save_optimized_module(module: FinQAModule, path: str):
    """Salva módulo otimizado em disco."""
    if not path.endswith((".json", ".pkl")):
        path = path + ".json"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    module.save(path)
    print(f"Módulo salvo em: {path}")


def load_optimized_module(path: str) -> FinQAModule:
    """Carrega módulo otimizado de disco."""
    if not path.endswith((".json", ".pkl")):
        path = path + ".json"
    module = FinQAModule()
    module.load(path)
    print(f"Módulo carregado de: {path}")
    return module