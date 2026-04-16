"""
Pipeline Baseline — Engenharia de Prompts Manual.

Implementa few-shot chain-of-thought prompting convencional
para raciocínio numérico sobre o dataset FinQA.

Este é o baseline contra o qual os otimizadores DSPy serão comparados.
"""

import re
from typing import Optional

from src.data_loader import FinQAExample
from src.llm_client import LLMClient, LLMResponse
from src.metrics import (
    PredictionResult,
    EvaluationReport,
    execution_accuracy,
    program_accuracy,
    normalize_answer,
)


# ── System prompt ───────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a financial analyst expert at numerical reasoning over financial reports. \
You will be given context from a financial report (text and tables) and a question \
that requires numerical computation to answer.

Your task:
1. Identify the relevant numbers from the context
2. Determine the mathematical operations needed
3. Show your step-by-step reasoning
4. Provide the final numerical answer

Important rules:
- Express percentages as numbers (e.g., 56.25% → write 56.25%)
- For yes/no questions, answer "yes" or "no"
- Show your work clearly before giving the final answer
- Use the EXACT format specified below for your final answer"""


# ── Exemplos demonstrativos (few-shot) ─────────────────────
# Selecionados manualmente para cobrir operações representativas
# do FinQA: subtract, divide, multi-step, porcentagem, booleano.

FEW_SHOT_EXAMPLES = [
    {
        "context": (
            "### Relevant Text (Before Table)\n"
            "the net change in the total valuation allowance for the years ended "
            "december 31 , 2008 , 2007 and 2006 was an increase of $ 2309.9 million , "
            "an increase of $ 2303.0 million , and an increase of $ 7.3 million , respectively ."
        ),
        "question": "what is the change in net assets from 2007 to 2008?",
        "reasoning": (
            "Step 1: Identify the relevant values.\n"
            "  - Net change in 2008: $2309.9 million\n"
            "  - Net change in 2007: $2303.0 million\n\n"
            "Step 2: Calculate the difference.\n"
            "  Change = 2309.9 - 2303.0 = 6.9\n\n"
            "The change in net assets from 2007 to 2008 is 6.9 million."
        ),
        "program": "subtract(2309.9, 2303.0)",
        "answer": "6.9",
    },
    {
        "context": (
            "### Financial Table\n"
            "                    | 2009    | 2008\n"
            "-----------------------------------------\n"
            "trading days        | 261     | 260\n"
            "days with gains > $210M | 12  | 5"
        ),
        "question": "on what percent of trading days were there market gains above $210 million?",
        "reasoning": (
            "Step 1: Identify the relevant values.\n"
            "  - Days with gains > $210M in 2009: 12\n"
            "  - Total trading days in 2009: 261\n\n"
            "Step 2: Calculate the percentage.\n"
            "  Percentage = 12 / 261 = 0.04598 = 4.6%\n\n"
            "The percentage of trading days with gains above $210 million is 4.6%."
        ),
        "program": "divide(12, 261)",
        "answer": "4.6%",
    },
    {
        "context": (
            "### Relevant Text (Before Table)\n"
            "loans held-for-sale that are carried at locom increased to $ 2.5 billion "
            "at december 31 , 2010 , compared to $ 1.6 billion at december 31 , 2009 ."
        ),
        "question": (
            "what was the growth rate of the loans held-for-sale that are carried "
            "at locom from 2009 to 2010?"
        ),
        "reasoning": (
            "Step 1: Identify the relevant values.\n"
            "  - Loans in 2010: $2.5 billion\n"
            "  - Loans in 2009: $1.6 billion\n\n"
            "Step 2: Calculate the growth rate.\n"
            "  Growth rate = (2.5 - 1.6) / 1.6 = 0.9 / 1.6 = 0.5625 = 56.25%\n\n"
            "The growth rate is 56.25%."
        ),
        "program": "divide(subtract(2.5, 1.6), 1.6)",
        "answer": "56.25%",
    },
    {
        "context": (
            "### Financial Table\n"
            "year | revenue (millions)\n"
            "-------------------------\n"
            "2017 | 959.2\n"
            "2018 | 991.1"
        ),
        "question": "what was the percentage change in revenue from 2017 to 2018?",
        "reasoning": (
            "Step 1: Identify the relevant values.\n"
            "  - Revenue in 2018: $991.1 million\n"
            "  - Revenue in 2017: $959.2 million\n\n"
            "Step 2: Calculate the percentage change.\n"
            "  Change = (991.1 - 959.2) / 959.2 = 31.9 / 959.2 = 0.03326 = 3.3%\n\n"
            "The percentage change in revenue is 3.3%."
        ),
        "program": "divide(subtract(991.1, 959.2), 959.2)",
        "answer": "3.3%",
    },
    {
        "context": (
            "### Relevant Text (Before Table)\n"
            "during fiscal 2009 , the company granted 607 thousand share-based awards "
            "at a weighted-average share price of $ 18.13 .\n"
            "### Financial Table\n"
            "                     | fiscal 2009\n"
            "------------------------------\n"
            "total compensation  | $ 3.3 million"
        ),
        "question": (
            "is the total value of shares granted greater than the total compensation?"
        ),
        "reasoning": (
            "Step 1: Calculate the total value of shares granted.\n"
            "  Value = 607 thousand × $18.13 = 607 × 18.13 × 1000 = $11,002,910\n"
            "  That's approximately $11.0 million.\n\n"
            "Step 2: Compare with total compensation.\n"
            "  Total compensation = $3.3 million\n"
            "  $11.0 million > $3.3 million\n\n"
            "Yes, the total value of shares granted is greater than the total compensation."
        ),
        "program": "greater(multiply(multiply(607, 18.13), const_1000), multiply(3.3, const_1000000))",
        "answer": "yes",
    },
]


# ── Construção do prompt ────────────────────────────────────
def build_few_shot_prompt(
    question: str,
    context: str,
    examples: list[dict] = FEW_SHOT_EXAMPLES,
    use_cot: bool = True,
) -> str:
    """
    Constrói o prompt few-shot com chain-of-thought.

    Args:
        question: Pergunta do FinQA
        context: Contexto pré-extraído (texto + tabela)
        examples: Exemplos demonstrativos
        use_cot: Se True, inclui cadeia de raciocínio nos exemplos

    Returns:
        Prompt formatado
    """
    parts = []

    # Exemplos demonstrativos
    for i, ex in enumerate(examples, 1):
        parts.append(f"--- Example {i} ---")
        parts.append(f"Context:\n{ex['context']}")
        parts.append(f"\nQuestion: {ex['question']}")

        if use_cot:
            parts.append(f"\nReasoning:\n{ex['reasoning']}")

        parts.append(f"\nFinal Answer: {ex['answer']}")
        parts.append("")

    # Pergunta alvo
    parts.append("--- Your Turn ---")
    parts.append(f"Context:\n{context}")
    parts.append(f"\nQuestion: {question}")

    if use_cot:
        parts.append(
            "\nShow your step-by-step reasoning, then provide your final answer "
            "on the last line in the format:\nFinal Answer: <your answer>"
        )
    else:
        parts.append("\nFinal Answer:")

    return "\n".join(parts)


# ── Extração de resposta ────────────────────────────────────
def extract_answer(response_text: str) -> str:
    """
    Extrai a resposta final da saída do LLM.

    Procura padrões como:
      - "Final Answer: X"
      - "Answer: X"
      - Última linha numérica
    """
    # Tentar "Final Answer: ..."
    match = re.search(
        r"(?:final\s+answer|answer)\s*:\s*(.+?)(?:\n|$)",
        response_text,
        re.IGNORECASE,
    )
    if match:
        answer = match.group(1).strip()
        # Limpar formatação residual
        answer = answer.strip("*").strip("`").strip()
        return answer

    # Fallback: última linha não-vazia
    lines = [l.strip() for l in response_text.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]

    return ""


def extract_program(response_text: str) -> str:
    """
    Tenta extrair um programa DSL da resposta do LLM.

    Procura padrões como operações aninhadas:
      subtract(X, Y), divide(X, Y), etc.
    """
    # Procurar padrões de programa DSL
    ops = r"(?:add|subtract|multiply|divide|greater|exp|table_sum|table_average|table_max|table_min)"
    pattern = rf"({ops}\(.+\))"
    match = re.search(pattern, response_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


# ── Pipeline principal ──────────────────────────────────────
class ManualBaselinePipeline:
    """
    Pipeline de engenharia de prompts manual.

    Implementa few-shot chain-of-thought prompting convencional
    como baseline para comparação com otimização automática de prompts.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        few_shot_examples: list[dict] = FEW_SHOT_EXAMPLES,
        use_chain_of_thought: bool = True,
    ):
        self.llm = llm_client
        self.examples = few_shot_examples
        self.use_cot = use_chain_of_thought

    def predict(self, example: FinQAExample) -> PredictionResult:
        """Faz predição para um exemplo FinQA."""
        prompt = build_few_shot_prompt(
            question=example.question,
            context=example.context,
            examples=self.examples,
            use_cot=self.use_cot,
        )

        response: LLMResponse = self.llm.complete(
            prompt=prompt,
            system=SYSTEM_PROMPT,
        )

        predicted_answer = extract_answer(response.text)
        predicted_program = extract_program(response.text)

        exec_acc = execution_accuracy(predicted_answer, example.exe_ans)
        prog_acc = program_accuracy(predicted_program, example.program)

        return PredictionResult(
            example_id=example.id,
            question=example.question,
            gold_exe_ans=example.exe_ans,
            gold_program=example.program,
            predicted_answer=predicted_answer,
            predicted_program=predicted_program,
            exec_acc=exec_acc,
            prog_acc=prog_acc,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_seconds=response.latency_seconds,
            raw_response=response.text,
        )

    def evaluate(
        self,
        examples: list[FinQAExample],
        verbose: bool = True,
        checkpoint_path: str = "results/.checkpoint_baseline.json",
    ) -> EvaluationReport:
        """
        Avalia o pipeline em uma lista de exemplos com checkpointing.

        Salva progresso a cada exemplo. Se interrompido, retoma de onde parou.
        """
        import json
        from pathlib import Path

        # Carregar checkpoint existente
        completed = {}
        if Path(checkpoint_path).exists():
            with open(checkpoint_path) as f:
                completed = json.load(f)
            print(f"  ✓ Checkpoint encontrado: {len(completed)} exemplos já processados")

        report = EvaluationReport(
            pipeline_name="manual_baseline",
            split="dev",
            num_examples=len(examples),
        )

        for i, ex in enumerate(examples):
            # Pular exemplos já processados
            if ex.id in completed:
                # Reconstruir PredictionResult do checkpoint
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
            try:
                result = self.predict(ex)
            except Exception as e:
                print(f"  [ERROR] Example {ex.id}: {e}")
                result = PredictionResult(
                    example_id=ex.id,
                    question=ex.question,
                    gold_exe_ans=ex.exe_ans,
                    gold_program=ex.program,
                    predicted_answer="",
                    predicted_program="",
                    exec_acc=False,
                    prog_acc=False,
                    latency_seconds=0,
                    raw_response=f"ERROR: {e}",
                )
            report.predictions.append(result)

            # Salvar no checkpoint
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
                done = len(report.predictions)
                acc_so_far = (
                    sum(1 for p in report.predictions if p.exec_acc)
                    / done * 100
                )
                print(
                    f"  [{done}/{len(examples)}] "
                    f"Exec Acc: {acc_so_far:.1f}% | "
                    f"LLM: {self.llm.status()}"
                )

        # Limpar checkpoint ao concluir com sucesso
        if Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()
            print("  ✓ Checkpoint removido (execução completa)")

        return report