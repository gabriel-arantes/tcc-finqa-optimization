"""
Módulo DSPy para raciocínio numérico sobre o dataset FinQA.

Define a assinatura e o módulo ChainOfThought que será otimizado
pelos três otimizadores: BootstrapFewShot, MIPROv2 e GEPA.
"""

import dspy

from src.data_loader import FinQAExample
from src.metrics import execution_accuracy, normalize_answer


# ── Assinatura DSPy ─────────────────────────────────────────
class FinQAReasoning(dspy.Signature):
    """Answer a numerical reasoning question about financial data.

    Given context from a financial report (text and/or tables) and a question,
    perform step-by-step numerical reasoning to compute the answer.

    Express percentages as numbers with % sign (e.g., 56.25%).
    For yes/no questions, answer 'yes' or 'no'.
    """

    context: str = dspy.InputField(
        desc="Financial report context including text and tables"
    )
    question: str = dspy.InputField(
        desc="Question requiring numerical reasoning over the financial data"
    )
    reasoning: str = dspy.OutputField(
        desc="Step-by-step numerical reasoning showing all calculations"
    )
    answer: str = dspy.OutputField(
        desc="Final numerical answer (e.g., '6.9', '56.25%', 'yes')"
    )


# ── Módulo DSPy ─────────────────────────────────────────────
class FinQAModule(dspy.Module):
    """
    Módulo DSPy para raciocínio numérico FinQA.

    Usa ChainOfThought para gerar raciocínio intermediário
    antes da resposta final.
    """

    def __init__(self):
        super().__init__()
        self.reason = dspy.ChainOfThought(FinQAReasoning)

    def forward(self, context: str, question: str) -> dspy.Prediction:
        return self.reason(context=context, question=question)


# ── Métrica DSPy ────────────────────────────────────────────
def finqa_execution_accuracy(example, prediction, trace=None) -> bool:
    """
    Métrica de execution accuracy para otimizadores DSPy.

    Compatível com a interface metric(example, prediction, trace)
    exigida pelos otimizadores.
    """
    predicted = prediction.answer if hasattr(prediction, "answer") else ""
    gold = example.exe_ans if hasattr(example, "exe_ans") else example.get("exe_ans", "")

    return execution_accuracy(predicted, gold)


def finqa_execution_accuracy_gepa(example, prediction, trace=None, pred_name=None, pred_trace=None) -> bool:
    """
    Métrica compatível com GEPA (requer 5 argumentos).
    """
    return finqa_execution_accuracy(example, prediction, trace)


# ── Conversão de dados ──────────────────────────────────────
def finqa_to_dspy_examples(
    examples: list[FinQAExample],
) -> list[dspy.Example]:
    """
    Converte exemplos FinQA para formato dspy.Example.

    Cada exemplo DSPy inclui:
      - Inputs: context, question
      - Labels: reasoning (vazio, será gerado), answer, exe_ans
    """
    dspy_examples = []
    for ex in examples:
        dspy_ex = dspy.Example(
            context=ex.context,
            question=ex.question,
            answer=str(ex.gold_answer),
            exe_ans=ex.exe_ans,
            program=ex.program,
            id=ex.id,
        ).with_inputs("context", "question")
        dspy_examples.append(dspy_ex)
    return dspy_examples