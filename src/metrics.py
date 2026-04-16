"""
Métricas de avaliação para o benchmark FinQA.

Métricas implementadas:
  - Execution accuracy (primária): compara resposta numérica do modelo com exe_ans
  - Program accuracy (secundária): compara programa gerado com programa gold
  - Métricas operacionais: tokens consumidos, latência
"""

import re
import math
from dataclasses import dataclass, field
from typing import Optional


# ── Normalização de respostas ───────────────────────────────
def normalize_answer(answer: str | float | int) -> Optional[float]:
    """
    Normaliza uma resposta para valor numérico comparável.

    Trata formatos comuns do FinQA:
      - Porcentagens: "56.25%", "-3.2%"
      - Negativos com parênteses: "(23158)"
      - Respostas booleanas: "yes"/"no" → 1.0/0.0
      - Números com vírgulas: "1,327,657"
      - Strings numéricas: "380", "6.9"
      - Unidades textuais: "114 million", "3.8 billion"
    """
    if isinstance(answer, (int, float)):
        return float(answer)

    if not isinstance(answer, str):
        return None

    s = str(answer).strip().lower()

    # Booleanos
    if s in ("yes", "true"):
        return 1.0
    if s in ("no", "false"):
        return 0.0

    # Remover unidades textuais comuns (mantém o número)
    s = re.sub(r"\s*(million|billion|thousand|percent|percentage points?|bps|basis points?)\s*$", "", s, flags=re.IGNORECASE)

    # Limpar caracteres comuns
    s = s.replace(",", "").replace("$", "").replace(" ", "")

    # Porcentagens
    if s.endswith("%"):
        try:
            return float(s[:-1])
        except ValueError:
            return None

    # Negativos com parênteses: (123) → -123
    match = re.match(r"^\(([0-9.]+)\)$", s)
    if match:
        try:
            return -float(match.group(1))
        except ValueError:
            return None

    # Número direto
    try:
        return float(s)
    except ValueError:
        return None


def _is_percentage_str(answer: str | float | int) -> bool:
    """Verifica se a resposta original contém sinal de %."""
    if isinstance(answer, str) and "%" in answer:
        return True
    return False


def execution_accuracy(
    predicted: str | float,
    gold_exe_ans: float | str,
    tolerance: float = 5e-3,
) -> bool:
    """
    Calcula execution accuracy para um par (predição, gold).

    A métrica principal do leaderboard FinQA. Compara o resultado
    numérico do modelo com o exe_ans do dataset.

    Estratégia de comparação:
      1. Comparação direta com tolerância relativa (0.5%)
      2. Se falhar e a predição tem "%": tenta pred/100 vs gold
         (cobre o caso onde gold=0.935 e modelo responde "93.5%")
      3. Se falhar e gold parece porcentagem: tenta pred vs gold/100
    """
    pred_val = normalize_answer(predicted)
    gold_val = normalize_answer(gold_exe_ans)

    if pred_val is None or gold_val is None:
        return False

    def _compare(a: float, b: float) -> bool:
        """Compara dois números com tolerância relativa."""
        # Ambos zero
        if abs(a) < 1e-10 and abs(b) < 1e-10:
            return True
        # Um é zero
        if abs(b) < 1e-10:
            return abs(a) < tolerance
        # Erro relativo
        return abs(a - b) / abs(b) < tolerance

    # 1. Comparação direta
    if _compare(pred_val, gold_val):
        return True

    # 2. Modelo respondeu com %, gold é fração decimal
    #    Ex: predicted="93.5%" → pred_val=93.5, gold=0.935 → 93.5/100=0.935 ✓
    if _is_percentage_str(predicted) and abs(gold_val) < 100:
        if _compare(pred_val / 100, gold_val):
            return True

    # 3. Gold é porcentagem "grande", modelo respondeu fração
    #    Ex: predicted="0.935", gold=93.5 → 0.935*100=93.5 ✓
    if not _is_percentage_str(predicted) and abs(gold_val) > 1:
        if _compare(pred_val * 100, gold_val):
            return True

    return False


def program_accuracy(
    predicted_program: str,
    gold_program: str,
) -> bool:
    """
    Calcula program accuracy — se o programa de raciocínio gerado
    é equivalente ao programa gold.

    Normaliza espaços e formatos antes da comparação.
    """
    def _normalize_program(prog: str) -> str:
        # Remove espaços extras e padroniza
        prog = prog.strip().lower()
        prog = re.sub(r"\s+", " ", prog)
        # Normaliza separadores
        prog = prog.replace(", ", ",").replace(" ,", ",")
        return prog

    return _normalize_program(predicted_program) == _normalize_program(gold_program)


# ── Resultado agregado ──────────────────────────────────────
@dataclass
class PredictionResult:
    """Resultado de uma predição individual."""

    example_id: str
    question: str
    gold_exe_ans: float | str
    gold_program: str
    predicted_answer: str
    predicted_program: str = ""
    exec_acc: bool = False
    prog_acc: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    raw_response: str = ""


@dataclass
class EvaluationReport:
    """Relatório agregado de avaliação."""

    pipeline_name: str
    split: str
    num_examples: int
    predictions: list[PredictionResult] = field(default_factory=list)

    @property
    def execution_accuracy(self) -> float:
        if not self.predictions:
            return 0.0
        correct = sum(1 for p in self.predictions if p.exec_acc)
        return correct / len(self.predictions)

    @property
    def program_accuracy(self) -> float:
        if not self.predictions:
            return 0.0
        correct = sum(1 for p in self.predictions if p.prog_acc)
        return correct / len(self.predictions)

    @property
    def total_input_tokens(self) -> int:
        return sum(p.input_tokens for p in self.predictions)

    @property
    def total_output_tokens(self) -> int:
        return sum(p.output_tokens for p in self.predictions)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_latency(self) -> float:
        if not self.predictions:
            return 0.0
        return sum(p.latency_seconds for p in self.predictions) / len(self.predictions)

    def summary(self) -> dict:
        return {
            "pipeline": self.pipeline_name,
            "split": self.split,
            "n": self.num_examples,
            "execution_accuracy": round(self.execution_accuracy * 100, 2),
            "program_accuracy": round(self.program_accuracy * 100, 2),
            "total_tokens": self.total_tokens,
            "avg_input_tokens": round(self.total_input_tokens / max(1, self.num_examples), 1),
            "avg_output_tokens": round(self.total_output_tokens / max(1, self.num_examples), 1),
            "avg_latency_s": round(self.avg_latency, 2),
        }

    def __str__(self) -> str:
        s = self.summary()
        lines = [
            f"\n{'='*60}",
            f"Pipeline: {s['pipeline']} | Split: {s['split']} | N={s['n']}",
            f"{'='*60}",
            f"  Execution Accuracy:  {s['execution_accuracy']:.2f}%",
            f"  Program Accuracy:    {s['program_accuracy']:.2f}%",
            f"  Total Tokens:        {s['total_tokens']:,}",
            f"  Avg Input Tokens:    {s['avg_input_tokens']:.1f}",
            f"  Avg Output Tokens:   {s['avg_output_tokens']:.1f}",
            f"  Avg Latency:         {s['avg_latency_s']:.2f}s",
            f"{'='*60}",
        ]
        return "\n".join(lines)


# ── Teste rápido ────────────────────────────────────────────
if __name__ == "__main__":
    # Testes de normalização
    assert normalize_answer("56.25%") == 56.25
    assert normalize_answer("-3.2%") == -3.2
    assert normalize_answer("(23158)") == -23158.0
    assert normalize_answer("yes") == 1.0
    assert normalize_answer("1,327,657") == 1327657.0
    assert normalize_answer(3.8) == 3.8

    # Testes de execution accuracy
    assert execution_accuracy("3.8", 3.8) is True
    assert execution_accuracy("3.80001", 3.8) is True
    assert execution_accuracy("56.25", 56.25) is True
    assert execution_accuracy("100", 200) is False
    assert execution_accuracy("yes", 1.0) is True

    print("Todos os testes de métricas passaram!")
