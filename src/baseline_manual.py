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
# 20 exemplos selecionados manualmente para cobrir operações
# representativas do FinQA, maximizando diversidade:
#
#  1. subtract        — diferença entre valores de anos consecutivos
#  2. divide          — cálculo de proporção/percentual simples
#  3. subtract+div    — taxa de crescimento (multi-step)
#  4. pct_change      — variação percentual
#  5. boolean         — comparação booleana (greater)
#  6. add             — soma de componentes
#  7. multiply        — multiplicação para cálculo de valor total
#  8. average         — cálculo de média aritmética
#  9. pct_of_total    — percentual de uma parte sobre o total
# 10. add+divide      — soma seguida de divisão (multi-step)
# 11. subtract (neg)  — diferença com resultado negativo (queda)
# 12. multiply+add    — cálculo de valor total com múltiplos itens
# 13. multi-row add   — soma de múltiplas linhas de tabela
# 14. divide (ratio)  — razão entre duas métricas financeiras
# 15. subtract+mult   — diferença convertida para unidade
# 16. pct_change (neg)— variação percentual negativa (queda)
# 17. margin          — cálculo de margem (lucro/receita)
# 18. boolean (no)    — comparação booleana com resposta "no"
# 19. add+subtract    — soma parcial seguida de subtração
# 20. divide+subtract — razão seguida de diferença entre razões

FEW_SHOT_EXAMPLES = [
    # ── Demo 1: subtract ──
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
    # ── Demo 2: divide (proporção simples) ──
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
    # ── Demo 3: multi-step subtract+divide (taxa de crescimento) ──
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
    # ── Demo 4: percentage change ──
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
    # ── Demo 5: boolean/greater ──
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
    # ── Demo 6: add (soma de componentes) ──
    {
        "context": (
            "### Relevant Text (Before Table)\n"
            "entergy arkansas's receivables from or ( payables to ) the money pool were "
            "as follows:\n"
            "### Financial Table\n"
            "                     | (in thousands)\n"
            "--------------------------------------\n"
            "2007 january          | ($  35643)\n"
            "2007 february         | ($  ## )\n"
            "net income            | $  105221\n"
            "other comprehensive income | $  2399"
        ),
        "question": "what is the total of net income and other comprehensive income?",
        "reasoning": (
            "Step 1: Identify the relevant values.\n"
            "  - Net income: $105,221 thousand\n"
            "  - Other comprehensive income: $2,399 thousand\n\n"
            "Step 2: Calculate the total.\n"
            "  Total = 105221 + 2399 = 107620\n\n"
            "The total of net income and other comprehensive income is $107,620 thousand."
        ),
        "program": "add(105221, 2399)",
        "answer": "107620",
    },
    # ── Demo 7: multiply (cálculo de valor) ──
    {
        "context": (
            "### Relevant Text (Before Table)\n"
            "the company repurchased 4.2 million shares at an average price of $35.50 "
            "per share during the fiscal year 2016."
        ),
        "question": "what was the total cost of share repurchases in 2016, in millions?",
        "reasoning": (
            "Step 1: Identify the relevant values.\n"
            "  - Shares repurchased: 4.2 million\n"
            "  - Average price: $35.50 per share\n\n"
            "Step 2: Calculate the total cost.\n"
            "  Total cost = 4.2 × 35.50 = 149.1 million\n\n"
            "The total cost of share repurchases was $149.1 million."
        ),
        "program": "multiply(4.2, 35.50)",
        "answer": "149.1",
    },
    # ── Demo 8: average (média aritmética) ──
    {
        "context": (
            "### Financial Table\n"
            "year | operating expenses (millions)\n"
            "-------------------------------------\n"
            "2014 | 580\n"
            "2015 | 612\n"
            "2016 | 647"
        ),
        "question": "what was the average annual operating expenses from 2014 to 2016?",
        "reasoning": (
            "Step 1: Identify the values for each year.\n"
            "  - 2014: $580 million\n"
            "  - 2015: $612 million\n"
            "  - 2016: $647 million\n\n"
            "Step 2: Calculate the average.\n"
            "  Average = (580 + 612 + 647) / 3 = 1839 / 3 = 613\n\n"
            "The average annual operating expenses from 2014 to 2016 was $613 million."
        ),
        "program": "divide(add(add(580, 612), 647), const_3)",
        "answer": "613",
    },
    # ── Demo 9: pct_of_total (percentual de uma parte sobre total) ──
    {
        "context": (
            "### Financial Table\n"
            "segment               | net sales 2015 (millions)\n"
            "-----------------------------------------------\n"
            "consumer products      | 1250\n"
            "industrial solutions   | 890\n"
            "total                  | 2140"
        ),
        "question": "what percentage of total net sales was from consumer products in 2015?",
        "reasoning": (
            "Step 1: Identify the relevant values.\n"
            "  - Consumer products sales: $1,250 million\n"
            "  - Total net sales: $2,140 million\n\n"
            "Step 2: Calculate the percentage.\n"
            "  Percentage = 1250 / 2140 = 0.5841 = 58.41%\n\n"
            "Consumer products represented 58.41% of total net sales."
        ),
        "program": "multiply(divide(1250, 2140), const_100)",
        "answer": "58.41%",
    },
    # ── Demo 10: multi-step add+divide (soma seguida de divisão) ──
    {
        "context": (
            "### Financial Table\n"
            "                    | 2018 (thousands)\n"
            "--------------------------------------\n"
            "short-term debt      | 450\n"
            "long-term debt       | 1320\n"
            "total equity         | 2850"
        ),
        "question": "what was the debt-to-equity ratio in 2018?",
        "reasoning": (
            "Step 1: Calculate total debt.\n"
            "  Total debt = short-term + long-term = 450 + 1320 = 1770\n\n"
            "Step 2: Calculate the debt-to-equity ratio.\n"
            "  Debt-to-equity = 1770 / 2850 = 0.62\n\n"
            "The debt-to-equity ratio in 2018 was 0.62."
        ),
        "program": "divide(add(450, 1320), 2850)",
        "answer": "0.62",
    },
    # ── Demo 11: subtract com resultado negativo (queda) ──
    {
        "context": (
            "### Financial Table\n"
            "                      | 2016 (millions) | 2015 (millions)\n"
            "-----------------------------------------------------------\n"
            "net income            | 312              | 487"
        ),
        "question": "what was the change in net income from 2015 to 2016?",
        "reasoning": (
            "Step 1: Identify the relevant values.\n"
            "  - Net income in 2016: $312 million\n"
            "  - Net income in 2015: $487 million\n\n"
            "Step 2: Calculate the change.\n"
            "  Change = 312 - 487 = -175\n\n"
            "Net income decreased by $175 million from 2015 to 2016."
        ),
        "program": "subtract(312, 487)",
        "answer": "-175",
    },
    # ── Demo 12: multiply+add (valor total de múltiplos itens) ──
    {
        "context": (
            "### Financial Table\n"
            "award type        | shares (thousands) | weighted avg price\n"
            "-------------------------------------------------------------\n"
            "stock options     | 250                | $42.00\n"
            "restricted stock  | 120                | $45.30"
        ),
        "question": "what was the combined total value of all awards granted, in thousands?",
        "reasoning": (
            "Step 1: Calculate value of stock options.\n"
            "  Options value = 250 × $42.00 = $10,500 thousand\n\n"
            "Step 2: Calculate value of restricted stock.\n"
            "  Restricted value = 120 × $45.30 = $5,436 thousand\n\n"
            "Step 3: Calculate total combined value.\n"
            "  Total = 10500 + 5436 = 15936\n\n"
            "The combined total value was $15,936 thousand."
        ),
        "program": "add(multiply(250, 42.00), multiply(120, 45.30))",
        "answer": "15936",
    },
    # ── Demo 13: soma de múltiplas linhas (table_sum) ──
    {
        "context": (
            "### Financial Table\n"
            "region          | sales 2017 (millions)\n"
            "-------------------------------------------\n"
            "north america    | 3420\n"
            "europe           | 1890\n"
            "asia pacific     | 1150\n"
            "latin america    | 640"
        ),
        "question": "what were the total global sales in 2017?",
        "reasoning": (
            "Step 1: Identify sales for each region.\n"
            "  - North America: $3,420 million\n"
            "  - Europe: $1,890 million\n"
            "  - Asia Pacific: $1,150 million\n"
            "  - Latin America: $640 million\n\n"
            "Step 2: Calculate total.\n"
            "  Total = 3420 + 1890 + 1150 + 640 = 7100\n\n"
            "Total global sales in 2017 were $7,100 million."
        ),
        "program": "add(add(add(3420, 1890), 1150), 640)",
        "answer": "7100",
    },
    # ── Demo 14: divide (razão entre métricas) ──
    {
        "context": (
            "### Financial Table\n"
            "                          | 2019 (millions)\n"
            "----------------------------------------------\n"
            "current assets             | 8540\n"
            "current liabilities        | 5230"
        ),
        "question": "what was the current ratio in 2019?",
        "reasoning": (
            "Step 1: Identify the relevant values.\n"
            "  - Current assets: $8,540 million\n"
            "  - Current liabilities: $5,230 million\n\n"
            "Step 2: Calculate the current ratio.\n"
            "  Current ratio = 8540 / 5230 = 1.63\n\n"
            "The current ratio in 2019 was 1.63."
        ),
        "program": "divide(8540, 5230)",
        "answer": "1.63",
    },
    # ── Demo 15: subtract+multiply (diferença em unidades diferentes) ──
    {
        "context": (
            "### Financial Table\n"
            "                     | 2020 (billions) | 2019 (billions)\n"
            "-----------------------------------------------------------\n"
            "total assets          | 15.8            | 14.3"
        ),
        "question": "what was the increase in total assets from 2019 to 2020, in millions?",
        "reasoning": (
            "Step 1: Identify the relevant values.\n"
            "  - Total assets 2020: $15.8 billion\n"
            "  - Total assets 2019: $14.3 billion\n\n"
            "Step 2: Calculate the increase in billions.\n"
            "  Increase = 15.8 - 14.3 = 1.5 billion\n\n"
            "Step 3: Convert to millions.\n"
            "  1.5 billion × 1000 = 1500 million\n\n"
            "The increase in total assets was $1,500 million."
        ),
        "program": "multiply(subtract(15.8, 14.3), const_1000)",
        "answer": "1500",
    },
    # ── Demo 16: variação percentual negativa ──
    {
        "context": (
            "### Financial Table\n"
            "                          | 2018 (millions) | 2017 (millions)\n"
            "----------------------------------------------------------------\n"
            "research & development     | 285             | 340"
        ),
        "question": "what was the percentage change in R&D spending from 2017 to 2018?",
        "reasoning": (
            "Step 1: Identify the relevant values.\n"
            "  - R&D spending 2018: $285 million\n"
            "  - R&D spending 2017: $340 million\n\n"
            "Step 2: Calculate the percentage change.\n"
            "  Change = (285 - 340) / 340 = -55 / 340 = -0.1618 = -16.18%\n\n"
            "R&D spending decreased by 16.18% from 2017 to 2018."
        ),
        "program": "divide(subtract(285, 340), 340)",
        "answer": "-16.18%",
    },
    # ── Demo 17: margin (lucro/receita) ──
    {
        "context": (
            "### Financial Table\n"
            "                          | 2020 (millions)\n"
            "----------------------------------------------\n"
            "total revenue              | 4200\n"
            "cost of goods sold         | 2730\n"
            "gross profit               | 1470"
        ),
        "question": "what was the gross margin in 2020?",
        "reasoning": (
            "Step 1: Identify the relevant values.\n"
            "  - Gross profit: $1,470 million\n"
            "  - Total revenue: $4,200 million\n\n"
            "Step 2: Calculate the gross margin.\n"
            "  Gross margin = 1470 / 4200 = 0.35 = 35.0%\n\n"
            "The gross margin in 2020 was 35.0%."
        ),
        "program": "multiply(divide(1470, 4200), const_100)",
        "answer": "35.0%",
    },
    # ── Demo 18: boolean com resposta "no" ──
    {
        "context": (
            "### Financial Table\n"
            "                          | 2019 (millions) | 2018 (millions)\n"
            "----------------------------------------------------------------\n"
            "operating income           | 523             | 610"
        ),
        "question": "did operating income increase from 2018 to 2019?",
        "reasoning": (
            "Step 1: Identify the relevant values.\n"
            "  - Operating income 2019: $523 million\n"
            "  - Operating income 2018: $610 million\n\n"
            "Step 2: Compare the values.\n"
            "  $523 million < $610 million\n"
            "  Operating income decreased from 2018 to 2019.\n\n"
            "No, operating income did not increase."
        ),
        "program": "greater(523, 610)",
        "answer": "no",
    },
    # ── Demo 19: add+subtract (soma parcial menos um valor) ──
    {
        "context": (
            "### Financial Table\n"
            "                          | 2017 (millions)\n"
            "----------------------------------------------\n"
            "total revenue              | 9800\n"
            "domestic revenue           | 6200\n"
            "european revenue           | 2400\n"
            "other international        | 1200"
        ),
        "question": "how much revenue came from non-domestic markets in 2017?",
        "reasoning": (
            "Step 1: Identify the relevant values.\n"
            "  - Total revenue: $9,800 million\n"
            "  - Domestic revenue: $6,200 million\n\n"
            "Step 2: Calculate non-domestic revenue.\n"
            "  Non-domestic = Total - Domestic = 9800 - 6200 = 3600\n\n"
            "Revenue from non-domestic markets was $3,600 million."
        ),
        "program": "subtract(9800, 6200)",
        "answer": "3600",
    },
    # ── Demo 20: diferença entre razões de dois anos ──
    {
        "context": (
            "### Financial Table\n"
            "                   | 2016 (millions) | 2015 (millions)\n"
            "---------------------------------------------------------\n"
            "operating expenses  | 720              | 680\n"
            "total revenue       | 3200             | 3000"
        ),
        "question": (
            "what was the change in operating expense ratio from 2015 to 2016, "
            "in percentage points?"
        ),
        "reasoning": (
            "Step 1: Calculate operating expense ratio for each year.\n"
            "  - 2016 ratio = 720 / 3200 = 0.225 = 22.5%\n"
            "  - 2015 ratio = 680 / 3000 = 0.2267 = 22.67%\n\n"
            "Step 2: Calculate the change in percentage points.\n"
            "  Change = 22.5 - 22.67 = -0.17 percentage points\n\n"
            "The operating expense ratio decreased by 0.17 percentage points."
        ),
        "program": "subtract(divide(720, 3200), divide(680, 3000))",
        "answer": "-0.17",
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

        if Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()
            print("  ✓ Checkpoint removido (execução completa)")

        return report