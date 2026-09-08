import argparse
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from statistics import fmean
from typing import Any, TypeVar

from dotenv import load_dotenv

from rag.generation.clients.gemini_client import GeminiClient
from rag.generation.generator import Generator
from rag.retrieval.reranking_retriever import RerankingRetriever

REFUSAL_TEXT = (
    "The retrieved context does not contain enough information "
    "to answer this question."
)

JUDGE_SYSTEM_PROMPT = """
You are evaluating the quality of a retrieval-augmented generation (RAG)
system.

Use only the evaluation materials provided:
- whether the dataset marks the question as answerable
- user question
- reference answer
- gold evidence
- retrieved context
- generated answer
- cited sources

Do not use outside knowledge.

IMPORTANT EVALUATION PRINCIPLE

Correctness and completeness are END-TO-END metrics. They are judged
against the reference answer and gold evidence, not merely against what
retrieval happened to return.

Faithfulness is a GROUNDING metric. It is judged only against the
retrieved context.

Therefore, if answerable=true and the system refuses to answer because
retrieval missed the evidence:
- correctness must be 1
- relevance must be 1
- completeness must be 1
- faithfulness should still be judged against the retrieved context

A retrieval failure must not receive a high correctness score simply
because the refusal was faithful to the retrieved context.

Score each dimension from 1 to 5.

correctness:
5 = fully consistent with the reference answer and gold evidence
4 = correct with only a minor omission or imprecision
3 = partially correct but missing or misstating an important point
2 = substantially incorrect
1 = incorrect, contradicts the reference material, or refuses an
    answerable question

relevance:
5 = directly and concisely answers the user's question
4 = answers the question with minor irrelevant material
3 = only partially addresses the question
2 = mostly off-topic
1 = does not answer the question, including refusing an answerable
    question

faithfulness:
5 = every substantive claim is supported by retrieved context
4 = almost fully supported, with one minor unsupported detail
3 = some supported and some unsupported claims
2 = major unsupported claims
1 = largely hallucinated or contradicted by retrieved context

completeness:
5 = includes all important information required by the reference answer
4 = misses only a minor detail
3 = misses an important component
2 = contains only a small part of the required answer
1 = effectively incomplete, including refusing an answerable question

citation_support:
5 = all cited sources directly support the claims made
4 = citations are mostly supportive, with a minor mismatch
3 = some citations support the answer and some do not
2 = citations provide weak support
1 = citations do not support the answer, or an answer requiring support
    has no useful citation

Return only valid JSON, with exactly this structure:

{
  "correctness": {"score": 1, "reason": "..."},
  "relevance": {"score": 1, "reason": "..."},
  "faithfulness": {"score": 1, "reason": "..."},
  "completeness": {"score": 1, "reason": "..."},
  "citation_support": {"score": 1, "reason": "..."}
}
""".strip()

JUDGE_DIMENSIONS = (
    "correctness",
    "relevance",
    "faithfulness",
    "completeness",
    "citation_support",
)

T = TypeVar("T")


class APIRateLimiter:
    """
    Keep Gemini calls below a configured request rate.

    For a billing-enabled Gemini project, the default interval is
    intentionally small (1 second) to avoid unnecessary throttling while
    still preventing an immediate burst of requests.

    The evaluator also handles HTTP 429 responses separately by reading
    Gemini's retry delay, waiting, and retrying automatically.
    """

    def __init__(self, min_interval_seconds: float) -> None:
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds cannot be negative.")

        self.min_interval_seconds = min_interval_seconds
        self._last_call_started_at: float | None = None

    def wait(self) -> None:
        if self._last_call_started_at is None:
            self._last_call_started_at = time.monotonic()
            return

        elapsed = time.monotonic() - self._last_call_started_at
        wait_seconds = self.min_interval_seconds - elapsed

        if wait_seconds > 0:
            print(f"  API pacing: waiting {wait_seconds:.1f}s")
            time.sleep(wait_seconds)

        self._last_call_started_at = time.monotonic()


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load and validate the generation-evaluation dataset."""

    with path.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("Evaluation dataset must be a JSON list.")

    required_fields = {
        "id",
        "query",
        "relevant_pages",
        "reference_answer",
        "answerable",
    }

    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Every dataset item must be a JSON object.")

        missing = required_fields - item.keys()

        if missing:
            raise ValueError(
                f"Question {item.get('id', '<unknown>')} "
                f"is missing fields: {sorted(missing)}"
            )

    return questions


def normalize_text(text: str) -> str:
    """Normalize text for exact deterministic comparisons."""

    return " ".join(text.strip().split()).casefold()


def is_expected_refusal(answer: str) -> bool:
    """Check whether the model returned our required grounded refusal."""

    return normalize_text(answer) == normalize_text(REFUSAL_TEXT)


def serialize_retrieved_chunks(chunks: list[Any]) -> list[dict[str, Any]]:
    """Convert retrieved chunks into JSON-serializable records."""

    return [
        {
            "source_id": source_id,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "score": float(chunk.score),
            "text": chunk.text,
        }
        for source_id, chunk in enumerate(chunks, start=1)
    ]


def build_judge_prompt(
    question: dict[str, Any],
    generated_answer: str,
    retrieved_chunks: list[dict[str, Any]],
    citation_ids: list[int],
) -> str:
    """Build the LLM-judge input."""

    cited_sources = [
        chunk
        for chunk in retrieved_chunks
        if chunk["source_id"] in citation_ids
    ]

    payload = {
        "answerable": question["answerable"],
        "question": question["query"],
        "reference_answer": question["reference_answer"],
        "gold_evidence": question.get("gold_evidence"),
        "generated_answer": generated_answer,
        "retrieved_context": retrieved_chunks,
        "citation_ids": citation_ids,
        "cited_sources": cited_sources,
    }

    return (
        "Evaluate this RAG answer using the rubric in the system "
        "instruction.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def parse_judge_response(raw_response: str) -> dict[str, dict[str, Any]]:
    """Parse and strictly validate the judge JSON."""

    cleaned = raw_response.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Judge returned invalid JSON.") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Judge response must be a JSON object.")

    if set(parsed.keys()) != set(JUDGE_DIMENSIONS):
        raise ValueError(
            "Judge response contains unexpected or missing dimensions."
        )

    for dimension in JUDGE_DIMENSIONS:
        value = parsed[dimension]

        if not isinstance(value, dict):
            raise ValueError(f"{dimension} must be an object.")

        if set(value.keys()) != {"score", "reason"}:
            raise ValueError(
                f"{dimension} must contain exactly score and reason."
            )

        score = value["score"]
        reason = value["reason"]

        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError(f"{dimension} score must be an integer.")

        if not 1 <= score <= 5:
            raise ValueError(f"{dimension} score must be between 1 and 5.")

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{dimension} reason must be non-empty.")

    return parsed


def is_rate_limit_error(exc: Exception) -> bool:
    """Detect Gemini HTTP 429 / RESOURCE_EXHAUSTED failures."""

    if getattr(exc, "status_code", None) == 429:
        return True

    message = str(exc).upper()
    return "429" in message and "RESOURCE_EXHAUSTED" in message


def extract_retry_delay_seconds(
    exc: Exception,
    default_seconds: float = 65.0,
) -> float:
    """Read Gemini's retry delay from the exception when possible."""

    message = str(exc)

    patterns = (
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)s",
        r"retryDelay['\"]?\s*:\s*['\"]([0-9]+(?:\.[0-9]+)?)s",
    )

    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)

        if match:
            return float(match.group(1)) + 2.0

    return default_seconds


def call_with_rate_limit_retry(
    operation: Callable[[], T],
    rate_limiter: APIRateLimiter,
    label: str,
    max_retries: int,
) -> T:
    """
    Execute one Gemini operation with local pacing and automatic 429 retry.

    Non-rate-limit exceptions are re-raised so real bugs are not hidden.
    """

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    attempt = 0

    while True:
        rate_limiter.wait()

        try:
            return operation()

        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise

            if attempt >= max_retries:
                raise RuntimeError(
                    f"{label} exceeded {max_retries} rate-limit retries."
                ) from exc

            attempt += 1
            retry_seconds = extract_retry_delay_seconds(exc)

            print(
                f"  Rate limit during {label}. Waiting "
                f"{retry_seconds:.1f}s before retry "
                f"{attempt}/{max_retries}."
            )

            time.sleep(retry_seconds)



def generate_answer_with_retry(
    generator: Generator,
    query: str,
    chunks: list[Any],
    rate_limiter: APIRateLimiter,
    max_rate_limit_retries: int,
    max_json_attempts: int,
) -> Any:
    """
    Generate one RAG answer with retries for malformed model JSON.

    Generator.generate() already validates the model's structured output.
    If Gemini occasionally returns malformed JSON, Generator raises a
    ValueError. That is a transient model-output failure, so we retry the
    generation call rather than failing the whole benchmark question.

    HTTP 429 handling is delegated to call_with_rate_limit_retry().
    """

    if max_json_attempts <= 0:
        raise ValueError("max_json_attempts must be greater than 0.")

    last_error: Exception | None = None

    for attempt in range(1, max_json_attempts + 1):
        try:
            return call_with_rate_limit_retry(
                operation=lambda: generator.generate(
                    query=query,
                    chunks=chunks,
                ),
                rate_limiter=rate_limiter,
                label="answer generation",
                max_retries=max_rate_limit_retries,
            )

        except ValueError as exc:
            last_error = exc

            if attempt < max_json_attempts:
                print(
                    "  Generator returned invalid structured output. "
                    f"Retrying {attempt}/{max_json_attempts}."
                )

    raise RuntimeError(
        "Generator failed to return valid structured output after "
        f"{max_json_attempts} attempts: {last_error}"
    )


def apply_answerable_refusal_override(
    question: dict[str, Any],
    generated_answer: str,
    judge_result: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Enforce end-to-end scoring for answerable questions.

    An answerable question that receives our refusal text is an
    end-to-end failure even if retrieval did not supply the gold passage.

    We deterministically force correctness, relevance, and completeness
    to 1. Faithfulness and citation support remain judge-scored because
    those depend on the actual retrieved context and citation behavior.
    """

    if (
        question["answerable"]
        and is_expected_refusal(generated_answer)
    ):
        judge_result["correctness"] = {
            "score": 1,
            "reason": (
                "The dataset marks this question as answerable, but the "
                "system refused instead of providing the reference answer. "
                "This is an end-to-end correctness failure."
            ),
        }
        judge_result["relevance"] = {
            "score": 1,
            "reason": (
                "The system did not answer an answerable user question; "
                "it returned a refusal instead."
            ),
        }
        judge_result["completeness"] = {
            "score": 1,
            "reason": (
                "The system omitted all required answer content by "
                "refusing an answerable question."
            ),
        }

    return judge_result


def judge_answer(
    judge_client: GeminiClient,
    question: dict[str, Any],
    generated_answer: str,
    retrieved_chunks: list[dict[str, Any]],
    citation_ids: list[int],
    rate_limiter: APIRateLimiter,
    max_rate_limit_retries: int,
    max_json_attempts: int,
) -> dict[str, dict[str, Any]]:
    """Score one answer with the LLM judge."""

    user_prompt = build_judge_prompt(
        question=question,
        generated_answer=generated_answer,
        retrieved_chunks=retrieved_chunks,
        citation_ids=citation_ids,
    )

    last_error: Exception | None = None

    for attempt in range(1, max_json_attempts + 1):
        try:
            raw_response = call_with_rate_limit_retry(
                operation=lambda: judge_client.generate(
                    system_prompt=JUDGE_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                ),
                rate_limiter=rate_limiter,
                label="judge",
                max_retries=max_rate_limit_retries,
            )

            judge_result = parse_judge_response(raw_response)

            return apply_answerable_refusal_override(
                question=question,
                generated_answer=generated_answer,
                judge_result=judge_result,
            )

        except ValueError as exc:
            last_error = exc

            if attempt < max_json_attempts:
                print(
                    "  Judge returned malformed JSON. "
                    f"Retrying {attempt}/{max_json_attempts}."
                )

    raise RuntimeError(
        "Judge failed to return valid JSON after "
        f"{max_json_attempts} attempts: {last_error}"
    )


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """
    Write JSON using a temporary file first.

    This reduces the chance of a corrupted checkpoint if the process
    stops while writing.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.tmp"

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    temporary_path.replace(path)



def is_complete_result(item: dict[str, Any]) -> bool:
    """
    Determine whether a stored result is actually complete.

    Older evaluator versions did not write status="completed". We still
    recognize those results when they contain all required output.
    """

    if item.get("status") == "completed":
        return True

    if not isinstance(item.get("generated_answer"), str):
        return False

    if item.get("answerable") is True:
        return isinstance(item.get("judge"), dict)

    if item.get("answerable") is False:
        return isinstance(item.get("refusal_correct"), bool)

    return False


def load_existing_results(
    checkpoint_path: Path,
    output_path: Path,
) -> dict[str, dict[str, Any]]:
    """
    Load existing results for resume support.

    Checkpoint is preferred. If none exists, an older final output is
    accepted, so a previous --limit run can also be reused.
    """

    for path in (checkpoint_path, output_path):
        if not path.exists():
            continue

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            raw_results = data.get("results", [])

            if not isinstance(raw_results, list):
                continue

            result_map: dict[str, dict[str, Any]] = {}

            for item in raw_results:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("id"), str)
                ):
                    continue

                if is_complete_result(item):
                    item["status"] = "completed"

                result_map[item["id"]] = item

            if result_map:
                print(
                    f"Resume data loaded from {path} "
                    f"({len(result_map)} result(s))."
                )
                return result_map

        except (OSError, json.JSONDecodeError):
            continue

    return {}


def build_checkpoint(
    configuration: dict[str, Any],
    questions: list[dict[str, Any]],
    result_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build checkpoint content in dataset order."""

    return {
        "configuration": configuration,
        "selected_question_ids": [item["id"] for item in questions],
        "results": [
            result_map[item["id"]]
            for item in questions
            if item["id"] in result_map
        ],
    }


def calculate_summary(
    results: list[dict[str, Any]],
    expected_total: int,
) -> dict[str, Any]:
    """Aggregate deterministic and LLM-judge metrics."""

    completed = [
        item for item in results if is_complete_result(item)
    ]
    failed = [
        item for item in results if not is_complete_result(item)
    ]

    summary: dict[str, Any] = {
        "expected_questions": expected_total,
        "completed_questions": len(completed),
        "failed_questions": len(failed),
    }

    if not completed:
        return summary

    answerable = [item for item in completed if item["answerable"]]
    unanswerable = [item for item in completed if not item["answerable"]]

    citation_validity_rate = fmean(
        float(item["citation_valid"]) for item in completed
    )
    duplicate_free_rate = fmean(
        float(item["citations_duplicate_free"]) for item in completed
    )

    summary.update(
        {
            "answerable_questions": len(answerable),
            "unanswerable_questions": len(unanswerable),
            "citation_validity_rate": citation_validity_rate,
            "citation_validity_percent": citation_validity_rate * 100,
            "duplicate_free_citation_rate": duplicate_free_rate,
            "duplicate_free_citation_percent": duplicate_free_rate * 100,
        }
    )

    if answerable:
        answerable_refusals = sum(
            is_expected_refusal(item["generated_answer"])
            for item in answerable
        )
        answerable_refusal_rate = (
            answerable_refusals / len(answerable)
        )
        answer_success_rate = 1.0 - answerable_refusal_rate

        summary["answerable_refusals"] = answerable_refusals
        summary["answerable_refusal_rate"] = answerable_refusal_rate
        summary["answerable_refusal_percent"] = (
            answerable_refusal_rate * 100
        )
        summary["answer_success_rate"] = answer_success_rate
        summary["answer_success_percent"] = (
            answer_success_rate * 100
        )

    if unanswerable:
        refusal_accuracy = fmean(
            float(item["refusal_correct"]) for item in unanswerable
        )
        summary["refusal_accuracy"] = refusal_accuracy
        summary["refusal_accuracy_percent"] = refusal_accuracy * 100

    judged = [
        item for item in answerable if item.get("judge") is not None
    ]
    summary["judged_answerable_questions"] = len(judged)

    for dimension in JUDGE_DIMENSIONS:
        if not judged:
            continue

        average_score = fmean(
            item["judge"][dimension]["score"] for item in judged
        )
        summary[f"{dimension}_mean_1_to_5"] = average_score
        summary[f"{dimension}_percent"] = average_score / 5.0 * 100

    return summary


def evaluate_one_question(
    question: dict[str, Any],
    retriever: RerankingRetriever,
    generator: Generator,
    judge_client: GeminiClient,
    rate_limiter: APIRateLimiter,
    max_rate_limit_retries: int,
    max_generation_json_attempts: int,
    max_judge_json_attempts: int,
) -> dict[str, Any]:
    """Run retrieval, generation, and evaluation for one question."""

    query = question["query"]
    started_at = time.perf_counter()

    retrieval_started_at = time.perf_counter()
    chunks = retriever.search(query=query, top_k=5)
    retrieval_ms = (
        time.perf_counter() - retrieval_started_at
    ) * 1000

    serialized_chunks = serialize_retrieved_chunks(chunks)

    generation_started_at = time.perf_counter()
    generated = generate_answer_with_retry(
        generator=generator,
        query=query,
        chunks=chunks,
        rate_limiter=rate_limiter,
        max_rate_limit_retries=max_rate_limit_retries,
        max_json_attempts=max_generation_json_attempts,
    )
    generation_ms = (
        time.perf_counter() - generation_started_at
    ) * 1000

    citation_ids = [
        citation.source_id for citation in generated.citations
    ]
    citation_valid = all(
        1 <= source_id <= len(chunks) for source_id in citation_ids
    )
    citations_duplicate_free = (
        len(citation_ids) == len(set(citation_ids))
    )

    result: dict[str, Any] = {
        "id": question["id"],
        "query": query,
        "query_type": question.get("query_type"),
        "difficulty": question.get("difficulty"),
        "answerable": question["answerable"],
        "reference_answer": question["reference_answer"],
        "gold_evidence": question.get("gold_evidence"),
        "relevant_pages": question["relevant_pages"],
        "generated_answer": generated.answer,
        "citation_ids": citation_ids,
        "citation_valid": citation_valid,
        "citations_duplicate_free": citations_duplicate_free,
        "retrieved_chunks": serialized_chunks,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "judge": None,
        "refusal_correct": None,
    }

    if question["answerable"]:
        judge_started_at = time.perf_counter()

        result["judge"] = judge_answer(
            judge_client=judge_client,
            question=question,
            generated_answer=generated.answer,
            retrieved_chunks=serialized_chunks,
            citation_ids=citation_ids,
            rate_limiter=rate_limiter,
            max_rate_limit_retries=max_rate_limit_retries,
            max_json_attempts=max_judge_json_attempts,
        )

        result["judge_ms"] = (
            time.perf_counter() - judge_started_at
        ) * 1000

    else:
        result["refusal_correct"] = (
            is_expected_refusal(generated.answer)
            and len(citation_ids) == 0
        )

    result["total_ms"] = (
        time.perf_counter() - started_at
    ) * 1000
    result["status"] = "completed"

    return result


def evaluate(
    questions_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    limit: int | None,
    judge_model: str,
    min_api_interval: float,
    max_rate_limit_retries: int,
    max_generation_json_attempts: int,
    max_judge_json_attempts: int,
    resume: bool,
) -> dict[str, Any]:
    """
    Run resumable end-to-end generation evaluation.

    Progress is checkpointed after every question. HTTP 429 failures are
    retried automatically, and rerunning the script resumes completed work.
    """

    questions = load_questions(questions_path)

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than 0.")

        questions = questions[:limit]

    configuration = {
        "retriever": "hybrid_bm25_dense_rrf_cross_encoder",
        "rerank_candidates": 30,
        "final_top_k": 5,
        "generator_model": "gemini-2.5-flash",
        "judge_model": judge_model,
        "judge_scale": "1-5",
        "min_api_interval_seconds": min_api_interval,
        "max_rate_limit_retries": max_rate_limit_retries,
        "max_generation_json_attempts": max_generation_json_attempts,
        "max_judge_json_attempts": max_judge_json_attempts,
    }

    result_map: dict[str, dict[str, Any]] = {}

    if resume:
        result_map = load_existing_results(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
        )

    selected_ids = {item["id"] for item in questions}
    result_map = {
        question_id: result
        for question_id, result in result_map.items()
        if question_id in selected_ids
    }

    retriever = RerankingRetriever(rerank_candidates=30)
    generator = Generator(llm_client=GeminiClient())
    judge_client = GeminiClient(model_name=judge_model)
    rate_limiter = APIRateLimiter(min_api_interval)

    for index, question in enumerate(questions, start=1):
        question_id = question["id"]
        existing = result_map.get(question_id)

        if (
            existing is not None
            and is_complete_result(existing)
        ):
            print(
                f"[{index}/{len(questions)}] {question_id}: "
                "SKIP (already completed)"
            )
            continue

        print(
            f"[{index}/{len(questions)}] {question_id}: "
            f"{question['query']}"
        )

        try:
            result_map[question_id] = evaluate_one_question(
                question=question,
                retriever=retriever,
                generator=generator,
                judge_client=judge_client,
                rate_limiter=rate_limiter,
                max_rate_limit_retries=max_rate_limit_retries,
                max_generation_json_attempts=max_generation_json_attempts,
                max_judge_json_attempts=max_judge_json_attempts,
            )
            print(f"  Saved {question_id}")

        except KeyboardInterrupt:
            print(
                "\nInterrupted by user. Saving checkpoint before exit."
            )
            atomic_write_json(
                checkpoint_path,
                build_checkpoint(
                    configuration=configuration,
                    questions=questions,
                    result_map=result_map,
                ),
            )
            raise

        except Exception as exc:
            print(
                f"  ERROR {question_id}: "
                f"{type(exc).__name__}: {exc}"
            )

            result_map[question_id] = {
                "id": question_id,
                "query": question["query"],
                "answerable": question["answerable"],
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        atomic_write_json(
            checkpoint_path,
            build_checkpoint(
                configuration=configuration,
                questions=questions,
                result_map=result_map,
            ),
        )

    ordered_results = [
        result_map[item["id"]]
        for item in questions
        if item["id"] in result_map
    ]

    summary = calculate_summary(
        results=ordered_results,
        expected_total=len(questions),
    )

    report = {
        "configuration": configuration,
        "summary": summary,
        "results": ordered_results,
    }

    atomic_write_json(output_path, report)
    return report


def print_summary(summary: dict[str, Any]) -> None:
    """Print the main benchmark metrics."""

    print("\nGENERATION EVALUATION SUMMARY")
    print("=" * 40)
    print(f"Expected questions: {summary['expected_questions']}")
    print(f"Completed questions: {summary['completed_questions']}")
    print(f"Failed questions: {summary['failed_questions']}")

    if "citation_validity_percent" in summary:
        print(
            "Citation validity: "
            f"{summary['citation_validity_percent']:.2f}%"
        )

    if "duplicate_free_citation_percent" in summary:
        print(
            "Duplicate-free citations: "
            f"{summary['duplicate_free_citation_percent']:.2f}%"
        )

    if "answer_success_percent" in summary:
        print(
            "Answer success (answerable questions): "
            f"{summary['answer_success_percent']:.2f}%"
        )
        print(
            "Answerable refusals: "
            f"{summary['answerable_refusals']} "
            f"({summary['answerable_refusal_percent']:.2f}%)"
        )

    if "refusal_accuracy_percent" in summary:
        print(
            "Refusal accuracy (unanswerable questions): "
            f"{summary['refusal_accuracy_percent']:.2f}%"
        )

    for dimension in JUDGE_DIMENSIONS:
        key = f"{dimension}_mean_1_to_5"

        if key in summary:
            print(
                f"{dimension.replace('_', ' ').title()}: "
                f"{summary[key]:.2f}/5 "
                f"({summary[f'{dimension}_percent']:.2f}%)"
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate end-to-end RAG generation quality with "
            "rate-limit handling, checkpointing, and resume support."
        )
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("evaluation/questions_generation_eval.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/generation_results.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("evaluation/results/generation_checkpoint.json"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--judge-model",
        default="gemini-2.5-flash",
    )
    parser.add_argument(
        "--min-api-interval",
        type=float,
        default=1.0,
        help=(
            "Minimum seconds between Gemini API calls. "
            "Default is 1 second for a billing-enabled project. "
            "HTTP 429 responses are still retried automatically."
        ),
    )
    parser.add_argument(
        "--max-rate-limit-retries",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-generation-json-attempts",
        type=int,
        default=3,
        help=(
            "Retry answer generation when the LLM returns malformed "
            "structured JSON."
        ),
    )
    parser.add_argument(
        "--max-judge-json-attempts",
        type=int,
        default=3,
        help=(
            "Retry the LLM judge when it returns malformed JSON."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoint/output and start from scratch.",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    load_dotenv()
    args = parse_args()

    report = evaluate(
        questions_path=args.questions,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        limit=args.limit,
        judge_model=args.judge_model,
        min_api_interval=args.min_api_interval,
        max_rate_limit_retries=args.max_rate_limit_retries,
        max_generation_json_attempts=args.max_generation_json_attempts,
        max_judge_json_attempts=args.max_judge_json_attempts,
        resume=not args.no_resume,
    )

    print_summary(report["summary"])

    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Final results: {args.output}")


if __name__ == "__main__":
    main()
