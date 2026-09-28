"""
Generates concrete text strings for each SessionTrace produced by the X layer.
"""

from __future__ import annotations

import random

from llminferchar.layers.workload.content_source import top_up_to_tokens, get_content_source, active_content_manifest
from llminferchar.layers.workload.verifier import (
    TokenizerWrapper,
    SessionTextSpec,
    WorkloadVerifier,
)
from llminferchar.layers.translation.spec import SessionTrace
from llminferchar.layers.workload.spec import WorkloadSpec


class GenerationError(Exception):
    """
    Raised when the generator cannot produce text hitting the target token
    count within max_attempts regeneration attempts.
    """
    pass


# Tokenization ratio calibration
_CHARS_PER_TOKEN_PROSE = 6.13
_CHARS_PER_TOKEN_INSTRUCTION = 6.32
_CHARS_PER_TOKEN_JSON = 4.33


# Content pools — realistic text for environmental research context
_SYSTEM_PROMPT_TEMPLATES = [
    (
        "You are a helpful research assistant specializing in environmental science "
        "and climate data. Answer the user's question based solely on the provided "
        "context documents. Be concise, accurate, and cite specific details from "
        "the context when relevant. If the answer is not contained in the provided "
        "context, state clearly that you cannot determine the answer from the "
        "available information. Do not speculate or introduce information not "
        "present in the context."
    ),
    (
        "You are an expert assistant supporting environmental researchers. Your task "
        "is to answer questions using only the information provided in the context "
        "below. Provide precise, scientifically accurate responses. When referencing "
        "specific measurements or datasets, include the relevant units and timeframes. "
        "If the context does not contain sufficient information to answer the question, "
        "acknowledge this limitation explicitly rather than guessing."
    ),
    (
        "You are a scientific data assistant for the ENVRI research community. "
        "Answer questions about environmental datasets, essential climate variables, "
        "and research methodologies based strictly on the provided context. Prioritize "
        "accuracy over completeness. When multiple sources in the context provide "
        "relevant information, synthesize them coherently. Indicate any uncertainties "
        "or limitations in the available data."
    ),
]


# Query templates — realistic environmental research questions
_QUERY_TEMPLATES = [
    "What datasets does the ENVRI community provide for computing essential climate variables?",
    "How can sea surface temperature data be used to measure climate change indicators?",
    "What are the primary sources of uncertainty in atmospheric CO2 measurements?",
    "Which biodiversity observation networks contribute data to the ENVRI platform?",
    "How do I access long-term ocean salinity records for climate trend analysis?",
    "What is the spatial resolution of the available land surface temperature datasets?",
    "Can you explain the methodology used to derive soil moisture ECVs from satellite data?",
    "What temporal coverage is available for the provided atmospheric aerosol measurements?",
    "How are the ocean heat content estimates calculated from the available Argo float data?",
    "What quality control procedures are applied to the in situ precipitation measurements?",
    "Which datasets are most suitable for studying Arctic sea ice extent variability?",
    "How can I combine multiple ECV datasets to study compound climate events?",
    "What are the known biases in the provided reanalysis datasets for wind speed?",
    "Can you describe the data format and access protocols for the provided datasets?",
    "What is the relationship between the available datasets and the GCOS ECV requirements?",
    "How do the spatial and temporal resolutions of available datasets affect trend detection?",
    "What validation studies have been conducted for the provided satellite-derived ECVs?",
    "Which research groups contribute observations to the environmental monitoring network?",
]

# Query padding phrases — appended to reach target length
_QUERY_PADDING = [
    "Please provide details about data availability and access methods.",
    "Include information about temporal coverage and spatial resolution.",
    "Describe any known limitations or quality flags in the dataset.",
    "Explain how this data relates to broader climate monitoring objectives.",
    "Specify the relevant units, reference periods, and measurement techniques.",
]

_SCHEMA_TEMPLATES = [
    '{{"name": "{name}", "description": "{desc}", "parameters": {{"type": "object", '
    '"properties": {{"query": {{"type": "string", "description": "Search query string"}}}}, '
    '"required": ["query"]}}}}',

    '{{"name": "{name}", "description": "{desc}", "parameters": {{"type": "object", '
    '"properties": {{"dataset_id": {{"type": "string", "description": "Dataset identifier"}}, '
    '"format": {{"type": "string", "enum": ["json", "csv", "netcdf"]}}}}, '
    '"required": ["dataset_id"]}}}}',

    '{{"name": "{name}", "description": "{desc}", "parameters": {{"type": "object", '
    '"properties": {{"variable": {{"type": "string"}}, "start_date": {{"type": "string", '
    '"format": "date"}}, "end_date": {{"type": "string", "format": "date"}}, '
    '"bbox": {{"type": "array", "items": {{"type": "number"}}, "minItems": 4, "maxItems": 4}}}}, '
    '"required": ["variable", "start_date", "end_date"]}}}}',
]

_TOOL_NAMES = [
    "search_datasets",
    "query_timeseries",
    "fetch_station_metadata",
    "list_variables",
    "retrieve_documents",
    "aggregate_measurements",
]

_TOOL_DESCRIPTIONS = [
    "Search the catalogue for datasets matching a query",
    "Query a time series for a variable over a time range",
    "Fetch metadata for an observation station",
    "List the variables available in a dataset",
    "Retrieve documents relevant to a query from the archive",
    "Aggregate measurements over a spatial or temporal window",
]

def _sample_paragraphs_for_chars(target_chars: int,
                                 rng: random.Random) -> list[str]:
    """
    Sample enough corpus paragraphs (without replacement within the
    draw) to cover target_chars, with headroom for trimming.
    """
    source = get_content_source()
    approx_per_para = 600
    k = max(3, (target_chars // approx_per_para) + 3)
    return source.paragraphs(rng, k)

def _build_text_to_token_count(
    base_paragraphs: list[str],
    target_chars: int,
    rng: random.Random,
) -> str:
    """
    Build a text string of approximately target_chars characters by
    sampling and concatenating paragraphs from base_paragraphs.
    """
    order = list(range(len(base_paragraphs)))
    rng.shuffle(order)

    parts: list[str] = []
    total = 0
    for idx in order:
        para = base_paragraphs[idx]
        if total + len(para) <= target_chars:
            parts.append(para)
            total += len(para) + 2
            continue
        remaining = target_chars - total
        if remaining > 20:
            words = para.split()
            partial: list[str] = []
            used = 0
            for w in words:
                if used + len(w) + 1 > remaining:
                    break
                partial.append(w)
                used += len(w) + 1
            if partial:
                parts.append(" ".join(partial))
        break

    if not parts and base_paragraphs:
        parts.append(base_paragraphs[order[0]][: max(20, target_chars)])
    return "\n\n".join(parts)


def _estimate_chars_for_tokens(
    target_tokens: int,
    chars_per_token: float,
) -> int:
    """
    Estimate character count needed to produce approximately target_tokens.
    """
    return max(1, int(target_tokens * chars_per_token))


def _adjust_text_to_tokens(
    text: str,
    target_tokens: int,
    tokenizer: TokenizerWrapper,
    filler_paragraphs: list[str],
    rng: random.Random,
    tolerance: float = 0.05,
) -> str:
    """
       Length adjustment: exact truncation when over
    """

    if target_tokens == 0:
        return ""
    return top_up_to_tokens(
        text=text,
        target_tokens=target_tokens,
        tokenizer=tokenizer,
        filler_paragraphs=filler_paragraphs,
        rng=rng,
        tolerance=tolerance,
        exact=False,
    )



def _generate_system_prompt(
    target_tokens: int,
    tokenizer: TokenizerWrapper,
    rng: random.Random,
    tolerance: float = 0.05,
) -> str:
    """
    Generate a system prompt of approximately target_tokens tokens.
    """
    if target_tokens == 0:
        return ""

    base = rng.choice(_SYSTEM_PROMPT_TEMPLATES)
    fillers = _sample_paragraphs_for_chars(
        _estimate_chars_for_tokens(target_tokens, _CHARS_PER_TOKEN_PROSE),
        rng,
    )
    return _adjust_text_to_tokens(
        text=base,
        target_tokens=target_tokens,
        tokenizer=tokenizer,
        filler_paragraphs=fillers,
        rng=rng,
        tolerance=tolerance,
    )


def _generate_query(
        target_tokens: int,
        tokenizer: TokenizerWrapper,
        rng: random.Random,
        tolerance: float = 0.05,
) -> str:
    """
    Query: one content-neutral stem, padded to target with corpus
    sentences when the target exceeds the stem.
    """
    if target_tokens == 0:
        return ""

    base = rng.choice(_QUERY_TEMPLATES)
    fillers = _sample_paragraphs_for_chars(
        _estimate_chars_for_tokens(target_tokens, _CHARS_PER_TOKEN_PROSE),
        rng,
    )
    return _adjust_text_to_tokens(
        text=base,
        target_tokens=target_tokens,
        tokenizer=tokenizer,
        filler_paragraphs=fillers,
        rng=rng,
        tolerance=tolerance,
    )


def _generate_chunk(
        target_tokens: int,
        tokenizer: TokenizerWrapper,
        rng: random.Random,
        tolerance: float = 0.05,
) -> str:
    """
    Retrieval chunk: corpus paragraphs assembled to the char budget,
    then adjusted to the token target.
    """
    if target_tokens == 0:
        return ""

    target_chars = _estimate_chars_for_tokens(
        target_tokens, _CHARS_PER_TOKEN_PROSE
    )
    paragraphs = _sample_paragraphs_for_chars(target_chars, rng)
    text = _build_text_to_token_count(paragraphs, target_chars, rng)

    return _adjust_text_to_tokens(
        text=text,
        target_tokens=target_tokens,
        tokenizer=tokenizer,
        filler_paragraphs=paragraphs,
        rng=rng,
        tolerance=tolerance,
    )


def _generate_schema_string(
        target_tokens: int,
        tool_index: int,
        tokenizer: TokenizerWrapper,
        rng: random.Random,
        tolerance: float = 0.05,
) -> str:
    """
    MCP-style tool schema string of approximately target_tokens tokens.
    Structural JSON
    """
    if target_tokens == 0:
        return ""

    name_idx = tool_index % len(_TOOL_NAMES)
    desc_idx = tool_index % len(_TOOL_DESCRIPTIONS)
    template_idx = tool_index % len(_SCHEMA_TEMPLATES)

    name = _TOOL_NAMES[name_idx]
    desc = _TOOL_DESCRIPTIONS[desc_idx]
    template = _SCHEMA_TEMPLATES[template_idx]

    base_schema = template.format(name=name, desc=desc)

    actual = tokenizer.count_tokens(base_schema)
    error = (actual - target_tokens) / target_tokens if target_tokens > 0 else 0

    if abs(error) <= tolerance:
        return base_schema

    if actual > target_tokens:
        return tokenizer.truncate_to_tokens(base_schema, target_tokens)

    additional_params = [
        '"time_range": {"type": "string", "description": "ISO 8601 time range specification"}',
        '"spatial_resolution": {"type": "number", "description": "Spatial resolution in degrees"}',
        '"variables": {"type": "array", "items": {"type": "string"}, "description": "List of variable names"}',
        '"quality_flag": {"type": "integer", "minimum": 0, "maximum": 3, "description": "Data quality threshold"}',
        '"output_format": {"type": "string", "enum": ["json", "csv", "netcdf", "zarr"], "description": "Output file format"}',
        '"max_results": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "Maximum number of results"}',
        '"include_metadata": {"type": "boolean", "description": "Whether to include dataset metadata"}',
        '"aggregation": {"type": "string", "enum": ["mean", "sum", "min", "max"], "description": "Temporal aggregation method"}',
    ]

    params_str = ", ".join(
        ['"query": {"type": "string", "description": "Search query"}']
        + additional_params[:6]
    )
    expanded = (
        f'{{"name": "{name}", "description": "{desc} with extended parameter support '
        f'for advanced data retrieval and filtering capabilities.", '
        f'"parameters": {{"type": "object", "properties": {{{params_str}}}, '
        f'"required": ["query"]}}}}'
    )

    if tokenizer.count_tokens(expanded) < target_tokens:
        blocks = [expanded]
        while tokenizer.count_tokens("\n".join(blocks)) < target_tokens:
            blocks.append(expanded)
            if len(blocks) > 200:
                break
        expanded = "\n".join(blocks)

    return tokenizer.truncate_to_tokens(expanded, target_tokens) \
        if tokenizer.count_tokens(expanded) > target_tokens else expanded

def generate_session(
        trace: SessionTrace,
        tokenizer: TokenizerWrapper,
        W: WorkloadSpec,
        max_attempts: int = 5,
        tolerance: float = 0.05,
        verbose: bool = False,
) -> SessionTextSpec:
    """
    Generate a complete SessionTextSpec for one SessionTrace.
    """
    if not trace.is_feasible:
        raise GenerationError(
            f"Cannot generate content for session {trace.session_id}: "
            f"pre_execution_status={trace.pre_execution_status}. "
            f"Context overflow sessions are not submitted to E."
        )

    step = trace.step_1

    rng = random.Random(trace.seed)

    system_prompt = _generate_with_retry(
        generator=lambda: _generate_system_prompt(
            target_tokens=W.C_base.system_prompt_tokens,
            tokenizer=tokenizer,
            rng=rng,
            tolerance=tolerance,
        ),
        tokenizer=tokenizer,
        target_tokens=W.C_base.system_prompt_tokens,
        role="system_prompt",
        max_attempts=max_attempts,
        tolerance=tolerance,
    )
    query_target = step.base_tokens - W.C_base.system_prompt_tokens
    query_target = max(1, query_target)  # guard against zero

    query = _generate_with_retry(
        generator=lambda: _generate_query(
            target_tokens=query_target,
            tokenizer=tokenizer,
            rng=rng,
            tolerance=tolerance,
        ),
        tokenizer=tokenizer,
        target_tokens=query_target,
        role="query",
        max_attempts=max_attempts,
        tolerance=tolerance,
    )

    chunks: list[str] = []
    chunks_per_step = W.C_aug.result.chunks_per_step
    tokens_per_chunk = int(W.C_aug.result.tokens_per_chunk_dist.mean_tokens)

    for i in range(chunks_per_step):
        chunk = _generate_with_retry(
            generator=lambda i=i: _generate_chunk(
                target_tokens=tokens_per_chunk,
                tokenizer=tokenizer,
                rng=random.Random(trace.seed + i + 1),
                tolerance=tolerance,
            ),
            tokenizer=tokenizer,
            target_tokens=tokens_per_chunk,
            role=f"chunk_{i}",
            max_attempts=max_attempts,
            tolerance=tolerance,
        )
        chunks.append(chunk)

    schema_strings: list[str] = []
    num_tools = W.C_aug.schema.num_tools
    schema_tokens_per_tool = W.C_aug.schema.schema_tokens_per_tool

    for i in range(num_tools):
        schema = _generate_with_retry(
            generator=lambda i=i: _generate_schema_string(
                target_tokens=schema_tokens_per_tool,
                tool_index=i,
                tokenizer=tokenizer,
                rng=random.Random(trace.seed + 1000 + i),
                tolerance=tolerance,
            ),
            tokenizer=tokenizer,
            target_tokens=schema_tokens_per_tool,
            role=f"schema_tool_{i}",
            max_attempts=max_attempts,
            tolerance=tolerance,
        )
        schema_strings.append(schema)

    spec = SessionTextSpec(
        session_id=trace.session_id,
        system_prompt=system_prompt,
        query=query,
        chunks=chunks,
        schema_strings=schema_strings,
        target_system_prompt_tokens=W.C_base.system_prompt_tokens,
        target_query_tokens=query_target,
        target_tokens_per_chunk=tokens_per_chunk,
        target_schema_tokens_per_tool=schema_tokens_per_tool,
    )

    verifier = WorkloadVerifier(tokenizer, tolerance=tolerance)
    all_passed, results = verifier.verify_and_report(
        spec=spec,
        target_prefill_tokens=step.prefill_tokens,
        verbose=verbose,
    )

    if not all_passed:
        failed = [r for r in results if not r.passed]
        failed_roles = [r.text_role for r in failed]
        full_prompt_result = results[-1]
        if not full_prompt_result.passed:
            raise GenerationError(
                f"Session {trace.session_id}: full_prompt verification failed "
                f"after generation. Failed components: {failed_roles}. "
                f"full_prompt: target={full_prompt_result.target_tokens} "
                f"actual={full_prompt_result.actual_tokens} "
                f"error={full_prompt_result.error_fraction:+.1%}. "
                f"Consider increasing max_attempts or widening tolerance."
            )

    return spec

def _generate_with_retry(
        generator,
        tokenizer: TokenizerWrapper,
        target_tokens: int,
        role: str,
        max_attempts: int,
        tolerance: float,
) -> str:
    if target_tokens == 0:
        return ""

    for attempt in range(max_attempts):
        text = generator()
        actual = tokenizer.count_tokens(text)
        error = (abs(actual - target_tokens) / target_tokens
                 if target_tokens > 0 else 0)
        if error <= tolerance:
            return text

    text = generator()
    actual = tokenizer.count_tokens(text)
    if actual > int(target_tokens * (1 + tolerance)):
        text = tokenizer.truncate_to_tokens(text, target_tokens)
        actual = tokenizer.count_tokens(text)

    error = (abs(actual - target_tokens) / target_tokens
             if target_tokens > 0 else 0)
    if error > tolerance:
        raise GenerationError(
            f"Failed to generate {role} within ±{tolerance:.0%} tolerance "
        )

    return text


def generate_workload(
        traces: list[SessionTrace],
        tokenizer: TokenizerWrapper,
        W: WorkloadSpec,
        max_attempts: int = 5,
        tolerance: float = 0.05,
        verbose: bool = False,
) -> tuple[list[tuple[SessionTrace, SessionTextSpec]], list[str]]:
    """
    Generate SessionTextSpec objects for all feasible sessions in a
    workload.
    """
    pairs: list[tuple[SessionTrace, SessionTextSpec]] = []
    failed_ids: list[str] = []

    feasible = [t for t in traces if t.is_feasible]
    skipped = len(traces) - len(feasible)

    if skipped > 0:
        print(f"  Skipping {skipped} context_overflow sessions")

    for i, trace in enumerate(feasible):
        try:
            spec = generate_session(
                trace=trace,
                tokenizer=tokenizer,
                W=W,
                max_attempts=max_attempts,
                tolerance=tolerance,
                verbose=verbose,
            )
            pairs.append((trace, spec))
        except GenerationError as e:
            print(f"  GenerationError for {trace.session_id}: {e}")
            failed_ids.append(trace.session_id)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(feasible)} sessions...")

    print(
        f"Generation complete: {len(pairs)}/{len(feasible)} succeeded, "
        f"{len(failed_ids)} failed"
    )

    return pairs, failed_ids

def calibrate_chars_per_token(
        tokenizer: TokenizerWrapper,
        sample_size: int = 5,
) -> dict[str, float]:
    """
    Measure the actual chars_per_token ratio for this tokenizer on
    samples of the three content classes the generator emits
    """
    global _CHARS_PER_TOKEN_PROSE, _CHARS_PER_TOKEN_INSTRUCTION, \
        _CHARS_PER_TOKEN_JSON

    rng = random.Random(0)
    prose_samples = get_content_source().paragraphs(
        rng, max(sample_size, 5))
    instr_samples = list(_SYSTEM_PROMPT_TEMPLATES) + _QUERY_TEMPLATES[:2]
    json_samples = [
        t.format(name=n, desc=d)
        for t, n, d in zip(
            _SCHEMA_TEMPLATES * len(_TOOL_NAMES),
            _TOOL_NAMES,
            _TOOL_DESCRIPTIONS,
        )
    ]

    def ratio(samples: list[str]) -> float:
        chars = sum(len(s) for s in samples)
        toks = sum(tokenizer.count_tokens(s) for s in samples)
        return chars / toks if toks else 0.0

    _CHARS_PER_TOKEN_PROSE = ratio(prose_samples)
    _CHARS_PER_TOKEN_INSTRUCTION = ratio(instr_samples)
    _CHARS_PER_TOKEN_JSON = ratio(json_samples)

    manifest = active_content_manifest()
    result = {
        "prose": round(_CHARS_PER_TOKEN_PROSE, 3),
        "instruction": round(_CHARS_PER_TOKEN_INSTRUCTION, 3),
        "json": round(_CHARS_PER_TOKEN_JSON, 3),
        "content_source": manifest.get("source_id", "unknown"),
    }
    print(f"  chars/token calibration: prose={result['prose']} "
          f"instruction={result['instruction']} json={result['json']} "
          f"(corpus: {result['content_source']})")
    return result
