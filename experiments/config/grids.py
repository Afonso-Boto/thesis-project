"""
All W configurations used across the five experiment domains.
"""
from llminferchar.layers.workload.spec import BaseContextProfile, OutputSpec, WorkloadSpec, AugmentationProfile, \
    SchemaAugProfile, ResultAugProfile, TokenDistribution

# Shared defaults — held constant across primary calibration grid
_BASE = BaseContextProfile.default_rag()
_OSPEC = OutputSpec.default_rag()
_N = 100
_T = "single_agent_rag"
_K = 1
_A = "sequential"
_SEED = 42

# PRIMARY GRID — chunks × chunk_size (Domains 1 and 2)

# Chunk counts from RAGN profile: {0, 3, 5, 10, 20}
# Chunk sizes: {256, 400} tokens
# Total: 10 configuration points × 100 sessions = 1,000 executions per regime

PRIMARY_GRID: list[WorkloadSpec] = []

for _chunks in [0, 3, 5, 10, 20]:
    for _chunk_tokens in [256, 400]:

        if _chunks == 0 and _chunk_tokens == 400:
            continue

        _aug = (
            AugmentationProfile.baseline()
            if _chunks == 0
            else AugmentationProfile.rag_only(_chunks, _chunk_tokens)
        )

        PRIMARY_GRID.append(WorkloadSpec(
            N=_N,
            T=_T,
            C_base=_BASE,
            C_aug=_aug,
            K=_K,
            O_spec=_OSPEC,
            A=_A,
            workload_profile="RAGN",
            config_label=f"chunks{_chunks}_tok{_chunk_tokens}",
            seed=_SEED,
        ))

BASELINE_CONFIG = PRIMARY_GRID[0]

# SCHEMA GRID — C_schema isolation experiment (Domain 3)

SCHEMA_GRID: list[WorkloadSpec] = []

for _num_tools in [0, 1, 3, 5, 10, 20]:
    _tpt_values = [0] if _num_tools == 0 else [50, 150, 300]

    for _tokens_per_tool in _tpt_values:
        _aug = AugmentationProfile(
            schema=SchemaAugProfile.mcp_tools(_num_tools, _tokens_per_tool),
            result=ResultAugProfile.no_retrieval(),
        )
        SCHEMA_GRID.append(WorkloadSpec(
            N=_N,
            T=_T,
            C_base=_BASE,
            C_aug=_aug,
            K=_K,
            O_spec=_OSPEC,
            A=_A,
            workload_profile="RAGN",
            config_label=f"tools{_num_tools}_schematok{_tokens_per_tool}",
            seed=_SEED,
        ))

# OUTPUT GRID — O_spec variation experiment (Domain 4)
_AUG_FIXED = AugmentationProfile.rag_only(chunks=10, tokens_per_chunk=400)

OUTPUT_GRID: list[WorkloadSpec] = []

for _output_mean in [50, 200, 500, 1000, 2000]:
    OUTPUT_GRID.append(WorkloadSpec(
        N=_N,
        T=_T,
        C_base=_BASE,
        C_aug=_AUG_FIXED,
        K=_K,
        O_spec=OutputSpec(
            output_length_dist=TokenDistribution.fixed(_output_mean),
            output_structure="free_form",
        ),
        A=_A,
        workload_profile="RAGN",
        config_label=f"output{_output_mean}_chunks10_tok400",
        seed=_SEED,
    ))

# BATCH GRID — B-layer throughput plateau experiments (Domain 5)
BATCH_GRID: list[WorkloadSpec] = []

for _chunks in [0, 3, 5, 10, 20]:
    _aug = (
        AugmentationProfile.baseline()
        if _chunks == 0
        else AugmentationProfile.rag_only(_chunks, tokens_per_chunk=400)
    )

    BATCH_GRID.append(WorkloadSpec(
        N=1,
        T=_T,
        C_base=_BASE,
        C_aug=_aug,
        K=_K,
        O_spec=_OSPEC,
        A="concurrent",
        concurrency_n=1,
        workload_profile="RAGN",
        config_label=f"batch_chunks{_chunks}_tok400",
        seed=_SEED,
    ))
