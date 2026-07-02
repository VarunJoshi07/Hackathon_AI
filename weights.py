"""System-wide configuration constants and tunable weights for candidate ranking.

This module acts as the single source of truth for all mathematical parameters,
clamping thresholds, and truncation targets utilized throughout the first-pass
and second-pass ranking pipelines. 

To maintain strict determinism, avoid cross-module side effects, and optimize
import speeds across distributed nodes, this file contains exclusively primitive
constants. It contains no executable code, functions, classes, or dynamic logic.
"""

# ==============================================================================
# FIRST-PASS RANKING WEIGHTS
# ==============================================================================
# Core features are aggregated into a unified base score using a weighted average.
# The following individual feature weights MUST sum to exactly 1.0.

# Highest priority weight because deep embeddings capture semantic intent and
# conceptual alignment between the candidate's profile and the job description,
# bypassing simple keyword mismatches.
EMBEDDING_WEIGHT: float = 0.35

# Second highest weight; ensures strict keyword and exact terminology compliance
# (e.g., specific framework versions, certifications) to complement the broader
# semantic matching of the embedding model.
TFIDF_WEIGHT: float = 0.25

# High importance; explicitly measures the direct intersection of hard/soft skill
# tokens requested in the job description against the candidate's declared profile.
SKILL_WEIGHT: float = 0.20

# Moderate influence; validates professional longevity and level of seniority, 
# preventing entry-level candidates from overriding specialized role requirements
# based on pure textual alignment.
EXPERIENCE_WEIGHT: float = 0.10

# Supporting signal; reflects historical platform-calculated conversion metrics,
# behavioral indicators, and operational likelihood of the candidate interviewing
# or accepting a matching offer.
HIRING_WEIGHT: float = 0.06

# Smallest positive weight; functions as a localized sanity filter to slightly 
# adjust candidate prioritization based on data verification and profile integrity.
CREDIBILITY_WEIGHT: float = 0.04


# ==============================================================================
# SECOND-PASS RERANKING & FUSION WEIGHTS
# ==============================================================================
# Parameters designed for routing and execution inside:
# - src.second_pass.cross_encoder.cross_encoder_score
# - src.second_pass.behavioural.behavioural_score
# - src.second_pass.career_quality.career_quality_score
# - src.second_pass.must_have.must_have_filter
# - src.second_pass.disqualifiers.apply_disqualifiers
# - src.second_pass.fusion.second_pass_fusion

# Tunable weight for src.second_pass.cross_encoder.cross_encoder_score
CROSS_ENCODER_WEIGHT: float = 0.40

# Tunable weight for src.second_pass.career_quality.career_quality_score
CAREER_QUALITY_WEIGHT: float = 0.30

# Tunable weight for src.second_pass.behavioural.behavioural_score
BEHAVIORAL_WEIGHT: float = 0.30

# Operational weight or threshold utilized inside src.second_pass.must_have.must_have_filter
MUST_HAVE_WEIGHT: float = 1.0


# ==============================================================================
# PENALTY & TRAP CONFIGURATIONS
# ==============================================================================
# Penalties represent disqualification vectors, compliance violations, or negative
# signals (e.g., bad formatting, data discrepancies). They are explicitly excluded
# from the normalized weighted average and applied subtractively afterwards.

# Fixed scale coefficient for penalty deduction applied across baseline metrics.
PENALTY_WEIGHT: float = 0.10

# Scale coefficient for penalizing malicious token patterns or impossible profile
# metrics caught during src.second_pass.disqualifiers.apply_disqualifiers
PENALTY_MULTIPLIER: float = 2.5


# ==============================================================================
# PIPELINE RANKING LIMITS
# ==============================================================================
# Maximum allocation boundaries used to slice candidate arrays across processing
# stages, optimizing compute efficiency and memory footprint for 100,000+ records.

# Truncation threshold for the fast, lightweight first-pass filtering stage.
TOP_K_FIRST_PASS: int = 1000

# Truncation threshold for the more computationally intensive second-pass reranking stage.
TOP_K_SECOND_PASS: int = 100

# Final result set limit delivered to downstream clients or rendering engines.
TOP_K_FINAL: int = 100


# ==============================================================================
# SCORE CLAMPING BOUNDS
# ==============================================================================
# Hard boundaries applied to normalized inputs and final aggregated outputs to 
# guarantee predictable value distributions across downstream services.

# Floor boundary for raw, normalized, or fused candidate scores.
MIN_SCORE: float = 0.0

# Ceiling boundary for raw, normalized, or fused candidate scores.
MAX_SCORE: float = 1.0


# ==============================================================================
# NUMERICAL STABILITY
# ==============================================================================

# Tiny fractional buffer utilized during normalization fractions to guarantee 
# structural safety against unexpected zero-sum division errors without shifting
# statistical distributions.
EPSILON: float = 1e-9