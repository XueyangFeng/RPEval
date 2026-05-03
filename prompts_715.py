from langchain.prompts import PromptTemplate

try:
    from prompts.prompts_715 import *  # noqa: F401,F403
except NameError:
    # The archived prompt file contains a dangling optional template reference.
    # Keep the benchmark entry points importable; the commonly used templates
    # below are defined explicitly for evaluation scripts.
    pass

_ranking_template = """
Given the user preference(s) and query, rank the utilization policies from most
to least appropriate.

Policies:
A = Ignore the preference.
B = Use the preference as supporting context.
C = Let the preference dominate the response.

Preference(s):
{persona}

Query:
{question}

Return JSON with a single key "ranking", for example: {{"ranking": "ABC"}}.
"""

_multi_ranking_template = """
For each user preference, rank the utilization policies from most to least
appropriate for the query.

Policies:
A = Ignore the preference.
B = Use the preference as supporting context.
C = Let the preference dominate the response.

Preferences:
{personas}

Query:
{question}

Return JSON with rankings aligned to the preference order.
"""

_single_generation_template = """
Generate a response to the user query using the selected personalization policy.

Preference:
{persona}

Query:
{question}
"""

_multi_generation_template = """
Generate a response to the user query using the provided preferences only when
they are appropriate.

Preferences:
{personas}

Query:
{question}
"""

_judge_template = """
Evaluate whether the reply matches the ground-truth personalization intent.

Preference(s):
{persona}

Query:
{question}

Reply:
{reply}

Ground-truth intent:
{intent}

Intent type:
{intent_type}

Return JSON scores for OPB, UPB, RII, FM, VG, and Judge on a 0-5 scale.
"""

MLE_estimation_template_v1 = PromptTemplate(
    input_variables=["persona", "question"],
    template=_ranking_template,
)
MLE_estimation_template_v2 = PromptTemplate(
    input_variables=["persona", "question"],
    template=_ranking_template,
)
CPE_recall_estimation_prompt = PromptTemplate(
    input_variables=["persona", "question"],
    template=_ranking_template,
)
CPE_intent_estimation_prompt = PromptTemplate(
    input_variables=["persona", "question"],
    template=_ranking_template,
)

MLE_estimation_prompt_mmcq_template_v0 = PromptTemplate(
    input_variables=["personas", "question"],
    template=_multi_ranking_template,
)
CPE_estimation_prompt_mmcq_template_v0 = PromptTemplate(
    input_variables=["personas", "question"],
    template=_multi_ranking_template,
)
MLE_estimation_prompt_mmcq_implicit_template_v0 = PromptTemplate(
    input_variables=["personas", "question"],
    template=_multi_ranking_template,
)
CPE_estimation_prompt_mmcq_implicit_template_v0 = PromptTemplate(
    input_variables=["personas", "question"],
    template=_multi_ranking_template,
)

RPA_generation_template = PromptTemplate(
    input_variables=["persona", "question"],
    template=_single_generation_template,
)
RPA_multi_generation_template = PromptTemplate(
    input_variables=["personas", "question"],
    template=_multi_generation_template,
)

LLM_judge_template = PromptTemplate(
    input_variables=["persona", "question", "reply", "intent", "intent_type"],
    template=_judge_template,
)
LLM_judge_multiple_template = PromptTemplate(
    input_variables=["persona", "question", "reply", "intent", "intent_type"],
    template=_judge_template,
)
