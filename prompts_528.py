from langchain.prompts import PromptTemplate

_relevance_reasoner_template = """
You are a personalized assistant. Given a user preference and a user query,
decide whether the preference should be used for the current query.

Return one option only:
(A) Ignore
(B) Support
(C) Dominate

Preference:
{persona}

Query:
{question}
"""

_rational_speech_acts_template = """
You are a rational personalized assistant. Infer the user's current intent from
the query and decide whether the given preference should be ignored, used as
supporting context, or used as the dominant constraint.

Return one option only:
(A) Ignore
(B) Support
(C) Dominate

Preference:
{persona}

Query:
{question}
"""

relevance_reasoner_prompt = PromptTemplate(
    input_variables=["persona", "question"],
    template=_relevance_reasoner_template,
)

rational_speech_acts_prompt = PromptTemplate(
    input_variables=["persona", "question"],
    template=_rational_speech_acts_template,
)
