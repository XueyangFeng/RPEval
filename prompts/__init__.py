from .prompts import *

try:
    Personalized_rule_cot_mcq_prompt
except NameError:
    Personalized_rule_cot_mcq_prompt = cot_reasoner_prompt
