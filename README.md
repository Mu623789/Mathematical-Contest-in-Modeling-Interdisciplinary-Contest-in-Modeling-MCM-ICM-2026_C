# Mathematical-Contest-in-Modeling-Interdisciplinary-Contest-in-Modeling-MCM-ICM-2026_C
This repo contains the implementation for our DWTS study. Eliminations combine observed judges’ scores with unobserved fan votes under season-specific rules. We provide an auditable pipeline that 

(1) infers weekly fan vote shares with uncertainty, 

(2) replays counterfactual aggregation/elimination rules, 

(3) compares how celebrity traits and pro-dancer effects differ between judges and fans, and 

(4) proposes a fairness-oriented mechanism for controversial weeks.

## Contents
Fan vote inference (weekly shares + uncertainty)

Rule simulator (season rules + alternative “what-if” rules)

Effect analysis (celebrity traits vs. pro-dancer effects; judges vs. fans)

Controversy/fairness module (handles unstable or polarized weeks)

## Inputs & Outputs 
Input: season/week-level scores, eliminations, and metadata (celebrity + pro dancer)

Output: inferred fan vote shares, uncertainty intervals, replay results under alternative rules, and effect estimates
