# <img src="assets/acl.png" height="20"> Contaminated Collaboration [![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-1f6feb)](https://aclanthology.org/2026.emnlp-main.1481/)

Code and data for **Contaminated Collaboration: Measuring Gender Bias Transfer in LLM-Assisted Student Writing** (EMNLP 2026).

Ariyan Hossain\*, Kazi Kamruzzaman Rabbi\*, Farig Sadeque, S M Taiabul Haque (\* equal contribution)

We first check that a gender-biased system prompt makes an LLM write gender-differentiated career plans (Phase A), then test whether that bias carries over into essays students write with the assistant (Phase B). Bias is measured two ways: the agentic language gap between essays about male and female subjects, and the Stereotype Congruence Rate (SCR) of the recommended occupations.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Tested with Python 3.13.

## Layout

```
├── common.py              paths, data loading, small helpers
│
├── agency_gap.py          agency score per essay and the agentic gap
├── occupation_labels.py   BLS stereotype labels and occupation matching
├── scr.py                 Stereotype Congruence Rate
├── stats_tests.py         every statistical test in the paper
│
├── train_classifier.py    fine-tune the agency classifier on LAC
├── biographies.py         build / check the stimulus biographies
├── generate_essays.py     Phase A essay generation
│
└── data/
    ├── stimuli/   biographies and names
    ├── prompts/   system prompts and the writing task
    ├── phase_a/   LLM-generated essays, agency scores, occupation labels
    ├── bls/       BLS CPSAAT11 and the occupation overrides
    └── lac/       LAC dataset
```

## Data

| File | Contents |
|---|---|
| `stimuli/biographies.csv` | The four biography pairs shown to participants. `{name}`, `{he}`, `{He}`, `{his}` mark the parts that change between versions. |
| `stimuli/biographies_phase_a.csv` | The templates used to generate the Phase A essays. |
| `stimuli/names.csv` | 100 male and 100 female names |
| `prompts/` | Neutral and biased system prompts and the writing task with its four questions |
| `phase_a/essays.csv` | 1,600 generated essays. `occupation_raw` is the model's answer, `occupation` the cleaned title (first job named, then checked by hand). |
| `phase_a/agency_scores.csv` | Agency score (`per_ac`) of every essay |
| `phase_a/occupation_labels.csv` | BLS match and stereotype label of every essay's occupation |
| `bls/cpsaat11.xlsx` | BLS Current Population Survey, table 11 (2025) |
| `bls/occupation_overrides.csv` | Hand-made mappings for job titles fuzzy matching gets wrong |
| `lac/` | LAC sentences from Wan and Chang (2025); 0 = communal, 1 = agentic |

**Participant data (Phase B) is not shared.** The essays, survey answers and chat logs of the 123 participants are not part of this repository. The Phase B code is included so the analysis can be read and checked; it expects `data/phase_b/participants.csv` (columns `pid, group, bio_id, gender, writer_gender, essay, occupation, n_prompts` and the survey items) and stops with a message if the file is missing.

## Pipeline

```bash
# 1. agency classifier (bert-base-cased on LAC, seed 0)
python train_classifier.py                       # -> checkpoints/agency-bert

# 2. stimuli: check the 2 agentic / 2 communal split of each biography
python biographies.py check
python biographies.py balance --drafts drafts.csv  # optional, rewrites drafts with an LLM

# 3. Phase A essays (needs GROQ_API_KEY)
export GROQ_API_KEY=...
python generate_essays.py                        # -> data/phase_a/essays_generated.csv

# 4. agentic gap (--score runs the classifier; without it the saved scores are used)
python agency_gap.py --phase a --score

# 5. SCR
python occupation_labels.py --phase a
python scr.py --phase a

# 6. tests
python stats_tests.py agency --phase a --tests all
python stats_tests.py scr --phase a
```

`agency_gap.py` and `biographies.py` take `--model` if the checkpoint is somewhere else (for `biographies.py` it goes before the subcommand).

## Statistical tests

`stats_tests.py` takes the metric as a subcommand and runs only the tests you list:

| Subcommand | Tests |
|---|---|
| `agency` | `describe`, `gap` (Welch's t, CI, Cohen's d), `bootstrap`, `anova` (gender x condition), `condition` (per subject gender, with Tukey HSD), `power`, `normality` |
| `scr` | `chi2`, `pairwise` (Fisher's exact, odds ratio, Cohen's h) |
| `interaction` | `prompts`, `likert` (Mann-Whitney U) |

Useful options: `--phase a|b`, `--groups neutral biased` to compare only some conditions, `--writers male` (phase B), and `--threshold 55` for SCR at another cutoff.


## Citation

```bibtex
@inproceedings{hossain-etal-2026-contaminated,
  title     = {Contaminated Collaboration: Measuring Gender Bias Transfer in LLM-Assisted Student Writing},
  author    = {Hossain, Ariyan and Rabbi, Kazi Kamruzzaman and Sadeque, Farig and Haque, S M Taiabul},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  url       = {https://aclanthology.org/2026.emnlp-main.1481/},
  doi       = {10.18653/2026.emnlp-main.1481}
}
```

The agency classifier and LAC dataset come from Wan and Chang (2025), *White Men Lead, Black Women Help? Benchmarking and Mitigating Language Agency Social Biases in LLMs* (ACL 2025).
