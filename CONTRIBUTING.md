# Contributing to BESS-Bench

Thanks for considering a contribution. BESS-Bench is a **frozen
benchmark**: the dataset, splits, labels, metrics and significance
rules do not change except at tagged major releases. Contributions
fall into one of two categories:

1. **Bug fixes** in evaluation or baseline code.
2. **v1.x protocol clarifications** (wording only, never numbers).

Please read `EVALUATION_PROTOCOL.md` before anything else.

---

## 1. Bug fixes

Bug = the code does not do what `EVALUATION_PROTOCOL.md` says it does.
Open an issue that quotes the protocol and the observed behaviour, then
a PR. We will bump a patch version (`v1.0.x`) and re-run every
baseline on the fixed code.

---

## 2. Protocol clarifications (v1.x)

If a protocol sentence is ambiguous and readers interpret it two ways,
we may issue a clarification in `EVALUATION_PROTOCOL.md` that *removes
the ambiguity* without changing any number anyone could have reported
under the charitable reading. Patch-version bump, no result churn.

Changes that would alter numbers (new splits, new metric, new
significance rule) are **v2.0** and require a new paper submission.

---

## 3. What we will **not** accept

- Changes to `splits.csv` (this is part of the dataset release).
- New tasks. If you have a new task in mind, open a discussion for
  v2.0 planning.
- Data files larger than 50 MB — they go on HF, not in git.

---

## 4. Code style

- Python 3.10+, `ruff` + `black` (configs in `pyproject.toml`).
- No new top-level dependencies without discussion.
- Scripts live under `stats/`, `downstream/`, `stage1_encoder/`,
  or `scripts/` — new top-level directories need a reason.

---

## 5. Licence & attribution

- Code & data: **CC-BY-4.0** (see `LICENSE`).
- Any work derived from BESS-Bench must credit the BeSS consortium
  and cite our paper *and* Neiner et al. 2011 (see `dataset/README.md`).
- By opening a PR you agree that your contribution is licensed under
  the same terms as the files it touches.
