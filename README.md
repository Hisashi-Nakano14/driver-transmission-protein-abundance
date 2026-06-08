# Pan-cancer driver-alteration transmission across molecular layers

Analysis code for the manuscript **"Pan-cancer quantification of driver alteration
transmission across molecular layers reveals limited propagation to protein
abundance"** (IJC-26-1558).

The framework quantifies, per driver gene and cancer type, how a somatic alteration
propagates mutation → mRNA (Layer 1) and mRNA → protein (Layer 2) using ordinary
least-squares partial R², and combines them into a Transmission Score
**TS_R² = R²_L1 × R²_L2**.

**Author:** Hisashi Nakano, PhD — Department of Health Data Science, Niigata University
of Health and Welfare, Niigata, Japan · ORCID: 0000-0002-9023-880X

---

## Scripts

Each script writes the CSV/JSON tables that the corresponding manuscript figures and
tables are drawn from; the scripts do not render figures themselves. Figure and table
numbers follow the published manuscript.

| Script | Purpose | Output tables | Manuscript figure/table |
|--------|---------|---------------|-------------------------|
| `run_regression_ts.py` | Main TS_R² pipeline (Layer-1/2/3 OLS, partial R², TS_R² = R²_L1 × R²_L2) | `{cancer}_regression_layer{1,2,3}.csv`, `{cancer}_ts_continuous.csv`, `pancancer_ts_continuous.csv` | Figures 2, 3; Table 1; Table S1 |
| `run_robustness.py` | Gene-class robustness (7-class Kruskal–Wallis + post-hoc), covariate / score-definition sensitivity, permutation null, bootstrap, leave-one-cancer-out | `gene_class_*.csv`, `covariate_sensitivity.csv`, `covariate_rank_correlation.csv`, `permutation_results.csv`, `permutation_null_distribution.csv`, `bootstrap_ci.csv`, `loco_results.csv` | Figures S5 (gene-class), S2 (sensitivity), S1 (permutation null) |
| `run_mixed_model.py` | Variance decomposition (additive two-way ANOVA) and mixed-effects ICC | `variance_decomposition.csv`, `mixed_model_layer2.csv`, `icc_ranking.csv` | Figure 4; Table S3 |
| `run_cna_vartype_survival.py` | CNA transmission, variant-type (missense vs truncating) transmission, and survival (Cox) | `cna_transmission.csv`, `vartype_transmission.csv`, `survival_cox.csv` | Figure 5; Figure S6 (missense vs truncating) |
| `run_msi_pathway.py` | MSI stratification (interaction tests) and pathway cis-vs-trans attenuation | `{cancer}_msi_layer2.csv`, `msi_interaction_tests.csv`, `pathway_transmission.csv` | Figures S7 (MSI), S4 (pathway attenuation) |
| `run_r1_2.py` | Measurement-quality sensitivity | `r1_2_quality_table.csv`, `r1_2_named_examples.csv` | Figure S3; Table S2 |
| `run_r1_3a.py` | Mutation-type stratified Layer-1 sensitivity | `r1_3a_stratified_L1_long.csv`, `r1_3a_pair_summary.csv`, `r1_3a_summary.json` | Table S5 |
| `run_r1_3b.py` | CNA-adjusted Layer-1 sensitivity | `r1_3b_cna_adjusted_L1.csv`, `r1_3b_summary.json` | Table S4 |
| `run_r1_4.py` | MSI/MMR definition sensitivity | `r1_4_results.json` | Results (MSI section) |

> **Note on reproducing aggregate numbers.** Headline aggregate statistics reported in
> the manuscript (e.g. "*N* of *M* pairs", medians, Wilcoxon *p*-values) are computed
> downstream from the per-pair tables above: each script emits the per-pair values and a
> summary CSV/JSON, and the reported aggregates are tallied from those outputs.

---

## Environment

See `requirements.txt` (or `environment.yml`). Core stack:
Python ≥ 3.11, cptac 1.5.14, pandas 2.3.3, numpy 1.26.4, statsmodels 0.14.6,
scipy 1.15.3, lifelines 0.30.3.

```bash
pip install -r requirements.txt
```

## Running

Input matrices are obtained at runtime through the `cptac` package (v1.5.14); no
local data are bundled. Paths are configured through environment variables (defaults
to the current directory):

| Variable | Meaning | Default |
|----------|---------|---------|
| `CPTAC_DATA_DIR` | Directory of the derived per-cohort CSVs that the producers read/write | `.` |
| `CPTAC_OUT_DIR` | Output directory for the revision sensitivity scripts | `.` |

The revision scripts import the producer modules from their own directory, so keep
all scripts together. Example:

```bash
export CPTAC_DATA_DIR=./results
python run_regression_ts.py --all      # produces the pan-cancer TS_R2 tables
python run_r1_3b.py                     # CNA-adjusted Layer-1 sensitivity
```

## Data availability

Proteogenomic data are from the NCI Clinical Proteomic Tumor Analysis Consortium
(CPTAC) and are accessed via the `cptac` Python package (which retrieves them from
the Proteomic Data Commons, PDC). **Raw CPTAC data are not redistributed in this
repository.**

## AI assistance

A large language model (Claude, Anthropic) was used to assist with code documentation
and formatting. All analytical decisions, code logic, and interpretation were made and
verified by the author, who takes full responsibility.

## License

MIT License — see `LICENSE`.
