#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Variance decomposition (additive two-way ANOVA) and mixed-model ICC
#
# Manuscript: Pan-cancer quantification of driver alteration transmission across
#             molecular layers reveals limited propagation to protein abundance (IJC-26-1558)
# Author:     Hisashi Nakano, PhD - Department of Health Data Science,
#             Niigata University of Health and Welfare, Niigata, Japan
# ORCID:      0000-0002-9023-880X

"""
Phase 2: Mixed Model — Gene-intrinsic vs Cancer-dependent Transmission
======================================================================
Uses Phase 1 regression results (Layer 2: mRNA→Protein) across 7 cancers.

Model: protein_z ~ mrna_z + (mrna_z | cancer_type)
  - Fixed effect β_mrna: gene-intrinsic transmission coefficient
  - Random slope variance: cancer-type dependency
  - ICC: proportion of variance due to cancer type

Plus: Grand variance decomposition via two-way ANOVA.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import os
import cptac
import warnings, sys, io
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.environ.get("CPTAC_DATA_DIR", ".")

# ── Load all cancer data and build long-format table ──────────────────────

def build_long_table():
    """Build long-format table: patient, cancer, gene, mrna_z, protein_z."""
    cancer_classes = {
        'COAD': cptac.Coad, 'LUAD': cptac.Luad, 'UCEC': cptac.Ucec,
        'CCRCC': cptac.Ccrcc, 'OV': cptac.Ov, 'BRCA': cptac.Brca, 'PDAC': cptac.Pdac,
    }

    # Genes present in ≥3 cancers' driver lists
    cancer_drivers = {
        'COAD': ["TP53", "KRAS", "PIK3CA", "SMAD4", "ERBB2", "SOX9", "TCF7L2", "BRAF"],
        'LUAD': ["TP53", "KRAS", "EGFR", "STK11", "KEAP1", "NF1", "BRAF",
                 "RBM10", "SETD2", "ARID1A", "RB1", "ERBB2", "MET", "ATM", "CDKN2A"],
        'UCEC': ["PTEN", "TP53", "PIK3CA", "ARID1A", "CTNNB1", "KRAS", "FGFR2",
                 "PPP2R1A", "PIK3R1", "FBXW7", "RPL22", "CTCF", "ZFHX3", "CHD4", "SPOP"],
        'CCRCC': ["VHL", "PBRM1", "SETD2", "BAP1", "KDM5C", "MTOR", "TP53",
                  "PIK3CA", "PTEN", "BIRC6", "DST"],
        'OV': ["TP53", "BRCA1", "BRCA2", "NF1", "RB1", "CDK12", "PIK3CA",
               "CSMD3", "FAT3", "HMCN1", "SYNE1", "USH2A", "TTN", "MUC16", "ANKRD30A"],
        'BRCA': ["TP53", "PIK3CA", "TTN", "MUC16", "KMT2C", "MAP3K1", "SYNE1",
                 "PTEN", "GATA3", "FLG", "CDH1", "AKT1", "CBFB", "FOXA1", "ARID1A"],
        'PDAC': ["KRAS", "TP53", "CDKN2A", "SMAD4", "CTNNA2", "MUC4",
                 "KMT2D", "ARID1A", "TGFBR2", "GNAS"],
    }

    # Find genes in ≥3 cancers
    gene_counts = {}
    for cancer, genes in cancer_drivers.items():
        for g in genes:
            gene_counts[g] = gene_counts.get(g, 0) + 1
    common_genes = sorted([g for g, c in gene_counts.items() if c >= 3])
    print(f"Genes in ≥3 cancers: {len(common_genes)}")
    for g in common_genes:
        cancers_with = [c for c, gl in cancer_drivers.items() if g in gl]
        print(f"  {g:12s}  ({gene_counts[g]} cancers: {', '.join(cancers_with)})")

    # Build long table
    all_rows = []
    stderr_bak = sys.stderr
    sys.stderr = io.StringIO()

    for cancer_name, cls in cancer_classes.items():
        print(f"\nLoading {cancer_name}...", file=sys.__stderr__)
        ds = cls()
        prot = ds.get_proteomics(source="umich")
        prot_tumor = prot[~prot.index.str.endswith('.N')].copy()
        prot_tumor.columns = prot_tumor.columns.get_level_values(0)
        prot_tumor = prot_tumor.T.groupby(level=0).mean().T

        trans = ds.get_transcriptomics(source="bcm")
        trans_tumor = trans[~trans.index.str.endswith('.N')].copy()
        trans_tumor.columns = trans_tumor.columns.get_level_values(0)
        trans_tumor = trans_tumor.T.groupby(level=0).mean().T

        common_patients = prot_tumor.index.intersection(trans_tumor.index)
        prot_t = prot_tumor.loc[common_patients]
        trans_t = trans_tumor.loc[common_patients]

        # Standardize within cancer
        avail_genes = set(prot_t.columns) & set(trans_t.columns) & set(common_genes)
        for gene in avail_genes:
            m = trans_t[gene].dropna()
            p = prot_t[gene].dropna()
            idx = m.index.intersection(p.index)
            if len(idx) < 10:
                continue
            mrna_z = (m.loc[idx] - m.loc[idx].mean()) / m.loc[idx].std()
            prot_z = (p.loc[idx] - p.loc[idx].mean()) / p.loc[idx].std()
            for patient in idx:
                all_rows.append({
                    'patient': patient,
                    'cancer': cancer_name,
                    'gene': gene,
                    'mrna_z': mrna_z[patient],
                    'protein_z': prot_z[patient]
                })

    sys.stderr = stderr_bak
    long_df = pd.DataFrame(all_rows)
    print(f"\nLong table: {len(long_df)} rows, {long_df['gene'].nunique()} genes, "
          f"{long_df['cancer'].nunique()} cancers")
    return long_df, common_genes


def run_mixed_models(long_df, common_genes):
    """Run per-gene mixed models: protein_z ~ mrna_z + (mrna_z | cancer_type)."""
    print(f"\n{'='*70}")
    print(f"  Phase 2: Mixed Model Analysis")
    print(f"{'='*70}")

    results = []

    for gene in sorted(common_genes):
        df_gene = long_df[long_df['gene'] == gene].dropna()
        n_cancers = df_gene['cancer'].nunique()

        if n_cancers < 3:
            continue

        n_total = len(df_gene)

        # Try random intercept + random slope
        try:
            model = MixedLM.from_formula(
                "protein_z ~ mrna_z",
                groups="cancer",
                re_formula="~mrna_z",
                data=df_gene
            )
            fit = model.fit(reml=True, maxiter=200)

            fixed_beta = fit.fe_params['mrna_z']
            fixed_se = fit.bse_fe['mrna_z']
            fixed_p = fit.pvalues['mrna_z']

            # Random effect variance
            cov_re = fit.cov_re
            if cov_re.shape[0] > 1 and cov_re.shape[1] > 1:
                random_var_slope = cov_re.iloc[1, 1]
            else:
                random_var_slope = 0
            residual_var = fit.scale

            # ICC for slope
            total_var = random_var_slope + residual_var
            icc = random_var_slope / total_var if total_var > 0 else 0

            # BLUP per cancer
            cancer_betas = {}
            for cancer, re in fit.random_effects.items():
                if len(re) > 1:
                    cancer_betas[cancer] = fixed_beta + re.iloc[1]
                else:
                    cancer_betas[cancer] = fixed_beta

            row = {
                'gene': gene,
                'fixed_beta': fixed_beta,
                'fixed_se': fixed_se,
                'fixed_p': fixed_p,
                'random_var_slope': random_var_slope,
                'residual_var': residual_var,
                'icc': icc,
                'n_cancers': n_cancers,
                'n_total': n_total,
                'converged': True,
                'model_type': 'random_slope',
            }
            row.update({f'beta_{c}': b for c, b in cancer_betas.items()})
            results.append(row)

        except Exception as e1:
            # Fallback: random intercept only
            try:
                model = MixedLM.from_formula(
                    "protein_z ~ mrna_z",
                    groups="cancer",
                    data=df_gene
                )
                fit = model.fit(reml=True, maxiter=200)

                fixed_beta = fit.fe_params['mrna_z']
                fixed_se = fit.bse_fe['mrna_z']
                fixed_p = fit.pvalues['mrna_z']

                # No random slope → estimate from per-cancer OLS
                cancer_betas = {}
                for cancer in df_gene['cancer'].unique():
                    sub = df_gene[df_gene['cancer'] == cancer]
                    if len(sub) >= 10:
                        r, _ = stats.spearmanr(sub['mrna_z'], sub['protein_z'])
                        cancer_betas[cancer] = r

                # Estimate ICC from per-cancer beta variance
                if len(cancer_betas) >= 3:
                    betas = list(cancer_betas.values())
                    random_var_slope = np.var(betas)
                    residual_var = fit.scale
                    icc = random_var_slope / (random_var_slope + residual_var) if (random_var_slope + residual_var) > 0 else 0
                else:
                    random_var_slope = 0
                    residual_var = fit.scale
                    icc = 0

                row = {
                    'gene': gene,
                    'fixed_beta': fixed_beta,
                    'fixed_se': fixed_se,
                    'fixed_p': fixed_p,
                    'random_var_slope': random_var_slope,
                    'residual_var': residual_var,
                    'icc': icc,
                    'n_cancers': n_cancers,
                    'n_total': n_total,
                    'converged': True,
                    'model_type': 'random_intercept',
                }
                row.update({f'beta_{c}': b for c, b in cancer_betas.items()})
                results.append(row)

            except Exception as e2:
                results.append({
                    'gene': gene, 'converged': False,
                    'n_cancers': n_cancers, 'n_total': n_total,
                    'error': str(e2)[:100]
                })

    results_df = pd.DataFrame(results)
    return results_df


def run_variance_decomposition(long_df, common_genes):
    """Two-way ANOVA: β_L2 ~ Gene + Cancer + Gene×Cancer."""
    print(f"\n{'='*70}")
    print(f"  Grand Variance Decomposition")
    print(f"{'='*70}")

    # Build per-gene-per-cancer beta table from Phase 1 results
    cancers = ['COAD', 'LUAD', 'UCEC', 'CCRCC', 'OV', 'BRCA', 'PDAC']
    rows = []
    for cancer in cancers:
        try:
            l2 = pd.read_csv(f"{OUTPUT_DIR}/{cancer}_regression_layer2.csv")
            l2_drv = l2[l2['group'] == 'driver']
            for _, r in l2_drv.iterrows():
                # PHASE0 FIX (Drift 2): publication used the FULL driver-gene set
                # (universe defined by the >=2-cancer filter below), NOT the >=3
                # hardcoded `common_genes`. Restricting to common_genes (5 genes)
                # changed total SS 2.30 -> 0.84 and broke the 49/29/22.7 split.
                if True:
                    rows.append({
                        'gene': r['gene'],
                        'cancer': cancer,
                        'beta_L2': r['beta'],
                        'R2_L2': r['partial_R2']
                    })
        except FileNotFoundError:
            pass

    beta_df = pd.DataFrame(rows)

    # PHASE0 FIX (Drift 2): publication excluded genes present in FEWER THAN TWO
    # cancer types (Methods: "Genes present in fewer than two cancer types ...");
    # i.e. keep genes seen in >=2 cancers (15 genes / 43 obs). The drifted code
    # used >=3 (5 genes / 23 obs) and gave 9/56/35 instead of 49/29/23.
    gene_cancer_counts = beta_df.groupby('gene')['cancer'].nunique()
    valid_genes = gene_cancer_counts[gene_cancer_counts >= 2].index.tolist()
    beta_valid = beta_df[beta_df['gene'].isin(valid_genes)]

    print(f"  Valid genes (≥2 cancers): {len(valid_genes)}")
    print(f"  Total observations: {len(beta_valid)}")

    if len(valid_genes) < 3 or len(beta_valid) < 10:
        print("  Insufficient data for variance decomposition")
        return pd.DataFrame()

    # PHASE0 FIX (Drift 2): publication Figure 4A is an ADDITIVE two-way ANOVA
    # (Gene + Cancer, NO interaction). The drifted code added C(gene):C(cancer),
    # which (a) yields a 5-component table that does not match the published
    # 3-component CSV and (b) errors with inf/NaN on the near-saturated design.
    # The additive model reproduces Gene 48.66% / Cancer 28.61% / Residual 22.73%.
    try:
        model = ols('beta_L2 ~ C(gene) + C(cancer)', data=beta_valid).fit()
        anova_table = anova_lm(model, typ=2)

        ss_gene = anova_table.loc['C(gene)', 'sum_sq']
        ss_cancer = anova_table.loc['C(cancer)', 'sum_sq']
        ss_resid = anova_table.loc['Residual', 'sum_sq']
        ss_total = ss_gene + ss_cancer + ss_resid

        decomp = pd.DataFrame([{
            'component': 'Gene',
            'sum_sq': ss_gene,
            'pct': ss_gene / ss_total * 100,
        }, {
            'component': 'Cancer',
            'sum_sq': ss_cancer,
            'pct': ss_cancer / ss_total * 100,
        }, {
            'component': 'Residual',
            'sum_sq': ss_resid,
            'pct': ss_resid / ss_total * 100,
        }])

        print(f"\n  Variance Decomposition of Layer 2 β (additive):")
        print(f"  {'Component':15s}  {'SS':>8s}  {'%':>6s}")
        print(f"  {'-'*35}")
        for _, r in decomp.iterrows():
            print(f"  {r['component']:15s}  {r['sum_sq']:8.3f}  {r['pct']:5.1f}%")

        decomp.to_csv(f"{OUTPUT_DIR}/variance_decomposition.csv", index=False)
        return decomp

    except Exception as e:
        print(f"  ANOVA error: {e}")
        return pd.DataFrame()


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build long table
    long_df, common_genes = build_long_table()

    # Mixed models
    mm_results = run_mixed_models(long_df, common_genes)

    # Print results
    converged = mm_results[mm_results['converged'] == True].copy()
    if len(converged) > 0:
        converged = converged.sort_values('icc', ascending=False)

        print(f"\n  {'Gene':12s}  {'β_fixed':>8s}  {'SE':>6s}  {'p':>10s}  {'σ²_slope':>8s}  {'ICC':>6s}  {'N_ca':>4s}  {'Model':>15s}")
        print(f"  {'-'*80}")
        for _, r in converged.iterrows():
            sig = "***" if r['fixed_p'] < 0.001 else "**" if r['fixed_p'] < 0.01 else "*" if r['fixed_p'] < 0.05 else ""
            print(f"  {r['gene']:12s}  {r['fixed_beta']:+8.3f}  {r['fixed_se']:6.3f}  {r['fixed_p']:10.2e}  "
                  f"{r['random_var_slope']:8.4f}  {r['icc']:6.3f}  {r['n_cancers']:4.0f}  {r['model_type']:>15s}  {sig}")

        # BLUP table
        cancers = ['COAD', 'LUAD', 'UCEC', 'CCRCC', 'OV', 'BRCA', 'PDAC']
        print(f"\n  BLUP per cancer (β_cancer = fixed + random):")
        print(f"  {'Gene':12s}  " + "  ".join(f"{c:>6s}" for c in cancers))
        print(f"  {'-'*62}")
        for _, r in converged.iterrows():
            vals = []
            for c in cancers:
                col = f'beta_{c}'
                if col in r and not np.isnan(r.get(col, np.nan)):
                    vals.append(f"{r[col]:+6.3f}")
                else:
                    vals.append("   N/A")
            print(f"  {r['gene']:12s}  " + "  ".join(vals))

        # Save
        converged.to_csv(f"{OUTPUT_DIR}/mixed_model_layer2.csv", index=False)

        # ICC ranking
        icc_ranking = converged[['gene', 'fixed_beta', 'icc', 'n_cancers',
                                  'random_var_slope', 'model_type']].sort_values('icc', ascending=False)
        icc_ranking.to_csv(f"{OUTPUT_DIR}/icc_ranking.csv", index=False)

        print(f"\n  ── ICC Interpretation ──")
        high_icc = converged[converged['icc'] > 0.1]
        low_icc = converged[converged['icc'] <= 0.1]
        print(f"  High ICC (>0.1, cancer-dependent): {', '.join(high_icc['gene'].tolist()) if len(high_icc) > 0 else 'none'}")
        print(f"  Low ICC (≤0.1, gene-intrinsic):    {', '.join(low_icc['gene'].tolist()) if len(low_icc) > 0 else 'none'}")

    failed = mm_results[mm_results['converged'] == False]
    if len(failed) > 0:
        print(f"\n  Failed to converge ({len(failed)} genes): {', '.join(failed['gene'].tolist())}")

    # Variance decomposition
    decomp = run_variance_decomposition(long_df, common_genes)

    print(f"\n{'='*70}")
    print(f"  Phase 2 Complete")
    print(f"{'='*70}")
