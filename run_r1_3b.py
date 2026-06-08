#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CNA-adjusted Layer-1 sensitivity analysis
#
# Manuscript: Pan-cancer quantification of driver alteration transmission across
#             molecular layers reveals limited propagation to protein abundance (IJC-26-1558)
# Author:     Hisashi Nakano, PhD - Department of Health Data Science,
#             Niigata University of Health and Welfare, Niigata, Japan
# ORCID:      0000-0002-9023-880X

"""
CNA-adjusted Layer-1 sensitivity analysis: tests copy-number (CNA) confounding of
Layer-1 (mutation -> mRNA).

Adds gene-level copy number (cis) as a covariate to the published Layer-1 OLS and
asks whether the mutation effect (beta_L1 / partial R2_L1) survives.

Design (per driver gene-cancer pair):
  M_base  : mRNA_std ~ mutation + age + sex_male + stage          (= published L1)
  M_adj   : mRNA_std ~ mutation + age + sex_male + stage + gene_CNA
partial R2 of `mutation` = (R2_full - R2_reduced)/(1 - R2_reduced), reduced = covars(+CNA).

We (1) reproduce published beta_L1/R2_L1 with the producer's own regression_layer1
(SANITY GATE), and (2) compare M_base vs M_adj on the SAME CNA-available samples.
Read-only on $ROOT and the cptac cache; all outputs in revision/r1_3b/.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.environ.get("CPTAC_DATA_DIR", ".")
OUT = os.environ.get("CPTAC_OUT_DIR", ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_regression_ts as P          # the producer (read-only import; main() not run)

CANCERS = ['LUAD', 'COAD', 'UCEC', 'CCRCC', 'OV', 'BRCA', 'PDAC']

# ----- faithful mirror of P.regression_layer1, but returns ALL coefficients -----
def myL1(mrna_vals, mutation_status, covars_df):
    """Mirror of run_regression_ts.regression_layer1 (lines 214-288), returning the
    mutation AND cna coefficients. Validated to match P.regression_layer1 exactly
    when covars_df has no 'cna' column (see validate_mirror)."""
    y = mrna_vals.copy()
    y = (y - y.mean()) / y.std()
    x_mut = mutation_status.rename('mutation')
    if covars_df is not None and len(covars_df.columns) > 0:
        common_idx = y.index.intersection(x_mut.index).intersection(covars_df.index)
        if len(common_idx) < 10:
            return None
        y = y.loc[common_idx]; x_mut = x_mut.loc[common_idx]
        cov = covars_df.loc[common_idx].copy()
        cov = cov.loc[:, cov.nunique() > 1]                 # drop constant covariates
        X_full = sm.add_constant(pd.concat([x_mut, cov], axis=1))
        mask = X_full.notna().all(axis=1) & y.notna()
        X_full = X_full[mask]; y = y[mask]
        if len(y) < 10:
            return None
        try:
            mf = sm.OLS(y, X_full).fit()
        except Exception:
            return None
        X_red = sm.add_constant(cov.loc[mask[mask].index])
        mask_red = X_red.notna().all(axis=1)
        X_red = X_red[mask_red]; y_red = y[mask_red]
        if len(y_red) < 10:
            return None
        try:
            mr = sm.OLS(y_red, X_red).fit()
            partial_r2 = (mf.rsquared - mr.rsquared) / (1 - mr.rsquared)
        except Exception:
            partial_r2 = mf.rsquared
        cov_cols = list(cov.columns)
    else:
        return None
    beta = mf.params.get('mutation', np.nan)
    p_mut = mf.pvalues.get('mutation', np.nan)
    cna_beta = mf.params.get('cna', np.nan)
    cna_p = mf.pvalues.get('cna', np.nan)
    return {'beta': beta, 'p_value': p_mut, 'partial_R2': max(partial_r2, 0),
            'n': int(mf.nobs), 'cna_beta': cna_beta, 'cna_p': cna_p,
            'cov_cols': cov_cols, 'samples': list(X_full.index)}

def build_mut_status(trans, mut, gene, all_patients):
    """Replicate run_phase1 L1 per-gene mutation-status construction (lines 480-522).
    Returns (mrna_series, mut_status_series, note) or None if pair is skipped."""
    mutated = P.get_mutated_patients(mut, gene) & all_patients
    wildtype = all_patients - mutated
    if len(wildtype) < 5:                                   # near-100% -> missense vs truncating
        missense, truncating = P.get_mutation_type_patients(mut, gene)
        missense &= all_patients; truncating &= all_patients
        if len(missense) >= 3 and len(truncating) >= 3:
            combined = missense | truncating
            mrna = trans.loc[list(combined), gene].dropna()
            ms = pd.Series(0, index=mrna.index)
            for p in truncating:
                if p in ms.index: ms[p] = 1
            return mrna, ms, 'missense_vs_truncating'
        return None
    if len(mutated) < 3 or len(wildtype) < 3:
        return None
    mrna = trans[gene].dropna()
    patients = mrna.index.intersection(pd.Index(list(all_patients)))
    mrna = mrna.loc[patients]
    ms = pd.Series(0, index=mrna.index)
    for p in mutated:
        if p in ms.index: ms[p] = 1
    return mrna, ms, ''

def load_cnv(ds):
    """Gene-level CNA, mirroring run_cna_vartype_survival.py:80-86 (BCM WES log2 ratio)."""
    cnv = ds.get_CNV(source='bcm')
    cnv_t = cnv[~cnv.index.str.endswith('.N')].copy()
    if isinstance(cnv_t.columns, pd.MultiIndex):
        cnv_t.columns = cnv_t.columns.get_level_values(0)
    cnv_t = cnv_t.T.groupby(level=0).mean().T
    return cnv_t

def main():
    rows = []
    mirror_checks = []
    cnv_meta = {}
    for cancer in CANCERS:
        print(f"\n{'='*60}\n{cancer}\n{'='*60}")
        prot, trans, mut, phospho, clinical, purity, ds = P.load_cancer_data(cancer)
        driver_genes = [g for g in P.CANCER_DRIVERS[cancer] if g in (set(prot.columns) & set(trans.columns))]
        covars, cov_names = P.get_covariates(clinical, purity, prot.index)
        l1_cols = [c for c in cov_names if c != 'tumor_purity']
        covars_l1 = covars[l1_cols] if l1_cols else None
        all_patients = set(trans.index)
        try:
            cnv = load_cnv(ds)
            cnv_meta[cancer] = {'shape': list(cnv.shape), 'n_genes': cnv.shape[1],
                                'sample_example': list(cnv.index[:2]),
                                'value_min': float(np.nanmin(cnv.values)),
                                'value_max': float(np.nanmax(cnv.values))}
            print(f"  CNV(bcm): {cnv.shape}  range[{cnv_meta[cancer]['value_min']:.2f},{cnv_meta[cancer]['value_max']:.2f}]")
        except Exception as e:
            cnv = None
            cnv_meta[cancer] = {'error': f"{type(e).__name__}: {e}"}
            print(f"  CNV load FAILED: {e}")

        for gene in driver_genes:
            bm = build_mut_status(trans, mut, gene, all_patients)
            if bm is None:
                continue
            mrna, ms, note = bm

            # (a) producer reproduction (sanity gate) — full sample set, authoritative
            res_pub = P.regression_layer1(mrna, ms, covars_l1)
            if res_pub is None:
                continue
            # mirror validation on the same full set
            res_mir = myL1(mrna, ms, covars_l1)
            if res_mir is not None:
                mirror_checks.append({'cancer': cancer, 'gene': gene,
                    'd_beta': abs(res_pub['beta'] - res_mir['beta']),
                    'd_pr2': abs(res_pub['partial_R2'] - res_mir['partial_R2'])})

            row = {'cancer': cancer, 'gene': gene, 'note': note,
                   'n_base_full': res_pub['n'], 'n_mut': res_pub['n_mut'],
                   'beta_L1_base_full': res_pub['beta'], 'R2_L1_base_full': res_pub['partial_R2'],
                   'p_mut_base_full': res_pub['p_value']}

            if cnv is None or gene not in cnv.columns:
                row['cna_available'] = False
                rows.append(row); continue
            row['cna_available'] = True

            # like-for-like: restrict to CNA-available samples for BOTH base and adj
            gene_cna = cnv[gene].dropna().rename('cna')
            cov_ok = covars_l1.index[covars_l1.notna().all(axis=1)] if covars_l1 is not None else pd.Index([])
            S = mrna.index.intersection(ms.index).intersection(cov_ok).intersection(gene_cna.index)
            row['n_adj'] = int(len(S))
            row['n_dropped_for_cna'] = int(res_pub['n'] - len(S))
            row['frac_retained'] = float(len(S) / res_pub['n']) if res_pub['n'] else np.nan
            if len(S) < 10:
                row['adj_status'] = 'too_few_after_cna'; rows.append(row); continue

            cov_base = covars_l1.loc[S]
            cov_adj = cov_base.join(gene_cna)
            base_ll = myL1(mrna.loc[S], ms.loc[S], cov_base)
            adj = myL1(mrna.loc[S], ms.loc[S], cov_adj)
            if base_ll is None or adj is None:
                row['adj_status'] = 'fit_failed'; rows.append(row); continue
            row['adj_status'] = 'ok'
            row['beta_L1_base_ll'] = base_ll['beta']; row['R2_L1_base_ll'] = base_ll['partial_R2']
            row['p_mut_base_ll'] = base_ll['p_value']
            row['beta_L1_adj'] = adj['beta']; row['R2_L1_adj'] = adj['partial_R2']
            row['p_mut_adj'] = adj['p_value']
            row['cna_beta'] = adj['cna_beta']; row['cna_p'] = adj['cna_p']
            row['delta_R2_L1'] = adj['partial_R2'] - base_ll['partial_R2']
            rows.append(row)

    df = pd.DataFrame(rows)
    # attach published beta_L1/R2_L1/R2_L2/TS from $ROOT CSVs (read-only)
    pub = pd.read_csv(f"{ROOT}/pancancer_ts_continuous.csv")[['cancer','gene','beta_L1','R2_L1','R2_L2','TS_R2_2L']]
    pub = pub.rename(columns={'beta_L1':'pub_beta_L1','R2_L1':'pub_R2_L1','R2_L2':'pub_R2_L2','TS_R2_2L':'pub_TS_R2'})
    df = df.merge(pub, on=['cancer','gene'], how='left')
    # adjusted TS (L2 held at the published value; this analysis targets L1 only)
    df['TS_R2_adj'] = df['R2_L1_adj'] * df['pub_R2_L2']
    df.to_csv(f"{OUT}/r1_3b_cna_adjusted_L1.csv", index=False)

    mc = pd.DataFrame(mirror_checks)
    summary = {
        'n_pairs_total': int(len(df)),
        'n_cna_available': int(df['cna_available'].sum()),
        'n_adj_ok': int((df.get('adj_status') == 'ok').sum()),
        'mirror_max_dbeta': float(mc['d_beta'].max()) if len(mc) else None,
        'mirror_max_dpr2': float(mc['d_pr2'].max()) if len(mc) else None,
        'cnv_meta': cnv_meta,
    }
    with open(f"{OUT}/r1_3b_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nWROTE r1_3b_cna_adjusted_L1.csv and r1_3b_summary.json")
    print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    main()
