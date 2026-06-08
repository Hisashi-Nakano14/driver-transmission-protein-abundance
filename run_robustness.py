#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Gene-class robustness (7-class Kruskal-Wallis + post-hoc) and permutation nulls
#
# Manuscript: Pan-cancer quantification of driver alteration transmission across
#             molecular layers reveals limited propagation to protein abundance (IJC-26-1558)
# Author:     Hisashi Nakano, PhD - Department of Health Data Science,
#             Niigata University of Health and Welfare, Niigata, Japan
# ORCID:      0000-0002-9023-880X

"""
Robustness Verification for TS_R2
==================================
A1. Permutation test / Null distribution
A2. Covariate adjustment robustness
A3. Gene class enrichment
B4. Sample-size dependency
B5. Leave-one-cancer-out (LOCO) stability
B6. Bootstrap CI for TS_R2
"""

import os
import cptac
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, percentileofscore, kruskal, mannwhitneyu
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import warnings, sys, io, time
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.environ.get("CPTAC_DATA_DIR", ".")

CANCER_CLASSES = {
    'LUAD': cptac.Luad, 'COAD': cptac.Coad, 'UCEC': cptac.Ucec,
    'CCRCC': cptac.Ccrcc, 'OV': cptac.Ov, 'BRCA': cptac.Brca, 'PDAC': cptac.Pdac,
}
MUT_SOURCES = {
    'LUAD': 'harmonized', 'COAD': 'washu', 'UCEC': 'harmonized',
    'CCRCC': 'harmonized', 'OV': 'harmonized', 'BRCA': 'harmonized', 'PDAC': 'washu',
}
CANCER_DRIVERS = {
    'LUAD': ["TP53", "KRAS", "EGFR", "STK11", "KEAP1", "NF1", "BRAF",
             "RBM10", "SETD2", "ARID1A", "RB1", "ERBB2", "MET", "ATM", "CDKN2A"],
    'COAD': ["TP53", "KRAS", "PIK3CA", "SMAD4", "ERBB2", "SOX9", "TCF7L2", "BRAF", "APC", "FBXW7"],
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

GENE_CLASSES = {
    'RTK': ['EGFR', 'ERBB2', 'FGFR2', 'MET'],
    'Chromatin': ['ARID1A', 'PBRM1', 'SETD2', 'BAP1', 'KDM5C', 'CHD4', 'KMT2C', 'KMT2D'],
    'Adhesion': ['CDH1'],
    'CellCycle': ['TP53', 'CDKN2A', 'RB1', 'FBXW7'],
    'Signaling': ['KRAS', 'BRAF', 'PIK3CA', 'PTEN', 'STK11', 'NF1', 'MAP3K1',
                  'AKT1', 'MTOR', 'PIK3R1', 'PPP2R1A', 'SMAD4', 'TGFBR2'],
    'TF': ['GATA3', 'SOX9', 'TCF7L2', 'CTNNB1', 'FOXA1', 'CBFB'],
}

# ── Data cache ──
_DATA_CACHE = {}


def load_cancer(cancer_name):
    """Load and cache cancer data."""
    if cancer_name in _DATA_CACHE:
        return _DATA_CACHE[cancer_name]

    stderr_bak = sys.stderr
    sys.stderr = io.StringIO()
    ds = CANCER_CLASSES[cancer_name]()

    prot = ds.get_proteomics(source="umich")
    prot_t = prot[~prot.index.str.endswith('.N')].copy()
    prot_t.columns = prot_t.columns.get_level_values(0)
    prot_t = prot_t.T.groupby(level=0).mean().T

    trans = ds.get_transcriptomics(source="bcm")
    trans_t = trans[~trans.index.str.endswith('.N')].copy()
    trans_t.columns = trans_t.columns.get_level_values(0)
    trans_t = trans_t.T.groupby(level=0).mean().T

    mut = ds.get_somatic_mutation(source=MUT_SOURCES[cancer_name])

    # Clinical
    clinical = None
    try:
        clinical = ds.get_clinical(source="mssm")
    except:
        pass

    # Purity
    purity = None
    try:
        pur = ds.get_tumor_purity(source="washu")
        if 'TumorPurity' in pur.columns:
            purity = pur['TumorPurity']
    except:
        pass

    common = prot_t.index.intersection(trans_t.index)
    sys.stderr = stderr_bak

    data = {
        'prot': prot_t.loc[common], 'trans': trans_t.loc[common],
        'mut': mut, 'clinical': clinical, 'purity': purity,
        'common': common, 'ds': ds,
    }
    _DATA_CACHE[cancer_name] = data
    return data


def get_mutated_patients(mut, gene):
    """Get set of patients with non-silent mutations."""
    non_silent = mut[~mut['Mutation'].isin(['Silent', 'Intron', "3'UTR", "5'UTR",
                                            "3'Flank", "5'Flank", 'IGR', 'RNA'])]
    return set(non_silent[non_silent['Gene'] == gene].index.unique())


def get_mutation_type_patients(mut, gene):
    non_silent = mut[~mut['Mutation'].isin(['Silent', 'Intron', "3'UTR", "5'UTR",
                                            "3'Flank", "5'Flank", 'IGR', 'RNA'])]
    gene_mut = non_silent[non_silent['Gene'] == gene]
    trunc_types = ['Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins',
                   'Splice_Site', 'Splice_Region']
    missense = set(gene_mut[gene_mut['Mutation'] == 'Missense_Mutation'].index.unique())
    truncating = set(gene_mut[gene_mut['Mutation'].isin(trunc_types)].index.unique())
    return missense - truncating, truncating


def build_covariates(clinical, purity, sample_ids, cov_list):
    """Build covariate DataFrame from requested list.
    cov_list subset of: ['age', 'sex', 'stage', 'tumor_purity']
    """
    if not cov_list:
        return None

    covars = pd.DataFrame(index=sample_ids)

    if clinical is not None:
        cidx = clinical.index.intersection(sample_ids)
        if len(cidx) > 0:
            if 'age' in cov_list and 'age' in clinical.columns:
                age = pd.to_numeric(clinical.loc[cidx, 'age'], errors='coerce')
                if age.notna().sum() > 10:
                    covars['age'] = age

            if 'sex' in cov_list and 'sex' in clinical.columns:
                sex = clinical.loc[cidx, 'sex']
                sex_bin = (sex == 'Male').astype(float)
                if sex_bin.nunique() > 1:
                    covars['sex_male'] = sex_bin

            if 'stage' in cov_list:
                stage_col = None
                for c in clinical.columns:
                    if 'stage' in c.lower() and 'pathological' in c.lower():
                        stage_col = c
                        break
                if stage_col is None:
                    for c in clinical.columns:
                        if 'stage' in c.lower():
                            stage_col = c
                            break
                if stage_col is not None:
                    stage = clinical.loc[cidx, stage_col].astype(str)
                    stage_map = {}
                    for v in stage.unique():
                        vl = v.lower()
                        if 'iv' in vl: stage_map[v] = 4
                        elif 'iii' in vl: stage_map[v] = 3
                        elif 'ii' in vl: stage_map[v] = 2
                        elif 'i' in vl: stage_map[v] = 1
                    stage_ord = stage.map(stage_map)
                    if stage_ord.notna().sum() > 10 and stage_ord.nunique() > 1:
                        covars['stage'] = stage_ord

    if 'tumor_purity' in cov_list and purity is not None:
        pidx = purity.index.intersection(sample_ids)
        if len(pidx) > 10:
            covars['tumor_purity'] = purity.loc[pidx]

    covars = covars.dropna(how='all')
    if len(covars.columns) == 0:
        return None
    return covars


def regress_l1(mrna_vals, mut_status, covars_df=None):
    """Layer 1 regression. Returns (beta, partial_R2, p, n) or Nones."""
    y = mrna_vals.copy()
    if y.std() == 0:
        return np.nan, np.nan, np.nan, len(y)
    y = (y - y.mean()) / y.std()
    x_mut = mut_status.rename('mutation')

    if covars_df is not None and len(covars_df.columns) > 0:
        cidx = y.index.intersection(x_mut.index).intersection(covars_df.index)
        if len(cidx) < 10:
            return np.nan, np.nan, np.nan, len(cidx)
        y, x_mut = y.loc[cidx], x_mut.loc[cidx]
        cov = covars_df.loc[cidx].copy()
        cov = cov.loc[:, cov.nunique() > 1]
        X_full = pd.concat([x_mut, cov], axis=1)
        X_full = sm.add_constant(X_full)
        mask = X_full.notna().all(axis=1) & y.notna()
        X_full, y = X_full[mask], y[mask]
        if len(y) < 10:
            return np.nan, np.nan, np.nan, len(y)
        try:
            mf = sm.OLS(y, X_full).fit()
            X_red = sm.add_constant(cov.loc[mask[mask].index])
            mr = sm.OLS(y, X_red[X_red.notna().all(axis=1)]).fit()
            pr2 = max((mf.rsquared - mr.rsquared) / (1 - mr.rsquared), 0)
        except:
            return np.nan, np.nan, np.nan, len(y)
        idx_name = 'mutation' if 'mutation' in mf.params.index else mf.params.index[1]
        return mf.params[idx_name], pr2, mf.pvalues[idx_name], int(mf.nobs)
    else:
        cidx = y.index.intersection(x_mut.index)
        y, x_mut = y.loc[cidx], x_mut.loc[cidx]
        mask = y.notna() & x_mut.notna()
        y, x_mut = y[mask], x_mut[mask]
        if len(y) < 10:
            return np.nan, np.nan, np.nan, len(y)
        X = sm.add_constant(x_mut.to_frame())
        try:
            m = sm.OLS(y, X).fit()
        except:
            return np.nan, np.nan, np.nan, len(y)
        idx_name = 'mutation' if 'mutation' in m.params.index else m.params.index[1]
        return m.params[idx_name], max(m.rsquared, 0), m.pvalues[idx_name], int(m.nobs)


def regress_l2(protein_vals, mrna_vals, covars_df=None):
    """Layer 2 regression. Returns (beta, partial_R2, p, n) or Nones."""
    common = protein_vals.dropna().index.intersection(mrna_vals.dropna().index)
    if len(common) < 10:
        return np.nan, np.nan, np.nan, len(common)
    y = protein_vals.loc[common]
    x = mrna_vals.loc[common]
    if y.std() == 0 or x.std() == 0:
        return np.nan, np.nan, np.nan, len(common)
    y = (y - y.mean()) / y.std()
    x = ((x - x.mean()) / x.std()).rename('mrna')

    if covars_df is not None and len(covars_df.columns) > 0:
        cidx = common.intersection(covars_df.index)
        if len(cidx) < 10:
            # fallback no covariates
            return _regress_l2_simple(y, x)
        y, x = y.loc[cidx], x.loc[cidx]
        cov = covars_df.loc[cidx].copy()
        cov = cov.loc[:, cov.nunique() > 1]
        X_full = pd.concat([x, cov], axis=1)
        X_full = sm.add_constant(X_full)
        mask = X_full.notna().all(axis=1) & y.notna()
        X_full, y = X_full[mask], y[mask]
        if len(y) < 10:
            return _regress_l2_simple(protein_vals.loc[common], mrna_vals.loc[common])
        try:
            mf = sm.OLS(y, X_full).fit()
            X_red = sm.add_constant(cov.loc[mask[mask].index])
            mr = sm.OLS(y, X_red[X_red.notna().all(axis=1)]).fit()
            pr2 = max((mf.rsquared - mr.rsquared) / (1 - mr.rsquared), 0)
        except:
            return np.nan, np.nan, np.nan, len(y)
        idx_name = 'mrna' if 'mrna' in mf.params.index else mf.params.index[1]
        return mf.params[idx_name], pr2, mf.pvalues[idx_name], int(mf.nobs)
    else:
        return _regress_l2_simple(y, x)


def _regress_l2_simple(y, x):
    mask = y.notna() & x.notna()
    y, x = y[mask], x[mask]
    if len(y) < 10:
        return np.nan, np.nan, np.nan, len(y)
    if y.std() == 0 or x.std() == 0:
        return np.nan, np.nan, np.nan, len(y)
    y = (y - y.mean()) / y.std()
    x = ((x - x.mean()) / x.std()).rename('mrna')
    X = sm.add_constant(x.to_frame())
    try:
        m = sm.OLS(y, X).fit()
    except:
        return np.nan, np.nan, np.nan, len(y)
    idx_name = 'mrna' if 'mrna' in m.params.index else m.params.index[1]
    return m.params[idx_name], max(m.rsquared, 0), m.pvalues[idx_name], int(m.nobs)


def compute_ts_for_gene(cancer_name, gene, cov_list_l1, cov_list_l2):
    """Compute TS_R2 for one gene with specified covariate sets."""
    data = load_cancer(cancer_name)
    trans, prot, mut = data['trans'], data['prot'], data['mut']
    clinical, purity = data['clinical'], data['purity']
    common = data['common']
    available = set(prot.columns) & set(trans.columns)

    if gene not in available:
        return np.nan, np.nan, np.nan

    all_patients = set(common)
    mutated = get_mutated_patients(mut, gene) & all_patients
    wildtype = all_patients - mutated

    # Handle near-100% mutation
    if len(wildtype) < 5:
        missense, truncating = get_mutation_type_patients(mut, gene)
        missense = missense & all_patients
        truncating = truncating & all_patients
        if len(missense) >= 3 and len(truncating) >= 3:
            combined = missense | truncating
            mrna = trans.loc[list(combined), gene].dropna()
            mut_status = pd.Series(0, index=mrna.index)
            for p in truncating:
                if p in mut_status.index:
                    mut_status[p] = 1
        else:
            return np.nan, np.nan, np.nan
    elif len(mutated) < 3:
        return np.nan, np.nan, np.nan
    else:
        mrna = trans[gene].loc[list(all_patients)].dropna()
        mut_status = pd.Series(0, index=mrna.index)
        for p in mutated:
            if p in mut_status.index:
                mut_status[p] = 1

    covars_l1 = build_covariates(clinical, purity, common, cov_list_l1)
    covars_l2 = build_covariates(clinical, purity, common, cov_list_l2)

    _, r2_l1, _, _ = regress_l1(mrna, mut_status, covars_l1)
    _, r2_l2, _, _ = regress_l2(prot[gene], trans[gene], covars_l2)

    if np.isfinite(r2_l1) and np.isfinite(r2_l2):
        return r2_l1 * r2_l2, r2_l1, r2_l2
    return np.nan, r2_l1, r2_l2


# ═══════════════════════════════════════════════════════════════════════════
# A1: PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════════════════

def run_a1_permutation(n_perm=1000):
    print("=" * 70)
    print(f"  A1: PERMUTATION TEST (n_perm={n_perm})")
    print("=" * 70)

    t0 = time.time()
    results = []
    null_distributions = {}  # store for top genes

    # Load observed TS_R2
    obs_ts = pd.read_csv(f"{OUTPUT_DIR}/sensitivity_all_variants.csv")

    gene_count = 0
    total = obs_ts['TS_R2'].notna().sum()

    for _, row in obs_ts.iterrows():
        cancer, gene = row['cancer'], row['gene']
        obs = row['TS_R2']
        if np.isnan(obs):
            continue

        gene_count += 1
        data = load_cancer(cancer)
        trans, prot, mut = data['trans'], data['prot'], data['mut']
        clinical, purity = data['clinical'], data['purity']
        common = data['common']
        all_patients = set(common)

        # Build mutation status
        mutated = get_mutated_patients(mut, gene) & all_patients
        wildtype = all_patients - mutated

        if len(wildtype) < 5:
            missense, truncating = get_mutation_type_patients(mut, gene)
            missense = missense & all_patients
            truncating = truncating & all_patients
            if len(missense) >= 3 and len(truncating) >= 3:
                combined = list(missense | truncating)
                mrna = trans.loc[trans.index.intersection(pd.Index(combined)), gene].dropna()
                mut_status = pd.Series(0, index=mrna.index)
                for p in truncating:
                    if p in mut_status.index:
                        mut_status[p] = 1
            else:
                continue
        elif len(mutated) < 3:
            continue
        else:
            mrna = trans[gene].loc[list(all_patients)].dropna()
            mut_status = pd.Series(0, index=mrna.index)
            for p in mutated:
                if p in mut_status.index:
                    mut_status[p] = 1

        # L1 covariates: age, sex, stage
        covars_l1 = build_covariates(clinical, purity, common, ['age', 'sex', 'stage'])
        # L2 covariates: age, sex, purity
        covars_l2 = build_covariates(clinical, purity, common, ['age', 'sex', 'tumor_purity'])

        rng = np.random.RandomState(42)
        null_ts = []

        for i in range(n_perm):
            # L1: shuffle mutation labels
            mut_perm_vals = mut_status.values.copy()
            rng.shuffle(mut_perm_vals)
            mut_perm = pd.Series(mut_perm_vals, index=mut_status.index, name='mutation')
            _, r2_l1_null, _, _ = regress_l1(mrna, mut_perm, covars_l1)

            # L2: shuffle patient IDs (break mRNA-protein pairing)
            prot_gene = prot[gene].copy()
            prot_perm_vals = prot_gene.values.copy()
            rng.shuffle(prot_perm_vals)
            prot_perm = pd.Series(prot_perm_vals, index=prot_gene.index)
            _, r2_l2_null, _, _ = regress_l2(prot_perm, trans[gene], covars_l2)

            if np.isfinite(r2_l1_null) and np.isfinite(r2_l2_null):
                null_ts.append(r2_l1_null * r2_l2_null)
            else:
                null_ts.append(0.0)

        null_arr = np.array(null_ts)

        # Empirical p-value
        p_perm = (np.sum(null_arr >= obs) + 1) / (n_perm + 1)
        z = (obs - null_arr.mean()) / (null_arr.std() + 1e-10)

        results.append({
            'cancer': cancer, 'gene': gene,
            'TS_R2_obs': obs,
            'null_mean': null_arr.mean(),
            'null_sd': null_arr.std(),
            'null_95': np.percentile(null_arr, 95),
            'null_99': np.percentile(null_arr, 99),
            'p_perm': p_perm,
            'z_score': z,
        })

        # Store null for top genes
        if obs >= 0.02:  # store for genes with meaningful TS
            null_distributions[f"{cancer}_{gene}"] = null_arr

        if gene_count % 10 == 0:
            elapsed = time.time() - t0
            rate = gene_count / elapsed * 60
            print(f"  {gene_count}/{total} genes, {elapsed:.0f}s elapsed, ~{rate:.0f} genes/min")

    perm_df = pd.DataFrame(results)

    # FDR correction
    if len(perm_df) > 0:
        _, fdr, _, _ = multipletests(perm_df['p_perm'], method='fdr_bh')
        perm_df['fdr_perm'] = fdr
    else:
        perm_df['fdr_perm'] = np.nan

    perm_df = perm_df.sort_values('p_perm')
    perm_df.to_csv(f"{OUTPUT_DIR}/permutation_results.csv", index=False)

    # Save null distributions for top genes
    if null_distributions:
        null_df = pd.DataFrame(null_distributions)
        null_df.to_csv(f"{OUTPUT_DIR}/permutation_null_distribution.csv", index=False)

    # Print results
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s ({len(perm_df)} genes)")
    print(f"\n  Top 15 by permutation p-value:")
    print(f"  {'Cancer':6s}  {'Gene':10s}  {'TS_R2':>8s}  {'null_μ':>8s}  {'null_95':>8s}  {'p_perm':>10s}  {'FDR':>10s}  {'z':>6s}")
    print(f"  {'-'*75}")
    for _, r in perm_df.head(15).iterrows():
        sig = "***" if r['fdr_perm'] < 0.001 else "**" if r['fdr_perm'] < 0.01 else "*" if r['fdr_perm'] < 0.05 else ""
        print(f"  {r['cancer']:6s}  {r['gene']:10s}  {r['TS_R2_obs']:8.5f}  "
              f"{r['null_mean']:8.5f}  {r['null_95']:8.5f}  "
              f"{r['p_perm']:10.3e}  {r['fdr_perm']:10.3e}  {r['z_score']:6.1f}  {sig}")

    # Summary
    n_sig_005 = (perm_df['p_perm'] < 0.05).sum()
    n_fdr_005 = (perm_df['fdr_perm'] < 0.05).sum()
    n_fdr_01 = (perm_df['fdr_perm'] < 0.1).sum()
    print(f"\n  Summary:")
    print(f"    p_perm < 0.05: {n_sig_005}/{len(perm_df)}")
    print(f"    FDR < 0.05:    {n_fdr_005}/{len(perm_df)}")
    print(f"    FDR < 0.10:    {n_fdr_01}/{len(perm_df)}")

    return perm_df


# ═══════════════════════════════════════════════════════════════════════════
# A2: COVARIATE ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════

def run_a2_covariate():
    print(f"\n{'='*70}")
    print("  A2: COVARIATE ADJUSTMENT ROBUSTNESS")
    print("=" * 70)

    covariate_sets = {
        'none':           {'l1': [], 'l2': []},
        'age_sex':        {'l1': ['age', 'sex'], 'l2': ['age', 'sex']},
        'age_sex_stage':  {'l1': ['age', 'sex', 'stage'], 'l2': ['age', 'sex']},
        'full':           {'l1': ['age', 'sex', 'stage'], 'l2': ['age', 'sex', 'tumor_purity']},
        'purity_only':    {'l1': [], 'l2': ['tumor_purity']},
    }

    results = []
    for cov_name, cov_spec in covariate_sets.items():
        print(f"\n  ── Covariate set: {cov_name} ──")
        for cancer in CANCER_DRIVERS:
            for gene in CANCER_DRIVERS[cancer]:
                ts, r2_l1, r2_l2 = compute_ts_for_gene(
                    cancer, gene, cov_spec['l1'], cov_spec['l2'])
                results.append({
                    'covariate_set': cov_name,
                    'cancer': cancer, 'gene': gene,
                    'TS_R2': ts, 'R2_L1': r2_l1, 'R2_L2': r2_l2,
                })
        n_valid = sum(1 for r in results if r['covariate_set'] == cov_name and np.isfinite(r['TS_R2']))
        print(f"    {n_valid} gene-cancer pairs computed")

    cov_df = pd.DataFrame(results)
    cov_df.to_csv(f"{OUTPUT_DIR}/covariate_sensitivity.csv", index=False)

    # Ranking correlations
    print(f"\n  Ranking correlations (Spearman ρ):")
    corr_results = []
    cov_names = list(covariate_sets.keys())
    print(f"  {'':18s}  " + "  ".join([f"{c:>12s}" for c in cov_names]))
    print(f"  {'-'*(20 + 14*len(cov_names))}")

    for c1 in cov_names:
        row_vals = []
        for c2 in cov_names:
            df1 = cov_df[cov_df['covariate_set'] == c1].set_index(['cancer', 'gene'])['TS_R2']
            df2 = cov_df[cov_df['covariate_set'] == c2].set_index(['cancer', 'gene'])['TS_R2']
            common = df1.dropna().index.intersection(df2.dropna().index)
            if len(common) >= 5:
                rho, p = spearmanr(df1[common].values, df2[common].values)
                row_vals.append(f"{float(rho):12.3f}")
                if c1 < c2:
                    corr_results.append({
                        'cov1': c1, 'cov2': c2,
                        'spearman_rho': float(rho), 'p': float(p), 'n': len(common)
                    })
            else:
                row_vals.append(f"{'N/A':>12s}")
        print(f"  {c1:18s}  {'  '.join(row_vals)}")

    corr_rank_df = pd.DataFrame(corr_results)
    corr_rank_df.to_csv(f"{OUTPUT_DIR}/covariate_rank_correlation.csv", index=False)

    # Top 10 per covariate set
    print(f"\n  Top 10 stability across covariate sets:")
    baseline_set = cov_df[cov_df['covariate_set'] == 'full']
    baseline_top10 = set(zip(
        baseline_set.nlargest(10, 'TS_R2')['cancer'],
        baseline_set.nlargest(10, 'TS_R2')['gene']
    ))

    for cov_name in cov_names:
        subset = cov_df[cov_df['covariate_set'] == cov_name].dropna(subset=['TS_R2'])
        if len(subset) == 0:
            continue
        top10 = set(zip(
            subset.nlargest(10, 'TS_R2')['cancer'],
            subset.nlargest(10, 'TS_R2')['gene']
        ))
        overlap = len(baseline_top10 & top10)
        top_entries = ', '.join([f"{r['cancer']}-{r['gene']}" for _, r in
                                 subset.nlargest(5, 'TS_R2').iterrows()])
        print(f"    {cov_name:18s}: {overlap:2d}/10 overlap  | Top 5: {top_entries}")

    return cov_df


# ═══════════════════════════════════════════════════════════════════════════
# A3: GENE CLASS ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_a3_gene_class():
    print(f"\n{'='*70}")
    print("  A3: GENE CLASS ENRICHMENT")
    print("=" * 70)

    # Load existing TS data
    ts = pd.read_csv(f"{OUTPUT_DIR}/sensitivity_all_variants.csv")

    def assign_class(gene):
        for cls, genes in GENE_CLASSES.items():
            if gene in genes:
                return cls
        return 'Other'

    ts['gene_class'] = ts['gene'].apply(assign_class)

    # Save annotated file
    ts[['cancer', 'gene', 'gene_class', 'TS_R2', 'R2_L1', 'R2_L2']].to_csv(
        f"{OUTPUT_DIR}/gene_class_ts.csv", index=False)

    # Class summary
    valid = ts.dropna(subset=['TS_R2'])
    summary = valid.groupby('gene_class')['TS_R2'].agg(['median', 'mean', 'count', 'std'])
    summary = summary.sort_values('median', ascending=False)
    print(f"\n  {'Class':12s}  {'Median':>8s}  {'Mean':>8s}  {'SD':>8s}  {'N':>4s}")
    print(f"  {'-'*48}")
    for cls, r in summary.iterrows():
        print(f"  {cls:12s}  {r['median']:8.5f}  {r['mean']:8.5f}  {r['std']:8.5f}  {r['count']:4.0f}")
    summary.to_csv(f"{OUTPUT_DIR}/gene_class_summary.csv")

    # Kruskal-Wallis
    # PHASE0 FIX (Drift 1): publication Figure S2 used ALL 7 classes (no min-size
    # filter), retaining the single-member Adhesion class (CDH1) -> KW p=0.031.
    # The original `if len(g) >= 3` dropped Adhesion -> 6 groups -> p=0.047 (drift).
    groups = [g['TS_R2'].dropna().values for _, g in valid.groupby('gene_class') if len(g) >= 1]
    group_names = [n for n, g in valid.groupby('gene_class') if len(g) >= 1]
    if len(groups) >= 2:
        h_stat, p_kw = kruskal(*groups)
        print(f"\n  Kruskal-Wallis: H={h_stat:.2f}, p={p_kw:.4f}")
    else:
        p_kw = np.nan
        print(f"\n  Kruskal-Wallis: not enough groups")

    # Post-hoc: RTK vs each other class
    test_results = []
    rtk = valid[valid['gene_class'] == 'RTK']['TS_R2'].dropna()
    print(f"\n  Post-hoc: RTK (N={len(rtk)}, median={rtk.median():.5f}) vs others:")
    print(f"  {'Class':12s}  {'N':>4s}  {'Median':>8s}  {'U':>8s}  {'p':>10s}")
    print(f"  {'-'*48}")
    for cls in GENE_CLASSES:
        if cls == 'RTK':
            continue
        other = valid[valid['gene_class'] == cls]['TS_R2'].dropna()
        if len(other) >= 3 and len(rtk) >= 3:
            u, p = mannwhitneyu(rtk.values, other.values, alternative='greater')
            print(f"  {cls:12s}  {len(other):4d}  {other.median():8.5f}  {u:8.0f}  {p:10.4f}")
            test_results.append({
                'comparison': f'RTK_vs_{cls}',
                'n_rtk': len(rtk), 'n_other': len(other),
                'U_stat': u, 'p_value': p,
            })

    # Also test each class vs Other
    other_class = valid[valid['gene_class'] == 'Other']['TS_R2'].dropna()
    if len(other_class) >= 3:
        print(f"\n  Each class vs 'Other' (N={len(other_class)}, median={other_class.median():.5f}):")
        for cls in GENE_CLASSES:
            if cls == 'Other':
                continue
            cls_vals = valid[valid['gene_class'] == cls]['TS_R2'].dropna()
            if len(cls_vals) >= 3:
                u, p = mannwhitneyu(cls_vals.values, other_class.values, alternative='two-sided')
                print(f"    {cls:12s} (N={len(cls_vals):2d}): U={u:.0f}, p={p:.4f}")
                test_results.append({
                    'comparison': f'{cls}_vs_Other',
                    'n_rtk': len(cls_vals), 'n_other': len(other_class),
                    'U_stat': u, 'p_value': p,
                })

    test_df = pd.DataFrame(test_results)
    test_df.to_csv(f"{OUTPUT_DIR}/gene_class_tests.csv", index=False)

    # Per-layer breakdown
    print(f"\n  Per-layer R² by class:")
    for layer, col in [('L1', 'R2_L1'), ('L2', 'R2_L2')]:
        print(f"\n  {layer}:")
        for cls in summary.index:
            vals = valid[valid['gene_class'] == cls][col].dropna()
            if len(vals) > 0:
                print(f"    {cls:12s}: median={vals.median():.4f}, mean={vals.mean():.4f} (N={len(vals)})")

    return ts


# ═══════════════════════════════════════════════════════════════════════════
# B4: SAMPLE-SIZE DEPENDENCY
# ═══════════════════════════════════════════════════════════════════════════

def run_b4_samplesize():
    print(f"\n{'='*70}")
    print("  B4: SAMPLE-SIZE DEPENDENCY")
    print("=" * 70)

    ts = pd.read_csv(f"{OUTPUT_DIR}/sensitivity_all_variants.csv")
    valid = ts.dropna(subset=['TS_R2'])

    # Cancer sample sizes
    cancer_n = {}
    for cancer in CANCER_DRIVERS:
        data = load_cancer(cancer)
        cancer_n[cancer] = len(data['common'])

    valid = valid.copy()
    valid['N'] = valid['cancer'].map(cancer_n)
    rho, p = spearmanr(valid['TS_R2'].values, valid['N'].values)
    print(f"  TS_R2 vs N: Spearman ρ={rho:+.3f}, p={p:.4f}")
    print(f"  Cancer sample sizes: {cancer_n}")

    # Also check R2_L1 and R2_L2 vs N
    for col in ['R2_L1', 'R2_L2']:
        v = valid.dropna(subset=[col])
        r, p2 = spearmanr(v[col].values, v['N'].values)
        print(f"  {col} vs N: ρ={r:+.3f}, p={p2:.4f}")

    # Downsampling: PDAC (N=140) and LUAD (N=111) to N=80
    print(f"\n  Downsampling to N=80 (10 repetitions):")
    for cancer in ['LUAD', 'PDAC']:
        data = load_cancer(cancer)
        common = data['common']
        n_full = len(common)
        if n_full < 85:
            print(f"    {cancer}: N={n_full}, skip (too small)")
            continue

        drivers = CANCER_DRIVERS[cancer]
        # Full TS
        full_ts = {}
        for gene in drivers:
            ts_val, _, _ = compute_ts_for_gene(
                cancer, gene, ['age', 'sex', 'stage'], ['age', 'sex', 'tumor_purity'])
            if np.isfinite(ts_val):
                full_ts[gene] = ts_val

        if len(full_ts) < 3:
            continue

        # Downsample
        down_ts = {g: [] for g in full_ts}
        rng = np.random.RandomState(42)

        trans, prot, mut = data['trans'], data['prot'], data['mut']
        clinical, purity = data['clinical'], data['purity']

        for rep in range(10):
            idx_sub = pd.Index(rng.choice(common, size=80, replace=False))
            # Temporarily replace common in cache
            orig_common = data['common']
            data['common'] = idx_sub
            # Also subset prot/trans
            orig_prot = data['prot']
            orig_trans = data['trans']
            data['prot'] = orig_prot.loc[orig_prot.index.intersection(idx_sub)]
            data['trans'] = orig_trans.loc[orig_trans.index.intersection(idx_sub)]

            for gene in full_ts:
                ts_val, _, _ = compute_ts_for_gene(
                    cancer, gene, ['age', 'sex', 'stage'], ['age', 'sex', 'tumor_purity'])
                if np.isfinite(ts_val):
                    down_ts[gene].append(ts_val)

            # Restore
            data['common'] = orig_common
            data['prot'] = orig_prot
            data['trans'] = orig_trans

        # Compare
        genes_both = [g for g in full_ts if len(down_ts[g]) >= 5]
        if len(genes_both) >= 3:
            full_vals = [full_ts[g] for g in genes_both]
            down_means = [np.mean(down_ts[g]) for g in genes_both]
            rho_d, p_d = spearmanr(full_vals, down_means)
            print(f"    {cancer} (N={n_full}→80): ρ={rho_d:+.3f}, p={p_d:.3f} (N_genes={len(genes_both)})")

            # Show individual gene changes
            for g in genes_both[:5]:
                mean_d = np.mean(down_ts[g])
                sd_d = np.std(down_ts[g])
                print(f"      {g:10s}: full={full_ts[g]:.5f}, down={mean_d:.5f}±{sd_d:.5f}")


# ═══════════════════════════════════════════════════════════════════════════
# B5: LOCO STABILITY
# ═══════════════════════════════════════════════════════════════════════════

def run_b5_loco():
    print(f"\n{'='*70}")
    print("  B5: LEAVE-ONE-CANCER-OUT (LOCO) STABILITY")
    print("=" * 70)

    from statsmodels.regression.mixed_linear_model import MixedLM

    # Build long-format table
    all_long = []
    for cancer in CANCER_DRIVERS:
        data = load_cancer(cancer)
        trans, prot = data['trans'], data['prot']
        common = data['common']
        available = set(prot.columns) & set(trans.columns)

        for gene in CANCER_DRIVERS[cancer]:
            if gene not in available:
                continue
            mrna = trans.loc[common, gene].dropna()
            protein = prot.loc[common, gene].dropna()
            cidx = mrna.index.intersection(protein.index)
            if len(cidx) < 10:
                continue
            m = mrna.loc[cidx]
            p = protein.loc[cidx]
            m_z = (m - m.mean()) / m.std() if m.std() > 0 else m
            p_z = (p - p.mean()) / p.std() if p.std() > 0 else p

            for idx in cidx:
                all_long.append({
                    'patient': idx, 'cancer_type': cancer, 'gene': gene,
                    'mrna_z': m_z[idx], 'protein_z': p_z[idx],
                })

    long_df = pd.DataFrame(all_long)

    # Genes in >= 3 cancers
    gene_cancer_count = long_df.groupby('gene')['cancer_type'].nunique()
    common_genes = gene_cancer_count[gene_cancer_count >= 3].index.tolist()
    print(f"  Genes in ≥3 cancers: {len(common_genes)}")

    # Full model
    full_betas = {}
    for gene in common_genes:
        gdf = long_df[long_df['gene'] == gene].dropna()
        if gdf['cancer_type'].nunique() < 3:
            continue
        try:
            model = MixedLM.from_formula(
                "protein_z ~ mrna_z", groups="cancer_type",
                re_formula="~mrna_z", data=gdf)
            fit = model.fit(reml=True)
            full_betas[gene] = fit.fe_params['mrna_z']
        except:
            try:
                model = MixedLM.from_formula(
                    "protein_z ~ mrna_z", groups="cancer_type",
                    re_formula="~1", data=gdf)
                fit = model.fit(reml=True)
                full_betas[gene] = fit.fe_params['mrna_z']
            except:
                pass

    # LOCO
    loco_results = []
    for held_out in CANCER_DRIVERS:
        remaining = [c for c in CANCER_DRIVERS if c != held_out]
        for gene in full_betas:
            gdf = long_df[(long_df['gene'] == gene) &
                          (long_df['cancer_type'].isin(remaining))].dropna()
            if gdf['cancer_type'].nunique() < 2:
                continue
            try:
                model = MixedLM.from_formula(
                    "protein_z ~ mrna_z", groups="cancer_type",
                    re_formula="~mrna_z", data=gdf)
                fit = model.fit(reml=True)
                loco_results.append({
                    'held_out': held_out, 'gene': gene,
                    'fixed_beta': fit.fe_params['mrna_z'],
                })
            except:
                try:
                    model = MixedLM.from_formula(
                        "protein_z ~ mrna_z", groups="cancer_type",
                        re_formula="~1", data=gdf)
                    fit = model.fit(reml=True)
                    loco_results.append({
                        'held_out': held_out, 'gene': gene,
                        'fixed_beta': fit.fe_params['mrna_z'],
                    })
                except:
                    pass

    loco_df = pd.DataFrame(loco_results)
    loco_df.to_csv(f"{OUTPUT_DIR}/loco_results.csv", index=False)

    print(f"\n  {'Gene':10s}  {'Full β':>8s}  {'LOCO min':>9s}  {'LOCO max':>9s}  {'SD':>6s}  {'Range':>6s}")
    print(f"  {'-'*55}")
    for gene in sorted(full_betas.keys()):
        g = loco_df[loco_df['gene'] == gene]
        if len(g) == 0:
            continue
        print(f"  {gene:10s}  {full_betas[gene]:+8.3f}  "
              f"{g['fixed_beta'].min():+9.3f}  {g['fixed_beta'].max():+9.3f}  "
              f"{g['fixed_beta'].std():6.3f}  {g['fixed_beta'].max()-g['fixed_beta'].min():6.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# B6: BOOTSTRAP CI
# ═══════════════════════════════════════════════════════════════════════════

def run_b6_bootstrap(n_boot=1000):
    print(f"\n{'='*70}")
    print(f"  B6: BOOTSTRAP CI FOR TOP 10 (n_boot={n_boot})")
    print("=" * 70)

    ts = pd.read_csv(f"{OUTPUT_DIR}/sensitivity_all_variants.csv")
    top10 = ts.nlargest(10, 'TS_R2')[['cancer', 'gene', 'TS_R2']]

    boot_results = []
    for _, row in top10.iterrows():
        cancer, gene, obs = row['cancer'], row['gene'], row['TS_R2']
        data = load_cancer(cancer)
        trans, prot, mut = data['trans'], data['prot'], data['mut']
        clinical, purity = data['clinical'], data['purity']
        common = data['common']
        all_patients = set(common)

        # Build mutation status
        mutated = get_mutated_patients(mut, gene) & all_patients
        wildtype = all_patients - mutated

        if len(wildtype) < 5:
            missense, truncating = get_mutation_type_patients(mut, gene)
            missense = missense & all_patients
            truncating = truncating & all_patients
            combined = list(missense | truncating)
            mrna = trans.loc[trans.index.intersection(pd.Index(combined)), gene].dropna()
            mut_status = pd.Series(0, index=mrna.index)
            for p_ in truncating:
                if p_ in mut_status.index:
                    mut_status[p_] = 1
        else:
            mrna = trans[gene].loc[list(all_patients)].dropna()
            mut_status = pd.Series(0, index=mrna.index)
            for p_ in mutated:
                if p_ in mut_status.index:
                    mut_status[p_] = 1

        covars_l1 = build_covariates(clinical, purity, common, ['age', 'sex', 'stage'])
        covars_l2 = build_covariates(clinical, purity, common, ['age', 'sex', 'tumor_purity'])

        # Common samples for L2
        l2_common = prot[gene].dropna().index.intersection(trans[gene].dropna().index)
        l2_common = l2_common.intersection(common)

        rng = np.random.RandomState(42)
        boot_ts = []

        for i in range(n_boot):
            # L1 bootstrap
            l1_idx = rng.choice(len(mrna), size=len(mrna), replace=True)
            mrna_b = mrna.iloc[l1_idx]
            mrna_b.index = mrna.index  # keep original index for covariate matching
            mut_b = mut_status.iloc[l1_idx]
            mut_b.index = mrna.index
            _, r2_l1, _, _ = regress_l1(mrna_b, mut_b, covars_l1)

            # L2 bootstrap
            l2_idx = rng.choice(len(l2_common), size=len(l2_common), replace=True)
            prot_b = prot.loc[l2_common, gene].iloc[l2_idx]
            prot_b.index = l2_common
            trans_b = trans.loc[l2_common, gene].iloc[l2_idx]
            trans_b.index = l2_common
            _, r2_l2, _, _ = regress_l2(prot_b, trans_b, covars_l2)

            if np.isfinite(r2_l1) and np.isfinite(r2_l2):
                boot_ts.append(r2_l1 * r2_l2)

        if len(boot_ts) >= 100:
            boot_arr = np.array(boot_ts)
            ci_lo = np.percentile(boot_arr, 2.5)
            ci_hi = np.percentile(boot_arr, 97.5)
            boot_results.append({
                'cancer': cancer, 'gene': gene,
                'TS_R2_obs': obs,
                'boot_mean': boot_arr.mean(),
                'boot_sd': boot_arr.std(),
                'CI_2.5': ci_lo, 'CI_97.5': ci_hi,
                'n_boot_valid': len(boot_ts),
            })

    boot_df = pd.DataFrame(boot_results)
    boot_df.to_csv(f"{OUTPUT_DIR}/bootstrap_ci.csv", index=False)

    print(f"\n  {'Cancer':6s}  {'Gene':10s}  {'TS_R2':>8s}  {'95% CI':>20s}  {'Width':>8s}")
    print(f"  {'-'*60}")
    for _, r in boot_df.iterrows():
        ci_str = f"[{r['CI_2.5']:.5f}, {r['CI_97.5']:.5f}]"
        width = r['CI_97.5'] - r['CI_2.5']
        print(f"  {r['cancer']:6s}  {r['gene']:10s}  {r['TS_R2_obs']:8.5f}  {ci_str:>20s}  {width:8.5f}")

    return boot_df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['a1', 'a2', 'a3', 'b4', 'b5', 'b6', 'all', 'priority_a'],
                        default='priority_a')
    parser.add_argument('--n_perm', type=int, default=1000)
    parser.add_argument('--n_boot', type=int, default=1000)
    args = parser.parse_args()

    if args.task in ('a3', 'all', 'priority_a'):
        run_a3_gene_class()

    if args.task in ('a2', 'all', 'priority_a'):
        run_a2_covariate()

    if args.task in ('a1', 'all', 'priority_a'):
        run_a1_permutation(n_perm=args.n_perm)

    if args.task in ('b4', 'all'):
        run_b4_samplesize()

    if args.task in ('b5', 'all'):
        run_b5_loco()

    if args.task in ('b6', 'all'):
        run_b6_bootstrap(n_boot=args.n_boot)

    print(f"\n{'='*70}")
    print("  ROBUSTNESS VERIFICATION COMPLETE")
    print(f"{'='*70}")
