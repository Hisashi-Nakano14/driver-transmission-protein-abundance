#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Main Transmission Score: Layer-1/Layer-2/Layer-3 OLS regression and TS_R2 = R2_L1 x R2_L2
#
# Manuscript: Pan-cancer quantification of driver alteration transmission across
#             molecular layers reveals limited propagation to protein abundance (IJC-26-1558)
# Author:     Hisashi Nakano, PhD - Department of Health Data Science,
#             Niigata University of Health and Welfare, Niigata, Japan
# ORCID:      0000-0002-9023-880X

"""
Phase 1: Regression-based Transmission Score
=============================================
Layer 1: mRNA ~ mutation_status + covariates  → β_mut, partial R²
Layer 2: Protein ~ mRNA + covariates          → β_rna, partial R²
Layer 3: Phospho ~ Protein + covariates       → β_prot, partial R²

Continuous TS: TS_product, TS_log, TS_R2

Run LUAD first for structure verification, then remaining 6 cancers.
"""

import os
import cptac
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import warnings, sys, io
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.environ.get("CPTAC_DATA_DIR", ".")

# ── Existing driver gene lists (from prior analyses) ──────────────────────

CANCER_DRIVERS = {
    'LUAD': ["TP53", "KRAS", "EGFR", "STK11", "KEAP1", "NF1", "BRAF",
             "RBM10", "SETD2", "ARID1A", "RB1", "ERBB2", "MET", "ATM", "CDKN2A"],
    'COAD': ["TP53", "KRAS", "PIK3CA", "SMAD4", "ERBB2", "SOX9", "TCF7L2", "BRAF", "APC", "FBXW7"],
    'UCEC': ["PTEN", "TP53", "PIK3CA", "ARID1A", "CTNNB1", "KRAS", "FGFR2",
             "PPP2R1A", "PIK3R1", "FBXW7", "RPL22", "CTCF", "ZFHX3", "CHD4", "SPOP"],
    'CCRCC': ["VHL", "PBRM1", "SETD2", "BAP1", "KDM5C", "MTOR", "TP53",
              "PIK3CA", "PTEN", "BIRC6", "DST"],  # CUBN excluded
    'OV': ["TP53", "BRCA1", "BRCA2", "NF1", "RB1", "CDK12", "PIK3CA",
           "CSMD3", "FAT3", "HMCN1", "SYNE1", "USH2A", "TTN", "MUC16", "ANKRD30A"],
    'BRCA': ["TP53", "PIK3CA", "TTN", "MUC16", "KMT2C", "MAP3K1", "SYNE1",
             "PTEN", "GATA3", "FLG", "CDH1", "AKT1", "CBFB", "FOXA1", "ARID1A"],
    'PDAC': ["KRAS", "TP53", "CDKN2A", "SMAD4", "CTNNA2", "MUC4",
             "KMT2D", "ARID1A", "TGFBR2", "GNAS"],
}

HOUSEKEEPING_GENES = [
    "ACTB", "GAPDH", "TUBB", "HSP90AA1", "LDHA", "ENO1", "PKM",
    "ALDOA", "PGK1", "TPI1", "RPL13A", "RPS18", "B2M", "HNRNPA1",
    "EEF1A1", "EEF2", "YWHAZ", "UBC", "VIM", "HSP90AB1",
    "PPIA", "TUBA1B", "CCT2", "TCP1", "HSPA8"
]

# ── Data loading (reuse existing pipeline structure) ──────────────────────

def load_cancer_data(cancer_name):
    """Load data from cptac. Returns prot, trans, mut, phospho, clinical, purity."""
    print(f"[1] Loading {cancer_name} data...")
    stderr_bak = sys.stderr
    sys.stderr = io.StringIO()

    cancer_classes = {
        'LUAD': cptac.Luad, 'COAD': cptac.Coad, 'UCEC': cptac.Ucec,
        'CCRCC': cptac.Ccrcc, 'OV': cptac.Ov, 'BRCA': cptac.Brca, 'PDAC': cptac.Pdac,
    }
    mut_sources = {
        'LUAD': 'harmonized', 'COAD': 'washu', 'UCEC': 'harmonized',
        'CCRCC': 'harmonized', 'OV': 'harmonized', 'BRCA': 'harmonized', 'PDAC': 'washu',
    }

    ds = cancer_classes[cancer_name]()

    # Proteomics
    prot = ds.get_proteomics(source="umich")
    prot_tumor = prot[~prot.index.str.endswith('.N')].copy()
    prot_tumor.columns = prot_tumor.columns.get_level_values(0)
    prot_tumor = prot_tumor.T.groupby(level=0).mean().T

    # Transcriptomics
    trans = ds.get_transcriptomics(source="bcm")
    trans_tumor = trans[~trans.index.str.endswith('.N')].copy()
    trans_tumor.columns = trans_tumor.columns.get_level_values(0)
    trans_tumor = trans_tumor.T.groupby(level=0).mean().T

    # Mutations
    mut = ds.get_somatic_mutation(source=mut_sources[cancer_name])

    # Phosphoproteomics
    phospho_tumor = None
    for psrc in ['umich', 'bcm']:
        try:
            phospho = ds.get_phosphoproteomics(source=psrc)
            phospho_tumor = phospho[~phospho.index.str.endswith('.N')].copy()
            break
        except:
            pass

    # Clinical
    clinical = None
    try:
        clinical = ds.get_clinical(source="mssm")
    except:
        pass

    # Tumor purity
    purity = None
    try:
        pur = ds.get_tumor_purity(source="washu")
        if 'TumorPurity' in pur.columns:
            purity = pur['TumorPurity']
    except:
        pass

    common = prot_tumor.index.intersection(trans_tumor.index)
    sys.stderr = stderr_bak

    print(f"  Prot: {prot_tumor.shape}, Trans: {trans_tumor.shape}, "
          f"Mut: {mut.shape}, Common: {len(common)}")
    if clinical is not None:
        print(f"  Clinical: {clinical.shape}")
    if purity is not None:
        print(f"  Tumor purity: {purity.notna().sum()} samples")

    return (prot_tumor.loc[common], trans_tumor.loc[common], mut,
            phospho_tumor, clinical, purity, ds)


def get_covariates(clinical, purity, sample_ids):
    """Extract and encode covariates from clinical data."""
    covars = pd.DataFrame(index=sample_ids)
    cov_names = []

    if clinical is not None:
        common = clinical.index.intersection(sample_ids)
        if len(common) > 0:
            # Age
            if 'age' in clinical.columns:
                age = pd.to_numeric(clinical.loc[common, 'age'], errors='coerce')
                if age.notna().sum() > 10:
                    covars['age'] = age
                    cov_names.append('age')

            # Sex
            if 'sex' in clinical.columns:
                sex = clinical.loc[common, 'sex']
                sex_binary = (sex == 'Male').astype(float)
                if sex_binary.nunique() > 1:
                    covars['sex_male'] = sex_binary
                    cov_names.append('sex_male')

            # Stage (ordinal encoding)
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
                stage = clinical.loc[common, stage_col].astype(str)
                stage_map = {}
                for v in stage.unique():
                    vl = v.lower()
                    if 'iv' in vl:
                        stage_map[v] = 4
                    elif 'iii' in vl:
                        stage_map[v] = 3
                    elif 'ii' in vl:
                        stage_map[v] = 2
                    elif 'i' in vl:
                        stage_map[v] = 1
                stage_ord = stage.map(stage_map)
                if stage_ord.notna().sum() > 10 and stage_ord.nunique() > 1:
                    covars['stage'] = stage_ord
                    cov_names.append('stage')

    # Tumor purity
    if purity is not None:
        common_pur = purity.index.intersection(sample_ids)
        if len(common_pur) > 10:
            covars['tumor_purity'] = purity.loc[common_pur]
            cov_names.append('tumor_purity')

    covars = covars.dropna(how='all')
    print(f"  Covariates available: {cov_names} ({len(covars)} samples)")
    return covars, cov_names


def get_mutated_patients(mut, gene):
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
    missense = missense - truncating
    return missense, truncating


# ── Regression functions ──────────────────────────────────────────────────

def regression_layer1(mrna_vals, mutation_status, covars_df):
    """
    Layer 1: mRNA ~ mutation_status + covariates
    Returns: beta_mut (standardized), se, p_value, partial_R2, n, n_mut
    """
    # Standardize mRNA
    y = mrna_vals.copy()
    y = (y - y.mean()) / y.std()

    # Build X
    x_mut = mutation_status.rename('mutation')
    if covars_df is not None and len(covars_df.columns) > 0:
        common_idx = y.index.intersection(x_mut.index).intersection(covars_df.index)
        if len(common_idx) < 10:
            return None
        y = y.loc[common_idx]
        x_mut = x_mut.loc[common_idx]
        cov = covars_df.loc[common_idx].copy()
        # Drop constant covariates
        cov = cov.loc[:, cov.nunique() > 1]
        X_full = pd.concat([x_mut, cov], axis=1)
        X_full = sm.add_constant(X_full)
        mask = X_full.notna().all(axis=1) & y.notna()
        X_full = X_full[mask]
        y = y[mask]
        if len(y) < 10:
            return None

        try:
            model_full = sm.OLS(y, X_full).fit()
        except:
            return None

        # Reduced model (without mutation)
        X_red = sm.add_constant(cov.loc[X_full.index.drop('const', errors='ignore')
                                        if 'const' in X_full.columns else X_full.index])
        X_red = sm.add_constant(cov.loc[mask[mask].index])
        mask_red = X_red.notna().all(axis=1)
        X_red = X_red[mask_red]
        y_red = y[mask_red]
        if len(y_red) < 10:
            return None
        try:
            model_red = sm.OLS(y_red, X_red).fit()
            partial_r2 = (model_full.rsquared - model_red.rsquared) / (1 - model_red.rsquared)
        except:
            partial_r2 = model_full.rsquared
    else:
        common_idx = y.index.intersection(x_mut.index)
        y = y.loc[common_idx]
        x_mut = x_mut.loc[common_idx]
        mask = y.notna() & x_mut.notna()
        y = y[mask]
        x_mut = x_mut[mask]
        if len(y) < 10:
            return None
        X_full = sm.add_constant(x_mut.to_frame())
        try:
            model_full = sm.OLS(y, X_full).fit()
        except:
            return None
        partial_r2 = model_full.rsquared

    # Extract mutation coefficient (index 1 = 'mutation')
    if 'mutation' in model_full.params.index:
        beta = model_full.params['mutation']
        se = model_full.bse['mutation']
        pval = model_full.pvalues['mutation']
    else:
        beta = model_full.params.iloc[1]
        se = model_full.bse.iloc[1]
        pval = model_full.pvalues.iloc[1]

    n_mut = int(x_mut.loc[mask if isinstance(mask, pd.Series) else x_mut.index].sum())

    return {
        'beta': beta, 'se': se, 'p_value': pval,
        'partial_R2': max(partial_r2, 0),  # clamp to 0
        'n': int(model_full.nobs), 'n_mut': n_mut
    }


def regression_layer2(protein_vals, mrna_vals, covars_df):
    """
    Layer 2: Protein ~ mRNA + covariates (all samples)
    Returns: beta_rna (standardized), se, p_value, partial_R2, n
    """
    common = protein_vals.dropna().index.intersection(mrna_vals.dropna().index)
    if len(common) < 10:
        return None

    y = protein_vals.loc[common]
    y = (y - y.mean()) / y.std()
    x_rna = mrna_vals.loc[common]
    x_rna = (x_rna - x_rna.mean()) / x_rna.std()
    x_rna = x_rna.rename('mrna')

    if covars_df is not None and len(covars_df.columns) > 0:
        cidx = common.intersection(covars_df.index)
        if len(cidx) < 10:
            # Fall back to no covariates
            return _layer2_simple(y, x_rna)
        y = y.loc[cidx]
        x_rna = x_rna.loc[cidx]
        cov = covars_df.loc[cidx].copy()
        cov = cov.loc[:, cov.nunique() > 1]

        X_full = pd.concat([x_rna, cov], axis=1)
        X_full = sm.add_constant(X_full)
        mask = X_full.notna().all(axis=1) & y.notna()
        X_full = X_full[mask]
        y = y[mask]
        if len(y) < 10:
            return _layer2_simple(protein_vals.loc[common],
                                  (mrna_vals.loc[common] - mrna_vals.loc[common].mean()) / mrna_vals.loc[common].std())

        try:
            model_full = sm.OLS(y, X_full).fit()
        except:
            return None

        X_red = sm.add_constant(cov.loc[mask[mask].index])
        mask_red = X_red.notna().all(axis=1)
        try:
            model_red = sm.OLS(y[mask_red], X_red[mask_red]).fit()
            partial_r2 = (model_full.rsquared - model_red.rsquared) / (1 - model_red.rsquared)
        except:
            partial_r2 = model_full.rsquared
    else:
        return _layer2_simple(y, x_rna)

    if 'mrna' in model_full.params.index:
        beta = model_full.params['mrna']
        se = model_full.bse['mrna']
        pval = model_full.pvalues['mrna']
    else:
        beta = model_full.params.iloc[1]
        se = model_full.bse.iloc[1]
        pval = model_full.pvalues.iloc[1]

    return {
        'beta': beta, 'se': se, 'p_value': pval,
        'partial_R2': max(partial_r2, 0), 'n': int(model_full.nobs)
    }


def _layer2_simple(y, x_rna):
    """Layer 2 without covariates."""
    mask = y.notna() & x_rna.notna()
    y = y[mask]
    x = x_rna[mask]
    if len(y) < 10:
        return None
    # Re-standardize after filtering
    y = (y - y.mean()) / y.std() if y.std() > 0 else y
    x = (x - x.mean()) / x.std() if x.std() > 0 else x
    x = x.rename('mrna')
    X = sm.add_constant(x.to_frame())
    try:
        model = sm.OLS(y, X).fit()
    except:
        return None
    beta = model.params['mrna'] if 'mrna' in model.params.index else model.params.iloc[1]
    se = model.bse['mrna'] if 'mrna' in model.bse.index else model.bse.iloc[1]
    pval = model.pvalues['mrna'] if 'mrna' in model.pvalues.index else model.pvalues.iloc[1]
    return {
        'beta': beta, 'se': se, 'p_value': pval,
        'partial_R2': max(model.rsquared, 0), 'n': int(model.nobs)
    }


def regression_layer3(phospho_vals, protein_vals, covars_df):
    """
    Layer 3: Phospho ~ Protein + covariates
    Returns: beta_prot (standardized), se, p_value, partial_R2, n
    """
    common = phospho_vals.dropna().index.intersection(protein_vals.dropna().index)
    # Handle duplicate indices
    common = common[~common.duplicated()]
    if len(common) < 10:
        return None

    y_raw = phospho_vals.loc[common].copy()
    x_raw = protein_vals.loc[common].copy()
    # Ensure no duplicate indices
    if y_raw.index.duplicated().any():
        y_raw = y_raw[~y_raw.index.duplicated(keep='first')]
        x_raw = x_raw.loc[y_raw.index]

    if y_raw.std() == 0 or x_raw.std() == 0:
        return None

    # Try with covariates first
    if covars_df is not None and len(covars_df.columns) > 0:
        cidx = common.intersection(covars_df.index)
        if len(cidx) >= 10:
            y_c = y_raw.loc[cidx]
            x_c = x_raw.loc[cidx]
            cov = covars_df.loc[cidx].copy()
            cov = cov.loc[:, cov.nunique() > 1]
            if len(cov.columns) > 0:
                # Standardize within this subset
                y_s = (y_c - y_c.mean()) / y_c.std()
                x_s = ((x_c - x_c.mean()) / x_c.std()).rename('protein')
                combined = pd.concat([y_s.rename('y'), x_s, cov], axis=1).dropna()
                if len(combined) >= 10:
                    try:
                        X_full = sm.add_constant(combined[['protein'] + list(cov.columns)])
                        model_full = sm.OLS(combined['y'], X_full).fit()
                        X_red = sm.add_constant(combined[list(cov.columns)])
                        model_red = sm.OLS(combined['y'], X_red).fit()
                        partial_r2 = (model_full.rsquared - model_red.rsquared) / (1 - model_red.rsquared)
                        return {
                            'beta': model_full.params['protein'],
                            'se': model_full.bse['protein'],
                            'p_value': model_full.pvalues['protein'],
                            'partial_R2': max(partial_r2, 0),
                            'n': int(model_full.nobs)
                        }
                    except:
                        pass

    # Fallback: no covariates
    y_s = (y_raw - y_raw.mean()) / y_raw.std()
    x_s = ((x_raw - x_raw.mean()) / x_raw.std()).rename('protein')
    X = sm.add_constant(x_s.to_frame())
    try:
        model = sm.OLS(y_s, X).fit()
    except:
        return None
    return {
        'beta': model.params['protein'],
        'se': model.bse['protein'],
        'p_value': model.pvalues['protein'],
        'partial_R2': max(model.rsquared, 0),
        'n': int(model.nobs)
    }


# ── Main analysis for one cancer ──────────────────────────────────────────

def run_phase1(cancer_name):
    print(f"\n{'='*70}")
    print(f"  Phase 1 Regression Analysis: {cancer_name}")
    print(f"{'='*70}")

    prot, trans, mut, phospho, clinical, purity, ds = load_cancer_data(cancer_name)
    driver_genes = CANCER_DRIVERS[cancer_name]

    # Available genes
    available = set(prot.columns) & set(trans.columns)
    driver_genes = [g for g in driver_genes if g in available]
    print(f"  Driver genes in data: {len(driver_genes)}")

    # Get covariates
    covars, cov_names = get_covariates(clinical, purity, prot.index)

    # Covariates for L1 (age, sex, stage — not purity)
    l1_cov_cols = [c for c in cov_names if c != 'tumor_purity']
    covars_l1 = covars[l1_cov_cols] if l1_cov_cols else None

    # Covariates for L2/L3 (age, sex, purity — not stage for molecular layers)
    l2_cov_cols = [c for c in cov_names if c != 'stage']
    covars_l2 = covars[l2_cov_cols] if l2_cov_cols else None

    all_patients = set(trans.index)

    # ── Layer 1: Mutation → mRNA ──
    print(f"\n[2] Layer 1: Mutation → mRNA (regression)...")
    l1_results = []

    for gene in driver_genes:
        if gene not in trans.columns:
            continue

        mutated = get_mutated_patients(mut, gene) & all_patients
        wildtype = all_patients - mutated

        # Near-100% mutation: missense vs truncating
        if len(wildtype) < 5:
            missense, truncating = get_mutation_type_patients(mut, gene)
            missense = missense & all_patients
            truncating = truncating & all_patients
            if len(missense) >= 3 and len(truncating) >= 3:
                # Use truncating as "mutated" vs missense as reference
                combined = missense | truncating
                mrna = trans.loc[list(combined), gene].dropna()
                mut_status = pd.Series(0, index=mrna.index)
                for p in truncating:
                    if p in mut_status.index:
                        mut_status[p] = 1
                res = regression_layer1(mrna, mut_status, covars_l1)
                if res:
                    res['gene'] = gene
                    res['note'] = 'missense_vs_truncating'
                    l1_results.append(res)
            continue

        if len(mutated) < 3 or len(wildtype) < 3:
            continue

        mrna = trans[gene].dropna()
        patients = mrna.index.intersection(pd.Index(list(all_patients)))
        mrna = mrna.loc[patients]
        mut_status = pd.Series(0, index=mrna.index)
        for p in mutated:
            if p in mut_status.index:
                mut_status[p] = 1

        res = regression_layer1(mrna, mut_status, covars_l1)
        if res:
            res['gene'] = gene
            res['note'] = ''
            l1_results.append(res)

    l1_df = pd.DataFrame(l1_results)
    if len(l1_df) > 0:
        _, fdr, _, _ = multipletests(l1_df['p_value'], method='fdr_bh')
        l1_df['FDR'] = fdr
        l1_df = l1_df.sort_values('FDR')
        print(f"\n  {'Gene':12s}  {'β_mut':>8s}  {'pR²':>8s}  {'FDR':>10s}  {'n':>4s}  {'n_mut':>5s}")
        print(f"  {'-'*55}")
        for _, r in l1_df.iterrows():
            sig = "***" if r['FDR'] < 0.001 else "**" if r['FDR'] < 0.01 else "*" if r['FDR'] < 0.05 else ""
            print(f"  {r['gene']:12s}  {r['beta']:+8.3f}  {r['partial_R2']:8.4f}  {r['FDR']:10.2e}  {r['n']:4.0f}  {r['n_mut']:5.0f}  {sig}")
    l1_df.to_csv(f"{OUTPUT_DIR}/{cancer_name}_regression_layer1.csv", index=False)

    # ── Layer 2: mRNA → Protein ──
    print(f"\n[3] Layer 2: mRNA → Protein (regression)...")
    l2_results = []

    all_genes = list(set(driver_genes) | set(HOUSEKEEPING_GENES))
    for gene in sorted(set(all_genes) & available):
        res = regression_layer2(prot[gene], trans[gene], covars_l2)
        if res:
            res['gene'] = gene
            res['group'] = 'driver' if gene in driver_genes else 'non-driver'
            l2_results.append(res)

    l2_df = pd.DataFrame(l2_results)
    if len(l2_df) > 0:
        _, fdr, _, _ = multipletests(l2_df['p_value'], method='fdr_bh')
        l2_df['FDR'] = fdr
        l2_df = l2_df.sort_values('partial_R2', ascending=False)

        # Print drivers
        l2_drv = l2_df[l2_df['group'] == 'driver'].sort_values('beta', ascending=False)
        print(f"\n  Drivers:")
        print(f"  {'Gene':12s}  {'β_rna':>8s}  {'pR²':>8s}  {'FDR':>10s}  {'n':>4s}")
        print(f"  {'-'*48}")
        for _, r in l2_drv.iterrows():
            sig = "***" if r['FDR'] < 0.001 else "**" if r['FDR'] < 0.01 else "*" if r['FDR'] < 0.05 else ""
            print(f"  {r['gene']:12s}  {r['beta']:+8.3f}  {r['partial_R2']:8.4f}  {r['FDR']:10.2e}  {r['n']:4.0f}  {sig}")

        # Positive control comparison
        l2_hk = l2_df[l2_df['group'] == 'non-driver']
        if len(l2_hk) > 0:
            print(f"\n  Positive control:")
            print(f"    HK median β = {l2_hk['beta'].median():+.3f}, median pR² = {l2_hk['partial_R2'].median():.4f} (N={len(l2_hk)})")
            print(f"    Driver median β = {l2_drv['beta'].median():+.3f}, median pR² = {l2_drv['partial_R2'].median():.4f} (N={len(l2_drv)})")

    l2_df.to_csv(f"{OUTPUT_DIR}/{cancer_name}_regression_layer2.csv", index=False)

    # ── Layer 3: Protein → Phosphoprotein ──
    print(f"\n[4] Layer 3: Protein → Phosphoprotein (regression)...")
    l3_results = []

    if phospho is not None:
        phospho_gs = phospho.copy()
        phospho_gs.columns = pd.MultiIndex.from_arrays([
            phospho.columns.get_level_values(0),
            phospho.columns.get_level_values(1)
        ], names=['gene', 'site'])
        phospho_agg = phospho_gs.T.groupby(level=['gene', 'site']).mean().T

        for gene in driver_genes:
            if gene not in prot.columns:
                continue
            gene_mask = phospho_agg.columns.get_level_values('gene') == gene
            gene_cols = phospho_agg.columns[gene_mask]
            if len(gene_cols) == 0:
                continue

            common_idx = prot.index.intersection(phospho_agg.index)
            if len(common_idx) < 10:
                continue

            prot_vals = prot.loc[common_idx, gene]

            for col in gene_cols:
                site = col[1]
                phospho_vals = phospho_agg.loc[common_idx, col]
                res = regression_layer3(phospho_vals, prot_vals, covars_l2)
                if res:
                    res['gene'] = gene
                    res['site'] = site
                    l3_results.append(res)

    l3_df = pd.DataFrame(l3_results)
    if len(l3_df) > 1:
        _, fdr, _, _ = multipletests(l3_df['p_value'], method='fdr_bh')
        l3_df['FDR'] = fdr
        l3_df = l3_df.sort_values(['gene', 'partial_R2'], ascending=[True, False])
        print(f"  {len(l3_df)} phosphosites across {l3_df['gene'].nunique()} genes")
        for gene in driver_genes:
            g3 = l3_df[l3_df['gene'] == gene]
            if len(g3) == 0:
                continue
            top = g3.head(2)
            for _, r in top.iterrows():
                sig = "***" if r['FDR'] < 0.001 else "**" if r['FDR'] < 0.01 else "*" if r['FDR'] < 0.05 else ""
                print(f"  {r['gene']:12s}  {str(r['site'])[:18]:18s}  β={r['beta']:+.3f}  pR²={r['partial_R2']:.3f}  n={r['n']:.0f}  {sig}")
    elif len(l3_df) == 1:
        l3_df['FDR'] = l3_df['p_value']
    else:
        print("  No phosphosite data for selected drivers")
    l3_df.to_csv(f"{OUTPUT_DIR}/{cancer_name}_regression_layer3.csv", index=False)

    # ── Continuous TS ──
    print(f"\n[5] Continuous Transmission Score...")
    ts_results = []

    for gene in driver_genes:
        row = {'gene': gene}

        # L1
        g1 = l1_df[l1_df['gene'] == gene] if len(l1_df) > 0 else pd.DataFrame()
        row['beta_L1'] = g1['beta'].values[0] if len(g1) > 0 else np.nan
        row['R2_L1'] = g1['partial_R2'].values[0] if len(g1) > 0 else np.nan
        row['p_L1'] = g1['FDR'].values[0] if len(g1) > 0 else np.nan

        # L2
        g2 = l2_df[(l2_df['gene'] == gene) & (l2_df['group'] == 'driver')] if len(l2_df) > 0 else pd.DataFrame()
        row['beta_L2'] = g2['beta'].values[0] if len(g2) > 0 else np.nan
        row['R2_L2'] = g2['partial_R2'].values[0] if len(g2) > 0 else np.nan
        row['p_L2'] = g2['FDR'].values[0] if len(g2) > 0 else np.nan

        # L3 (median across sites)
        g3 = l3_df[l3_df['gene'] == gene] if len(l3_df) > 0 else pd.DataFrame()
        if len(g3) > 0:
            row['beta_L3'] = g3['beta'].median()
            row['R2_L3'] = g3['partial_R2'].median()
            row['n_sites'] = len(g3)
        else:
            row['beta_L3'] = np.nan
            row['R2_L3'] = np.nan
            row['n_sites'] = 0

        # 3 TS definitions
        b1 = row['beta_L1'] if not np.isnan(row.get('beta_L1', np.nan)) else 0
        b2 = row['beta_L2'] if not np.isnan(row.get('beta_L2', np.nan)) else 0
        b3 = row['beta_L3'] if not np.isnan(row.get('beta_L3', np.nan)) else 0
        r1 = row['R2_L1'] if not np.isnan(row.get('R2_L1', np.nan)) else 0
        r2 = row['R2_L2'] if not np.isnan(row.get('R2_L2', np.nan)) else 0
        r3 = row['R2_L3'] if not np.isnan(row.get('R2_L3', np.nan)) else 0

        # 3-layer versions (only if all 3 layers have data)
        if row['n_sites'] > 0 and not np.isnan(row.get('beta_L1', np.nan)):
            row['TS_product_3L'] = b1 * b2 * b3
            row['TS_log_3L'] = np.log1p(abs(b1)) + np.log1p(abs(b2)) + np.log1p(abs(b3))
            row['TS_R2_3L'] = r1 * r2 * r3
        else:
            row['TS_product_3L'] = np.nan
            row['TS_log_3L'] = np.nan
            row['TS_R2_3L'] = np.nan

        # 2-layer versions (always computed if L1 available)
        if not np.isnan(row.get('beta_L1', np.nan)):
            row['TS_product_2L'] = b1 * b2
            row['TS_log_2L'] = np.log1p(abs(b1)) + np.log1p(abs(b2))
            row['TS_R2_2L'] = r1 * r2
        else:
            row['TS_product_2L'] = np.nan
            row['TS_log_2L'] = np.nan
            row['TS_R2_2L'] = np.nan

        # Old metrics for comparison
        old_ts = pd.read_csv(f"{OUTPUT_DIR}/{cancer_name}_transmission_summary.csv")
        old_row = old_ts[old_ts['gene'] == gene]
        if len(old_row) > 0:
            row['old_TS_mult'] = old_row['TS_mult'].values[0]
            row['old_TS_add'] = old_row['TS_additive'].values[0]
            row['old_rho'] = old_row['layer2_rho'].values[0]
            row['old_d'] = old_row['layer1_d'].values[0]
        else:
            row['old_TS_mult'] = np.nan
            row['old_TS_add'] = np.nan
            row['old_rho'] = np.nan
            row['old_d'] = np.nan

        ts_results.append(row)

    ts_df = pd.DataFrame(ts_results)
    ts_df = ts_df.sort_values('TS_R2_2L', ascending=False)

    print(f"\n  {'Gene':12s}  {'β_L1':>7s}  {'β_L2':>7s}  {'R²_L1':>7s}  {'R²_L2':>7s}  {'TS_R2':>8s}  {'old_TS':>6s}  {'old_ρ':>7s}")
    print(f"  {'-'*75}")
    for _, r in ts_df.iterrows():
        b1s = f"{r['beta_L1']:+7.3f}" if not np.isnan(r['beta_L1']) else "    N/A"
        b2s = f"{r['beta_L2']:+7.3f}" if not np.isnan(r['beta_L2']) else "    N/A"
        r1s = f"{r['R2_L1']:7.4f}" if not np.isnan(r['R2_L1']) else "    N/A"
        r2s = f"{r['R2_L2']:7.4f}" if not np.isnan(r['R2_L2']) else "    N/A"
        ts_s = f"{r['TS_R2_2L']:8.5f}" if not np.isnan(r['TS_R2_2L']) else "     N/A"
        old_s = f"{r['old_TS_mult']:6.0f}" if not np.isnan(r['old_TS_mult']) else "   N/A"
        rho_s = f"{r['old_rho']:+7.3f}" if not np.isnan(r['old_rho']) else "    N/A"
        print(f"  {r['gene']:12s}  {b1s}  {b2s}  {r1s}  {r2s}  {ts_s}  {old_s}  {rho_s}")

    ts_df.to_csv(f"{OUTPUT_DIR}/{cancer_name}_ts_continuous.csv", index=False)

    # ── Summary statistics ──
    print(f"\n  ── Summary ──")
    valid_ts = ts_df['TS_R2_2L'].dropna()
    if len(valid_ts) > 0:
        print(f"  TS_R2 (2-layer): mean={valid_ts.mean():.5f}, max={valid_ts.max():.5f}, "
              f"median={valid_ts.median():.5f}")
        print(f"  Top gene: {ts_df.iloc[0]['gene']} (TS_R2={ts_df.iloc[0]['TS_R2_2L']:.5f})")

    # Correlation with old metrics
    both = ts_df.dropna(subset=['TS_R2_2L', 'old_TS_add'])
    if len(both) >= 5:
        rho_corr, p_corr = stats.spearmanr(both['TS_R2_2L'], both['old_TS_add'])
        print(f"  Spearman(TS_R2, old_TS_add): ρ={rho_corr:+.3f}, p={p_corr:.3f}")

    print(f"\n{'='*70}")
    print(f"  {cancer_name} Phase 1 complete.")
    print(f"{'='*70}")

    return ts_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer', default='LUAD', help='Cancer type')
    parser.add_argument('--all', action='store_true', help='Run all 7 cancers')
    args = parser.parse_args()

    if args.all:
        all_ts = []
        for cancer in ['COAD', 'LUAD', 'UCEC', 'CCRCC', 'OV', 'BRCA', 'PDAC']:
            ts = run_phase1(cancer)
            ts['cancer'] = cancer
            all_ts.append(ts)

        # Pan-cancer integration
        pan_ts = pd.concat(all_ts, ignore_index=True)
        pan_ts.to_csv(f"{OUTPUT_DIR}/pancancer_ts_continuous.csv", index=False)

        # Ranking
        ranking = pan_ts.dropna(subset=['TS_R2_2L']).sort_values('TS_R2_2L', ascending=False)
        ranking[['cancer', 'gene', 'beta_L1', 'beta_L2', 'R2_L1', 'R2_L2',
                 'TS_R2_2L', 'TS_product_2L', 'TS_log_2L',
                 'old_TS_mult', 'old_TS_add']].head(30).to_csv(
            f"{OUTPUT_DIR}/pancancer_ts_ranking.csv", index=False)

        print(f"\n{'='*70}")
        print(f"  Top 15 genes by TS_R2 (2-layer) across 7 cancers:")
        print(f"  {'Cancer':6s}  {'Gene':12s}  {'β_L1':>7s}  {'β_L2':>7s}  {'R²_L1':>7s}  {'R²_L2':>7s}  {'TS_R2':>8s}  {'old_TS':>6s}")
        print(f"  {'-'*70}")
        for _, r in ranking.head(15).iterrows():
            print(f"  {r['cancer']:6s}  {r['gene']:12s}  {r['beta_L1']:+7.3f}  {r['beta_L2']:+7.3f}  "
                  f"{r['R2_L1']:7.4f}  {r['R2_L2']:7.4f}  {r['TS_R2_2L']:8.5f}  {r['old_TS_mult']:6.0f}")
        print(f"{'='*70}")
    else:
        run_phase1(args.cancer)
