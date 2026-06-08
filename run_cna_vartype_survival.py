#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CNA / variant-type / survival analyses (Fig. 5)
#
# Manuscript: Pan-cancer quantification of driver alteration transmission across
#             molecular layers reveals limited propagation to protein abundance (IJC-26-1558)
# Author:     Hisashi Nakano, PhD - Department of Health Data Science,
#             Niigata University of Health and Welfare, Niigata, Japan
# ORCID:      0000-0002-9023-880X

"""
Task 1: CNA-based Transmission Score (CNA→mRNA→Protein)
Task 2: Variant Type TS (Missense vs Truncating)
Task 3: Survival Analysis (Cox regression, TS_R2 vs HR)
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

# Task 1: CNA target genes (amplification/deletion-driven)
CNA_TARGETS = ['ERBB2', 'EGFR', 'MYC', 'CDKN2A', 'PTEN', 'VHL']

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data(cancer_name, need_cnv=False, need_survival=False):
    """Load cptac data. Returns dict of dataframes."""
    stderr_bak = sys.stderr
    sys.stderr = io.StringIO()

    ds = CANCER_CLASSES[cancer_name]()

    prot = ds.get_proteomics(source="umich")
    prot_tumor = prot[~prot.index.str.endswith('.N')].copy()
    prot_tumor.columns = prot_tumor.columns.get_level_values(0)
    prot_tumor = prot_tumor.T.groupby(level=0).mean().T

    trans = ds.get_transcriptomics(source="bcm")
    trans_tumor = trans[~trans.index.str.endswith('.N')].copy()
    trans_tumor.columns = trans_tumor.columns.get_level_values(0)
    trans_tumor = trans_tumor.T.groupby(level=0).mean().T

    mut = ds.get_somatic_mutation(source=MUT_SOURCES[cancer_name])

    common = prot_tumor.index.intersection(trans_tumor.index)

    data = {
        'prot': prot_tumor.loc[common],
        'trans': trans_tumor.loc[common],
        'mut': mut,
        'common': common,
        'ds': ds,
    }

    if need_cnv:
        cnv = ds.get_CNV(source='bcm')
        cnv_tumor = cnv[~cnv.index.str.endswith('.N')].copy()
        if isinstance(cnv_tumor.columns, pd.MultiIndex):
            cnv_tumor.columns = cnv_tumor.columns.get_level_values(0)
        cnv_tumor = cnv_tumor.T.groupby(level=0).mean().T
        data['cnv'] = cnv_tumor

    if need_survival:
        try:
            clinical = ds.get_clinical(source="mssm")
            data['clinical'] = clinical
        except:
            data['clinical'] = None

    sys.stderr = stderr_bak
    return data


# ═══════════════════════════════════════════════════════════════════════════
# TASK 1: CNA-BASED TRANSMISSION SCORE
# ═══════════════════════════════════════════════════════════════════════════

def cna_layer1(cnv_vals, mrna_vals):
    """Layer 1 CNA: mRNA ~ CNA (OLS). Returns beta, R², p, n."""
    common = cnv_vals.dropna().index.intersection(mrna_vals.dropna().index)
    if len(common) < 10:
        return None
    y = mrna_vals.loc[common]
    x = cnv_vals.loc[common]
    y_z = (y - y.mean()) / y.std() if y.std() > 0 else y
    x_z = ((x - x.mean()) / x.std()).rename('cna') if x.std() > 0 else x.rename('cna')
    X = sm.add_constant(x_z.to_frame())
    try:
        model = sm.OLS(y_z, X, missing='drop').fit()
    except:
        return None
    beta = model.params['cna'] if 'cna' in model.params.index else model.params.iloc[1]
    se = model.bse['cna'] if 'cna' in model.bse.index else model.bse.iloc[1]
    pval = model.pvalues['cna'] if 'cna' in model.pvalues.index else model.pvalues.iloc[1]
    return {
        'beta': beta, 'se': se, 'p_value': pval,
        'R2': max(model.rsquared, 0), 'n': int(model.nobs)
    }


def cna_layer2(protein_vals, mrna_vals):
    """Layer 2: Protein ~ mRNA (same as regular L2, simplified)."""
    common = protein_vals.dropna().index.intersection(mrna_vals.dropna().index)
    if len(common) < 10:
        return None
    y = protein_vals.loc[common]
    x = mrna_vals.loc[common]
    y_z = (y - y.mean()) / y.std() if y.std() > 0 else y
    x_z = ((x - x.mean()) / x.std()).rename('mrna') if x.std() > 0 else x.rename('mrna')
    X = sm.add_constant(x_z.to_frame())
    try:
        model = sm.OLS(y_z, X, missing='drop').fit()
    except:
        return None
    beta = model.params['mrna'] if 'mrna' in model.params.index else model.params.iloc[1]
    return {
        'beta': beta, 'R2': max(model.rsquared, 0), 'n': int(model.nobs)
    }


def run_task1():
    """CNA→mRNA→Protein transmission for amplification/deletion genes."""
    print("=" * 70)
    print("  TASK 1: CNA-BASED TRANSMISSION SCORE")
    print("=" * 70)

    all_results = []

    for cancer_name in ['LUAD', 'COAD', 'UCEC', 'CCRCC', 'OV', 'BRCA', 'PDAC']:
        print(f"\n── {cancer_name} ──")
        data = load_data(cancer_name, need_cnv=True)
        cnv = data['cnv']
        trans = data['trans']
        prot = data['prot']
        common = data['common']

        # Intersect with CNV samples
        cnv_common = common.intersection(cnv.index)
        if len(cnv_common) < 10:
            print(f"  Skip: only {len(cnv_common)} CNV samples")
            continue

        for gene in CNA_TARGETS:
            if gene not in cnv.columns or gene not in trans.columns or gene not in prot.columns:
                continue

            # Layer 1: CNA → mRNA
            l1 = cna_layer1(cnv.loc[cnv_common, gene], trans.loc[cnv_common, gene])
            if l1 is None:
                continue

            # Layer 2: mRNA → Protein
            l2 = cna_layer2(prot.loc[cnv_common, gene], trans.loc[cnv_common, gene])
            if l2 is None:
                continue

            # CNA TS
            ts_r2 = l1['R2'] * l2['R2']

            # Also load mutation-based TS for comparison
            try:
                ts_file = pd.read_csv(f"{OUTPUT_DIR}/{cancer_name}_ts_continuous.csv")
                mut_row = ts_file[ts_file['gene'] == gene]
                mut_ts_r2 = mut_row['TS_R2_2L'].values[0] if len(mut_row) > 0 else np.nan
            except:
                mut_ts_r2 = np.nan

            row = {
                'cancer': cancer_name, 'gene': gene,
                'cna_beta_L1': l1['beta'], 'cna_R2_L1': l1['R2'],
                'cna_p_L1': l1['p_value'], 'cna_n_L1': l1['n'],
                'cna_beta_L2': l2['beta'], 'cna_R2_L2': l2['R2'], 'cna_n_L2': l2['n'],
                'cna_TS_R2': ts_r2,
                'mut_TS_R2': mut_ts_r2,
            }
            all_results.append(row)

    cna_df = pd.DataFrame(all_results)
    if len(cna_df) == 0:
        print("  No CNA results!")
        return cna_df

    cna_df = cna_df.sort_values('cna_TS_R2', ascending=False)
    cna_df.to_csv(f"{OUTPUT_DIR}/cna_transmission.csv", index=False)

    # Print results
    print(f"\n{'='*70}")
    print(f"  CNA Transmission Summary ({len(cna_df)} gene-cancer pairs)")
    print(f"  {'Cancer':6s}  {'Gene':8s}  {'β_CNA':>7s}  {'R²_L1':>7s}  {'R²_L2':>7s}  {'CNA_TS':>8s}  {'Mut_TS':>8s}  {'Ratio':>6s}")
    print(f"  {'-'*68}")
    for _, r in cna_df.iterrows():
        mut_s = f"{r['mut_TS_R2']:8.5f}" if not np.isnan(r['mut_TS_R2']) else "     N/A"
        ratio = r['cna_TS_R2'] / r['mut_TS_R2'] if not np.isnan(r['mut_TS_R2']) and r['mut_TS_R2'] > 0 else np.nan
        ratio_s = f"{ratio:6.1f}" if not np.isnan(ratio) else "   N/A"
        print(f"  {r['cancer']:6s}  {r['gene']:8s}  {r['cna_beta_L1']:+7.3f}  "
              f"{r['cna_R2_L1']:7.4f}  {r['cna_R2_L2']:7.4f}  {r['cna_TS_R2']:8.5f}  {mut_s}  {ratio_s}")

    # Summary stats
    print(f"\n  CNA vs Mutation TS comparison:")
    both = cna_df.dropna(subset=['cna_TS_R2', 'mut_TS_R2'])
    both = both[both['mut_TS_R2'] > 0]
    if len(both) >= 3:
        print(f"    CNA TS_R2 median:  {both['cna_TS_R2'].median():.5f}")
        print(f"    Mut TS_R2 median:  {both['mut_TS_R2'].median():.5f}")
        ratio_med = both['cna_TS_R2'].median() / both['mut_TS_R2'].median()
        print(f"    Ratio (CNA/Mut):   {ratio_med:.1f}x")
        if len(both) >= 5:
            rho, p = stats.spearmanr(both['cna_TS_R2'], both['mut_TS_R2'])
            print(f"    Spearman: ρ={rho:+.3f}, p={p:.3f}")

    # Per-gene summary
    print(f"\n  Per-gene CNA R²_L1 (CNA→mRNA) across cancers:")
    for gene in CNA_TARGETS:
        gd = cna_df[cna_df['gene'] == gene]
        if len(gd) > 0:
            cancers_str = ', '.join([f"{r['cancer']}={r['cna_R2_L1']:.3f}" for _, r in gd.iterrows()])
            print(f"    {gene:8s}: {cancers_str}")

    return cna_df


# ═══════════════════════════════════════════════════════════════════════════
# TASK 2: VARIANT TYPE TRANSMISSION SCORE
# ═══════════════════════════════════════════════════════════════════════════

def get_mutation_type_patients(mut, gene):
    """Return (missense_set, truncating_set) patient IDs."""
    non_silent = mut[~mut['Mutation'].isin(['Silent', 'Intron', "3'UTR", "5'UTR",
                                            "3'Flank", "5'Flank", 'IGR', 'RNA'])]
    gene_mut = non_silent[non_silent['Gene'] == gene]
    trunc_types = ['Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins',
                   'Splice_Site', 'Splice_Region']
    missense = set(gene_mut[gene_mut['Mutation'] == 'Missense_Mutation'].index.unique())
    truncating = set(gene_mut[gene_mut['Mutation'].isin(trunc_types)].index.unique())
    # Patients with both → classify as truncating (more severe)
    missense_only = missense - truncating
    return missense_only, truncating


def vartype_regression_l1(mrna_vals, mut_patients, wt_patients):
    """Layer 1 for a specific variant group vs WT."""
    combined = list(mut_patients) + list(wt_patients)
    mrna = mrna_vals.loc[mrna_vals.index.intersection(pd.Index(combined))].dropna()
    if len(mrna) < 10:
        return None
    y = (mrna - mrna.mean()) / mrna.std()
    x = pd.Series(0, index=y.index, name='mutation')
    for p in mut_patients:
        if p in x.index:
            x[p] = 1
    n_mut = int(x.sum())
    if n_mut < 3 or (len(y) - n_mut) < 3:
        return None
    X = sm.add_constant(x.to_frame())
    try:
        model = sm.OLS(y, X).fit()
    except:
        return None
    beta = model.params['mutation']
    return {
        'beta': beta, 'se': model.bse['mutation'],
        'p_value': model.pvalues['mutation'],
        'R2': max(model.rsquared, 0),
        'n': int(model.nobs), 'n_mut': n_mut
    }


def vartype_regression_l2(protein_vals, mrna_vals):
    """Layer 2 (same for all — mRNA→Protein)."""
    common = protein_vals.dropna().index.intersection(mrna_vals.dropna().index)
    if len(common) < 10:
        return None
    y = protein_vals.loc[common]
    x = mrna_vals.loc[common]
    y_z = (y - y.mean()) / y.std() if y.std() > 0 else y
    x_z = ((x - x.mean()) / x.std()).rename('mrna') if x.std() > 0 else x.rename('mrna')
    X = sm.add_constant(x_z.to_frame())
    try:
        model = sm.OLS(y_z, X, missing='drop').fit()
    except:
        return None
    return {
        'beta': model.params['mrna'], 'R2': max(model.rsquared, 0),
        'n': int(model.nobs)
    }


def run_task2():
    """Missense vs Truncating TS comparison across all 7 cancers."""
    print(f"\n\n{'='*70}")
    print("  TASK 2: VARIANT TYPE TRANSMISSION SCORE")
    print("=" * 70)

    all_results = []

    for cancer_name in ['LUAD', 'COAD', 'UCEC', 'CCRCC', 'OV', 'BRCA', 'PDAC']:
        print(f"\n── {cancer_name} ──")
        data = load_data(cancer_name)
        trans = data['trans']
        prot = data['prot']
        mut = data['mut']
        common = data['common']
        all_patients = set(common)

        drivers = CANCER_DRIVERS[cancer_name]
        available = set(prot.columns) & set(trans.columns)

        for gene in drivers:
            if gene not in available:
                continue

            missense, truncating = get_mutation_type_patients(mut, gene)
            missense = missense & all_patients
            truncating = truncating & all_patients
            wildtype = all_patients - missense - truncating

            # Layer 2 (shared across variant types)
            l2 = vartype_regression_l2(prot[gene], trans[gene])
            r2_l2 = l2['R2'] if l2 else 0

            # Missense vs WT
            if len(missense) >= 3 and len(wildtype) >= 3:
                l1_mis = vartype_regression_l1(trans[gene], missense, wildtype)
                if l1_mis:
                    ts = l1_mis['R2'] * r2_l2
                    all_results.append({
                        'cancer': cancer_name, 'gene': gene,
                        'variant_type': 'missense',
                        'beta_L1': l1_mis['beta'], 'R2_L1': l1_mis['R2'],
                        'p_L1': l1_mis['p_value'],
                        'beta_L2': l2['beta'] if l2 else np.nan,
                        'R2_L2': r2_l2,
                        'TS_R2': ts,
                        'n': l1_mis['n'], 'n_mut': l1_mis['n_mut'],
                    })

            # Truncating vs WT
            if len(truncating) >= 3 and len(wildtype) >= 3:
                l1_trun = vartype_regression_l1(trans[gene], truncating, wildtype)
                if l1_trun:
                    ts = l1_trun['R2'] * r2_l2
                    all_results.append({
                        'cancer': cancer_name, 'gene': gene,
                        'variant_type': 'truncating',
                        'beta_L1': l1_trun['beta'], 'R2_L1': l1_trun['R2'],
                        'p_L1': l1_trun['p_value'],
                        'beta_L2': l2['beta'] if l2 else np.nan,
                        'R2_L2': r2_l2,
                        'TS_R2': ts,
                        'n': l1_trun['n'], 'n_mut': l1_trun['n_mut'],
                    })

            # Near-100% mutation (e.g., KRAS in PDAC): missense vs truncating directly
            if len(wildtype) < 5 and len(missense) >= 3 and len(truncating) >= 3:
                l1_direct = vartype_regression_l1(trans[gene], truncating, missense)
                if l1_direct:
                    ts = l1_direct['R2'] * r2_l2
                    all_results.append({
                        'cancer': cancer_name, 'gene': gene,
                        'variant_type': 'truncating_vs_missense',
                        'beta_L1': l1_direct['beta'], 'R2_L1': l1_direct['R2'],
                        'p_L1': l1_direct['p_value'],
                        'beta_L2': l2['beta'] if l2 else np.nan,
                        'R2_L2': r2_l2,
                        'TS_R2': ts,
                        'n': l1_direct['n'], 'n_mut': l1_direct['n_mut'],
                    })

    vt_df = pd.DataFrame(all_results)
    if len(vt_df) == 0:
        print("  No variant type results!")
        return vt_df

    vt_df.to_csv(f"{OUTPUT_DIR}/vartype_transmission.csv", index=False)

    # Print comparison for genes with both types
    print(f"\n{'='*70}")
    print(f"  Variant Type TS Comparison")
    print(f"  {'Cancer':6s}  {'Gene':10s}  {'Type':12s}  {'β_L1':>7s}  {'R²_L1':>7s}  {'TS_R2':>8s}  {'n_mut':>5s}")
    print(f"  {'-'*65}")
    for cancer_name in ['LUAD', 'COAD', 'UCEC', 'CCRCC', 'OV', 'BRCA', 'PDAC']:
        cd = vt_df[vt_df['cancer'] == cancer_name].sort_values(['gene', 'variant_type'])
        for _, r in cd.iterrows():
            print(f"  {r['cancer']:6s}  {r['gene']:10s}  {r['variant_type']:12s}  "
                  f"{r['beta_L1']:+7.3f}  {r['R2_L1']:7.4f}  {r['TS_R2']:8.5f}  {r['n_mut']:5.0f}")

    # TP53 systematic comparison
    tp53 = vt_df[vt_df['gene'] == 'TP53']
    if len(tp53) > 0:
        print(f"\n  ── TP53 Systematic Comparison ──")
        print(f"  {'Cancer':6s}  {'Type':12s}  {'β_L1':>7s}  {'R²_L1':>7s}  {'TS_R2':>8s}  {'n_mut':>5s}")
        print(f"  {'-'*55}")
        for cancer_name in ['LUAD', 'COAD', 'UCEC', 'CCRCC', 'OV', 'BRCA', 'PDAC']:
            tp = tp53[tp53['cancer'] == cancer_name].sort_values('variant_type')
            for _, r in tp.iterrows():
                print(f"  {r['cancer']:6s}  {r['variant_type']:12s}  "
                      f"{r['beta_L1']:+7.3f}  {r['R2_L1']:7.4f}  {r['TS_R2']:8.5f}  {r['n_mut']:5.0f}")

    # Summary: missense vs truncating median comparison
    mis = vt_df[vt_df['variant_type'] == 'missense']
    trun = vt_df[vt_df['variant_type'] == 'truncating']
    # Match gene-cancer pairs that have both
    pairs = set(zip(mis['cancer'], mis['gene'])) & set(zip(trun['cancer'], trun['gene']))
    if pairs:
        mis_ts = []
        trun_ts = []
        for c, g in pairs:
            m = mis[(mis['cancer'] == c) & (mis['gene'] == g)]['TS_R2'].values[0]
            t = trun[(trun['cancer'] == c) & (trun['gene'] == g)]['TS_R2'].values[0]
            mis_ts.append(m)
            trun_ts.append(t)

        print(f"\n  Paired comparison (N={len(pairs)} gene-cancer pairs with both types):")
        print(f"    Missense TS_R2 median:    {np.median(mis_ts):.5f}")
        print(f"    Truncating TS_R2 median:  {np.median(trun_ts):.5f}")
        if len(pairs) >= 5:
            stat, p = stats.wilcoxon(mis_ts, trun_ts)
            print(f"    Wilcoxon signed-rank: p={p:.3f}")
        else:
            stat, p = stats.mannwhitneyu(mis_ts, trun_ts, alternative='two-sided')
            print(f"    Mann-Whitney: p={p:.3f}")

    # β direction comparison
    print(f"\n  β_L1 direction (missense vs truncating):")
    for vt in ['missense', 'truncating']:
        sub = vt_df[vt_df['variant_type'] == vt]
        n_neg = (sub['beta_L1'] < 0).sum()
        n_pos = (sub['beta_L1'] > 0).sum()
        print(f"    {vt:12s}: {n_neg} negative / {n_pos} positive β_L1 (of {len(sub)})")

    return vt_df


# ═══════════════════════════════════════════════════════════════════════════
# TASK 3: SURVIVAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def run_task3():
    """Cox regression per gene per cancer, correlate with TS_R2."""
    print(f"\n\n{'='*70}")
    print("  TASK 3: SURVIVAL ANALYSIS")
    print("=" * 70)

    try:
        from lifelines import CoxPHFitter
    except ImportError:
        print("  Installing lifelines...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lifelines', '-q'])
        from lifelines import CoxPHFitter

    all_cox = []

    # Skip BRCA (only 2 deaths)
    cancers_for_surv = ['LUAD', 'COAD', 'UCEC', 'CCRCC', 'OV', 'PDAC']

    for cancer_name in cancers_for_surv:
        print(f"\n── {cancer_name} ──")
        data = load_data(cancer_name, need_survival=True)
        prot = data['prot']
        trans = data['trans']
        clinical = data['clinical']
        common = data['common']

        if clinical is None:
            print(f"  No clinical data!")
            continue

        # Extract OS — prefer exact column names first
        os_col = None
        status_col = None
        if 'Overall survival, days' in clinical.columns:
            os_col = 'Overall survival, days'
        if 'Survival status (1, dead; 0, alive)' in clinical.columns:
            status_col = 'Survival status (1, dead; 0, alive)'

        if os_col is None:
            for c in clinical.columns:
                cl = c.lower()
                if 'overall survival' in cl and 'day' in cl and 'collection' not in cl:
                    os_col = c
                    break
            if os_col is None:
                for c in clinical.columns:
                    if 'survival' in c.lower() and 'day' in c.lower():
                        os_col = c
                        break
        if status_col is None:
            for c in clinical.columns:
                if 'survival status' in c.lower() or 'vital_status' in c.lower():
                    status_col = c
                    break

        if os_col is None or status_col is None:
            print(f"  Missing OS columns (os_col={os_col}, status_col={status_col})")
            print(f"  Available columns: {list(clinical.columns[:20])}")
            continue

        surv = clinical[[os_col, status_col]].copy()
        surv.columns = ['os_days', 'os_status']
        surv['os_days'] = pd.to_numeric(surv['os_days'], errors='coerce')

        # Parse status: 1=dead, 0=alive (various formats)
        status_raw = surv['os_status'].astype(str).str.lower()
        surv['event'] = np.where(
            status_raw.isin(['1', 'dead', 'deceased', '1.0']), 1,
            np.where(status_raw.isin(['0', 'alive', 'living', '0.0']), 0, np.nan)
        )
        surv['event'] = pd.to_numeric(surv['event'], errors='coerce')
        surv = surv.dropna(subset=['os_days', 'event'])
        surv = surv[surv['os_days'] > 0]

        # Intersect with common samples
        surv_idx = surv.index.intersection(common)
        n_events = int(surv.loc[surv_idx, 'event'].sum())
        print(f"  OS data: {len(surv_idx)} patients, {n_events} events")

        if n_events < 5:
            print(f"  Skip: too few events ({n_events})")
            continue

        drivers = CANCER_DRIVERS[cancer_name]
        available = set(prot.columns) & set(trans.columns)

        for gene in drivers:
            if gene not in available:
                continue

            # Protein expression (z-score) as predictor
            prot_gene = prot.loc[surv_idx, gene].dropna()
            valid = prot_gene.index.intersection(surv.index)
            if len(valid) < 20:
                continue

            cox_df = pd.DataFrame({
                'os_days': surv.loc[valid, 'os_days'],
                'event': surv.loc[valid, 'event'],
                'protein_z': (prot_gene.loc[valid] - prot_gene.loc[valid].mean()) / prot_gene.loc[valid].std()
            }).dropna()

            if len(cox_df) < 20 or cox_df['event'].sum() < 3:
                continue

            try:
                cph = CoxPHFitter()
                cph.fit(cox_df, duration_col='os_days', event_col='event')
                hr = np.exp(cph.params_['protein_z'])
                pval = cph.summary.loc['protein_z', 'p']
                ci_low = np.exp(cph.confidence_intervals_.loc['protein_z'].iloc[0])
                ci_high = np.exp(cph.confidence_intervals_.loc['protein_z'].iloc[1])

                # Load TS_R2
                try:
                    ts_file = pd.read_csv(f"{OUTPUT_DIR}/{cancer_name}_ts_continuous.csv")
                    ts_row = ts_file[ts_file['gene'] == gene]
                    ts_r2 = ts_row['TS_R2_2L'].values[0] if len(ts_row) > 0 else np.nan
                except:
                    ts_r2 = np.nan

                all_cox.append({
                    'cancer': cancer_name, 'gene': gene,
                    'HR': hr, 'HR_ci_low': ci_low, 'HR_ci_high': ci_high,
                    'cox_p': pval, 'log_HR': np.log(hr),
                    'abs_log_HR': abs(np.log(hr)),
                    'TS_R2': ts_r2,
                    'n': len(cox_df), 'n_events': int(cox_df['event'].sum()),
                })
            except Exception as e:
                print(f"  {gene}: Cox failed ({str(e)[:50]})")

    cox_df = pd.DataFrame(all_cox)
    if len(cox_df) == 0:
        print("  No Cox results!")
        return cox_df

    cox_df.to_csv(f"{OUTPUT_DIR}/survival_cox.csv", index=False)

    # FDR correction
    _, fdr, _, _ = multipletests(cox_df['cox_p'], method='fdr_bh')
    cox_df['FDR'] = fdr

    # Print results
    print(f"\n{'='*70}")
    print(f"  Cox Regression Results ({len(cox_df)} gene-cancer pairs)")
    print(f"  {'Cancer':6s}  {'Gene':10s}  {'HR':>6s}  {'95% CI':>16s}  {'p':>10s}  {'FDR':>10s}  {'TS_R2':>8s}")
    print(f"  {'-'*75}")
    for _, r in cox_df.sort_values('cox_p').head(30).iterrows():
        sig = "***" if r['FDR'] < 0.001 else "**" if r['FDR'] < 0.01 else "*" if r['FDR'] < 0.05 else ""
        ts_s = f"{r['TS_R2']:8.5f}" if not np.isnan(r['TS_R2']) else "     N/A"
        print(f"  {r['cancer']:6s}  {r['gene']:10s}  {r['HR']:6.3f}  "
              f"({r['HR_ci_low']:.2f}-{r['HR_ci_high']:.2f})  "
              f"{r['cox_p']:10.3e}  {r['FDR']:10.3e}  {ts_s}  {sig}")

    # Significant results
    sig_cox = cox_df[cox_df['FDR'] < 0.05]
    print(f"\n  Significant (FDR<0.05): {len(sig_cox)} gene-cancer pairs")
    nom_sig = cox_df[cox_df['cox_p'] < 0.05]
    print(f"  Nominally significant (p<0.05): {len(nom_sig)} gene-cancer pairs")

    # TS_R2 vs |log(HR)| correlation
    both = cox_df.dropna(subset=['TS_R2', 'abs_log_HR'])
    both = both[both['TS_R2'] > 0]
    if len(both) >= 5:
        rho, p = stats.spearmanr(both['TS_R2'], both['abs_log_HR'])
        print(f"\n  TS_R2 vs |log(HR)| correlation:")
        print(f"    N = {len(both)}, Spearman ρ = {rho:+.3f}, p = {p:.3f}")

        # Also Pearson on log scale
        r_pear, p_pear = stats.pearsonr(np.log10(both['TS_R2'] + 1e-10), both['abs_log_HR'])
        print(f"    Pearson (log10 TS_R2): r = {r_pear:+.3f}, p = {p_pear:.3f}")

    # Per-cancer summary
    print(f"\n  Per-cancer summary:")
    for cancer in cancers_for_surv:
        cc = cox_df[cox_df['cancer'] == cancer]
        if len(cc) > 0:
            n_sig = (cc['cox_p'] < 0.05).sum()
            print(f"    {cancer:6s}: {len(cc):2d} genes tested, {n_sig} nominally significant, "
                  f"median |log(HR)| = {cc['abs_log_HR'].median():.3f}")

    return cox_df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['1', '2', '3', 'all'], default='all')
    args = parser.parse_args()

    if args.task in ('1', 'all'):
        cna_df = run_task1()

    if args.task in ('2', 'all'):
        vt_df = run_task2()

    if args.task in ('3', 'all'):
        cox_df = run_task3()

    print(f"\n{'='*70}")
    print("  ALL TASKS COMPLETE")
    print(f"{'='*70}")
