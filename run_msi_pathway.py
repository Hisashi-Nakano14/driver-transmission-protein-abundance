#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MSI stratification and pathway attenuation
#
# Manuscript: Pan-cancer quantification of driver alteration transmission across
#             molecular layers reveals limited propagation to protein abundance (IJC-26-1558)
# Author:     Hisashi Nakano, PhD - Department of Health Data Science,
#             Niigata University of Health and Welfare, Niigata, Japan
# ORCID:      0000-0002-9023-880X

"""
Task A: MSI stratification (UCEC + COAD)
Task B: Pathway/Kinase Activity Score
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

# ═══════════════════════════════════════════════════════════════════════════
# TASK A: MSI STRATIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def estimate_msi_status(ds, mut_source, cancer_name):
    """Estimate MSI from TMB (threshold = 10 mut/Mb)."""
    mut = ds.get_somatic_mutation(source=mut_source)
    tmb = mut.groupby(mut.index).size()
    tmb_per_mb = tmb / 30  # ~30Mb exome

    msi_status = pd.Series('MSS', index=tmb.index)
    msi_status[tmb_per_mb > 10] = 'MSI-H'

    n_h = (msi_status == 'MSI-H').sum()
    n_s = (msi_status == 'MSS').sum()
    print(f"  {cancer_name} MSI: MSI-H={n_h} ({n_h/(n_h+n_s):.1%}), MSS={n_s}")
    return msi_status


def run_msi_stratification(cancer_name, CancerClass, mut_source):
    """MSI-stratified Layer 2 regression for target genes."""
    print(f"\n{'='*70}")
    print(f"  Task A: MSI Stratification — {cancer_name}")
    print(f"{'='*70}")

    stderr_bak = sys.stderr
    sys.stderr = io.StringIO()
    ds = CancerClass()

    prot = ds.get_proteomics(source="umich")
    prot_tumor = prot[~prot.index.str.endswith('.N')].copy()
    prot_tumor.columns = prot_tumor.columns.get_level_values(0)
    prot_tumor = prot_tumor.T.groupby(level=0).mean().T

    trans = ds.get_transcriptomics(source="bcm")
    trans_tumor = trans[~trans.index.str.endswith('.N')].copy()
    trans_tumor.columns = trans_tumor.columns.get_level_values(0)
    trans_tumor = trans_tumor.T.groupby(level=0).mean().T

    common = prot_tumor.index.intersection(trans_tumor.index)
    prot_t = prot_tumor.loc[common]
    trans_t = trans_tumor.loc[common]

    mut = ds.get_somatic_mutation(source=mut_source)
    sys.stderr = stderr_bak

    msi_status = estimate_msi_status(ds, mut_source, cancer_name)
    # Restrict to samples in our data
    msi_status = msi_status.loc[msi_status.index.intersection(common)]

    target_genes = ['TP53', 'PIK3CA', 'KRAS', 'PTEN', 'ARID1A', 'SMAD4',
                    'FBXW7', 'BRAF', 'ERBB2']
    available = set(prot_t.columns) & set(trans_t.columns)
    target_genes = [g for g in target_genes if g in available]

    # ── 2a: MSI-stratified Layer 2 regression ──
    results = []
    for gene in target_genes:
        for msi_group in ['MSI-H', 'MSS']:
            idx = msi_status[msi_status == msi_group].index
            idx = idx.intersection(trans_t[gene].dropna().index)
            idx = idx.intersection(prot_t[gene].dropna().index)

            if len(idx) < 10:
                results.append({
                    'cancer': cancer_name, 'gene': gene, 'msi_group': msi_group,
                    'beta': np.nan, 'se': np.nan, 'p': np.nan,
                    'r2': np.nan, 'n': len(idx)
                })
                continue

            y = prot_t.loc[idx, gene]
            x = trans_t.loc[idx, gene]
            y_z = (y - y.mean()) / y.std()
            x_z = (x - x.mean()) / x.std()
            x_z = x_z.rename('mrna')

            X = sm.add_constant(x_z.to_frame())
            model = sm.OLS(y_z, X, missing='drop').fit()

            results.append({
                'cancer': cancer_name, 'gene': gene, 'msi_group': msi_group,
                'beta': model.params['mrna'],
                'se': model.bse['mrna'],
                'p': model.pvalues['mrna'],
                'r2': model.rsquared,
                'n': int(model.nobs)
            })

    msi_df = pd.DataFrame(results)

    # Print comparison
    print(f"\n  {'Gene':10s}  {'MSS β':>8s} {'(n)':>5s}  {'MSI-H β':>8s} {'(n)':>5s}  {'Δβ':>8s}")
    print(f"  {'-'*52}")
    for gene in target_genes:
        mss = msi_df[(msi_df['gene'] == gene) & (msi_df['msi_group'] == 'MSS')]
        msih = msi_df[(msi_df['gene'] == gene) & (msi_df['msi_group'] == 'MSI-H')]
        b_mss = mss['beta'].values[0] if len(mss) > 0 and not np.isnan(mss['beta'].values[0]) else np.nan
        b_msih = msih['beta'].values[0] if len(msih) > 0 and not np.isnan(msih['beta'].values[0]) else np.nan
        n_mss = mss['n'].values[0] if len(mss) > 0 else 0
        n_msih = msih['n'].values[0] if len(msih) > 0 else 0
        delta = b_msih - b_mss if not np.isnan(b_msih) and not np.isnan(b_mss) else np.nan

        b_mss_s = f"{b_mss:+8.3f}" if not np.isnan(b_mss) else "     N/A"
        b_msih_s = f"{b_msih:+8.3f}" if not np.isnan(b_msih) else "     N/A"
        delta_s = f"{delta:+8.3f}" if not np.isnan(delta) else "     N/A"
        print(f"  {gene:10s}  {b_mss_s} ({n_mss:3d})  {b_msih_s} ({n_msih:3d})  {delta_s}")

    msi_df.to_csv(f"{OUTPUT_DIR}/{cancer_name}_msi_layer2.csv", index=False)

    # ── 2b: Interaction test ──
    interaction_results = []
    for gene in target_genes:
        idx = msi_status.index.intersection(trans_t[gene].dropna().index)
        idx = idx.intersection(prot_t[gene].dropna().index)
        if len(idx) < 20:
            continue

        df_g = pd.DataFrame({
            'protein_z': (prot_t.loc[idx, gene] - prot_t.loc[idx, gene].mean()) / prot_t.loc[idx, gene].std(),
            'mrna_z': (trans_t.loc[idx, gene] - trans_t.loc[idx, gene].mean()) / trans_t.loc[idx, gene].std(),
            'msi': (msi_status.loc[idx] == 'MSI-H').astype(float)
        }).dropna()

        if len(df_g) < 20 or df_g['msi'].nunique() < 2:
            continue

        try:
            model = sm.OLS.from_formula('protein_z ~ mrna_z * msi', data=df_g).fit()
            interaction_results.append({
                'cancer': cancer_name,
                'gene': gene,
                'interaction_beta': model.params.get('mrna_z:msi', np.nan),
                'interaction_p': model.pvalues.get('mrna_z:msi', np.nan),
                'main_mrna_beta': model.params['mrna_z'],
                'main_msi_beta': model.params['msi'],
                'n': int(model.nobs)
            })
        except:
            pass

    int_df = pd.DataFrame(interaction_results)
    if len(int_df) > 0:
        print(f"\n  Interaction tests (mRNA × MSI):")
        print(f"  {'Gene':10s}  {'β_interact':>10s}  {'p':>10s}  {'n':>4s}")
        print(f"  {'-'*40}")
        for _, r in int_df.sort_values('interaction_p').iterrows():
            sig = "***" if r['interaction_p'] < 0.001 else "**" if r['interaction_p'] < 0.01 else "*" if r['interaction_p'] < 0.05 else ""
            print(f"  {r['gene']:10s}  {r['interaction_beta']:+10.3f}  {r['interaction_p']:10.3e}  {r['n']:4.0f}  {sig}")

    return msi_df, int_df, msi_status, prot_t, trans_t, mut


# ═══════════════════════════════════════════════════════════════════════════
# TASK B: PATHWAY / KINASE ACTIVITY SCORE
# ═══════════════════════════════════════════════════════════════════════════

# Substrate dictionaries (PhosphoSitePlus-based)
EGFR_SUBSTRATES = {
    'EGFR': ['Y1068', 'Y1173', 'Y1148', 'Y992', 'Y1045', 'S1064'],
    'PLCG1': ['Y783'],
    'SHC1': ['Y317', 'Y239'],
    'GAB1': ['Y627', 'Y659'],
    'STAT3': ['Y705'],
    'MAPK1': ['T185', 'Y187'],
    'MAPK3': ['T202', 'Y204'],
    'AKT1': ['S473', 'T308'],
}

FGFR2_SUBSTRATES = {
    'FGFR2': ['Y656', 'Y657'],
    'FRS2': ['Y196', 'Y306', 'Y349', 'Y392'],
    'PLCG1': ['Y783'],
    'MAPK1': ['T185', 'Y187'],
    'MAPK3': ['T202', 'Y204'],
}

ERBB2_SUBSTRATES = {
    'ERBB2': ['Y1248', 'Y1221', 'Y1222', 'Y877', 'S1083', 'S1078'],
    'ERBB3': ['Y1289', 'Y1197'],
    'AKT1': ['S473', 'T308'],
    'MAPK1': ['T185', 'Y187'],
}


def compute_kinase_activity(phospho_df, substrate_dict, cancer_name):
    """Compute kinase activity score from averaged substrate z-scores."""
    matched_vals = []
    matched_names = []
    total_targets = sum(len(v) for v in substrate_dict.values())

    # Build gene-site index
    genes = phospho_df.columns.get_level_values(0)
    sites = phospho_df.columns.get_level_values(1)

    for protein, site_list in substrate_dict.items():
        for site in site_list:
            site_num = ''.join(filter(str.isdigit, site))
            site_aa = site[0].upper()

            # Search for matching column
            found = False
            for i, (g, s) in enumerate(zip(genes, sites)):
                if protein in str(g) and site_num in str(s) and site_aa in str(s).upper():
                    col = phospho_df.columns[i]
                    vals = phospho_df[col].dropna()
                    if len(vals) >= 10:
                        z = (vals - vals.mean()) / vals.std()
                        matched_vals.append(z)
                        matched_names.append(f"{protein}_{site}")
                        found = True
                        break

    print(f"  {cancer_name}: {len(matched_vals)}/{total_targets} substrates matched")
    for n in matched_names:
        print(f"    ✓ {n}")

    if len(matched_vals) >= 2:
        activity = pd.concat(matched_vals, axis=1).mean(axis=1)
        return activity, matched_names
    return None, matched_names


def run_pathway_transmission(cancer_name, CancerClass, kinase_gene, substrate_dict, kinase_name):
    """Compare site-level vs pathway-level transmission."""
    print(f"\n  ── {kinase_name} pathway: {cancer_name} ──")

    stderr_bak = sys.stderr
    sys.stderr = io.StringIO()
    ds = CancerClass()

    prot = ds.get_proteomics(source="umich")
    prot_tumor = prot[~prot.index.str.endswith('.N')].copy()
    prot_tumor.columns = prot_tumor.columns.get_level_values(0)
    prot_tumor = prot_tumor.T.groupby(level=0).mean().T

    trans = ds.get_transcriptomics(source="bcm")
    trans_tumor = trans[~trans.index.str.endswith('.N')].copy()
    trans_tumor.columns = trans_tumor.columns.get_level_values(0)
    trans_tumor = trans_tumor.T.groupby(level=0).mean().T

    phospho = ds.get_phosphoproteomics(source="umich")
    phospho_tumor = phospho[~phospho.index.str.endswith('.N')].copy()
    sys.stderr = stderr_bak

    common = prot_tumor.index.intersection(trans_tumor.index)
    prot_t = prot_tumor.loc[common]
    trans_t = trans_tumor.loc[common]

    if kinase_gene not in trans_t.columns:
        print(f"  {kinase_gene} not in transcriptomics")
        return None

    # A. Site-level (from Phase 1 L3 results)
    try:
        l3 = pd.read_csv(f"{OUTPUT_DIR}/{cancer_name}_regression_layer3.csv")
        gene_l3 = l3[l3['gene'] == kinase_gene]
        if len(gene_l3) > 0:
            site_median_beta = gene_l3['beta'].median()
            site_median_r2 = gene_l3['partial_R2'].median()
            n_sites = len(gene_l3)
        else:
            site_median_beta = np.nan
            site_median_r2 = np.nan
            n_sites = 0
    except FileNotFoundError:
        site_median_beta = np.nan
        site_median_r2 = np.nan
        n_sites = 0

    # B. Pathway-level
    # Aggregate phospho to gene-site level
    phospho_gs = phospho_tumor.copy()
    phospho_gs.columns = pd.MultiIndex.from_arrays([
        phospho_tumor.columns.get_level_values(0),
        phospho_tumor.columns.get_level_values(1)
    ], names=['gene', 'site'])

    activity, matched = compute_kinase_activity(phospho_gs, substrate_dict, cancer_name)

    pathway_beta = np.nan
    pathway_r2 = np.nan
    pathway_n = 0

    if activity is not None:
        # mRNA of kinase gene → pathway activity
        mrna = trans_t[kinase_gene]
        cidx = activity.dropna().index.intersection(mrna.dropna().index)
        if len(cidx) >= 10:
            y = activity.loc[cidx]
            y_z = (y - y.mean()) / y.std()
            x = mrna.loc[cidx]
            x_z = ((x - x.mean()) / x.std()).rename('mrna')
            X = sm.add_constant(x_z.to_frame())
            try:
                model = sm.OLS(y_z, X, missing='drop').fit()
                pathway_beta = model.params['mrna']
                pathway_r2 = model.rsquared
                pathway_n = int(model.nobs)
            except:
                pass

        # Also: total protein → pathway activity
        if kinase_gene in prot_t.columns:
            pidx = activity.dropna().index.intersection(prot_t[kinase_gene].dropna().index)
            if len(pidx) >= 10:
                y_p = activity.loc[pidx]
                y_pz = (y_p - y_p.mean()) / y_p.std()
                x_p = prot_t.loc[pidx, kinase_gene]
                x_pz = ((x_p - x_p.mean()) / x_p.std()).rename('protein')
                X_p = sm.add_constant(x_pz.to_frame())
                try:
                    model_p = sm.OLS(y_pz, X_p, missing='drop').fit()
                    prot_pathway_beta = model_p.params['protein']
                    prot_pathway_r2 = model_p.rsquared
                except:
                    prot_pathway_beta = np.nan
                    prot_pathway_r2 = np.nan
            else:
                prot_pathway_beta = np.nan
                prot_pathway_r2 = np.nan
        else:
            prot_pathway_beta = np.nan
            prot_pathway_r2 = np.nan

    else:
        prot_pathway_beta = np.nan
        prot_pathway_r2 = np.nan

    result = {
        'kinase': kinase_name,
        'kinase_gene': kinase_gene,
        'cancer': cancer_name,
        'n_sites_total': n_sites,
        'site_median_beta': site_median_beta,
        'site_median_r2': site_median_r2,
        'n_substrates_matched': len(matched),
        'pathway_beta_mrna': pathway_beta,
        'pathway_r2_mrna': pathway_r2,
        'pathway_beta_prot': prot_pathway_beta,
        'pathway_r2_prot': prot_pathway_r2,
        'pathway_n': pathway_n,
    }

    print(f"  Site-level:    median β={site_median_beta:+.3f}, median R²={site_median_r2:.3f} ({n_sites} sites)" if not np.isnan(site_median_beta) else "  Site-level:    N/A")
    print(f"  Pathway(mRNA): β={pathway_beta:+.3f}, R²={pathway_r2:.3f} (n={pathway_n})" if not np.isnan(pathway_beta) else "  Pathway(mRNA): N/A")
    print(f"  Pathway(Prot): β={prot_pathway_beta:+.3f}, R²={prot_pathway_r2:.3f}" if not np.isnan(prot_pathway_beta) else "  Pathway(Prot): N/A")

    ratio = pathway_r2 / site_median_r2 if not np.isnan(pathway_r2) and not np.isnan(site_median_r2) and site_median_r2 > 0 else np.nan
    if not np.isnan(ratio):
        print(f"  Pathway/Site R² ratio: {ratio:.2f}x")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── TASK A ──
    print("=" * 70)
    print("  TASK A: MSI STRATIFICATION")
    print("=" * 70)

    all_msi = []
    all_interact = []

    for cancer_name, CancerClass, mut_src in [
        ('UCEC', cptac.Ucec, 'harmonized'),
        ('COAD', cptac.Coad, 'washu'),
    ]:
        msi_df, int_df, msi_status, prot_t, trans_t, mut = \
            run_msi_stratification(cancer_name, CancerClass, mut_src)
        all_msi.append(msi_df)
        if len(int_df) > 0:
            all_interact.append(int_df)

    # Save combined interaction results
    if all_interact:
        interact_all = pd.concat(all_interact, ignore_index=True)
        interact_all.to_csv(f"{OUTPUT_DIR}/msi_interaction_tests.csv", index=False)

    # ── TASK B ──
    print(f"\n\n{'='*70}")
    print("  TASK B: PATHWAY / KINASE ACTIVITY")
    print("=" * 70)

    pathway_results = []

    configs = [
        ('LUAD', cptac.Luad, 'EGFR', EGFR_SUBSTRATES, 'EGFR'),
        ('UCEC', cptac.Ucec, 'FGFR2', FGFR2_SUBSTRATES, 'FGFR2'),
        ('BRCA', cptac.Brca, 'ERBB2', ERBB2_SUBSTRATES, 'ERBB2'),
        ('COAD', cptac.Coad, 'EGFR', EGFR_SUBSTRATES, 'EGFR'),  # control (low transmission)
    ]

    for cancer_name, CancerClass, kinase_gene, sub_dict, kinase_name in configs:
        res = run_pathway_transmission(cancer_name, CancerClass, kinase_gene, sub_dict, kinase_name)
        if res:
            pathway_results.append(res)

    pw_df = pd.DataFrame(pathway_results)
    pw_df.to_csv(f"{OUTPUT_DIR}/pathway_transmission.csv", index=False)

    # Summary table
    print(f"\n  ── Pathway vs Site Transmission Summary ──")
    print(f"  {'Kinase':8s}  {'Cancer':6s}  {'Site R²':>8s}  {'Path R²':>8s}  {'Ratio':>6s}  {'N_sub':>5s}")
    print(f"  {'-'*50}")
    for _, r in pw_df.iterrows():
        sr = f"{r['site_median_r2']:.3f}" if not np.isnan(r['site_median_r2']) else "   N/A"
        pr = f"{r['pathway_r2_prot']:.3f}" if not np.isnan(r['pathway_r2_prot']) else "   N/A"
        ratio = r['pathway_r2_prot'] / r['site_median_r2'] if not np.isnan(r['pathway_r2_prot']) and not np.isnan(r['site_median_r2']) and r['site_median_r2'] > 0 else np.nan
        ratio_s = f"{ratio:6.2f}" if not np.isnan(ratio) else "   N/A"
        print(f"  {r['kinase']:8s}  {r['cancer']:6s}  {sr:>8s}  {pr:>8s}  {ratio_s}  {r['n_substrates_matched']:5.0f}")

    print(f"\n{'='*70}")
    print("  ALL TASKS COMPLETE")
    print(f"{'='*70}")
