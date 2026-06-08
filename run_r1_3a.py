#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Mutation-type stratified Layer-1 analysis
#
# Manuscript: Pan-cancer quantification of driver alteration transmission across
#             molecular layers reveals limited propagation to protein abundance (IJC-26-1558)
# Author:     Hisashi Nakano, PhD - Department of Health Data Science,
#             Niigata University of Health and Welfare, Niigata, Japan
# ORCID:      0000-0002-9023-880X

"""
Mutation-type stratified Layer-1 analysis.

Rationale: pooling all non-silent mutations (hotspot gain-of-function + truncating
loss-of-function) can cancel or dilute opposite-direction effects, which would lower the
apparent transmission. This analysis refits the covariate-adjusted Layer-1 model
(mRNA ~ mutation + age + sex + stage) with the mutation term restricted to one variant
class vs clean wild-type:
  pooled      : any non-silent vs WT            (= published)
  missense    : missense-only vs WT
  truncating  : truncating vs WT
  hotspot     : recurrent-codon missense vs WT  (oncogenes only)
WT = patients with NO non-silent mutation in the gene (same reference as pooled).
MIN_N = 10 mutated samples per stratum (strata below this are NOT formed).
Layer 2 is held at the published value (this analysis targets the mutation->mRNA step).
OLS engine = the producer's regression_layer1 (validated exact against the CNA-adjusted
analysis). Cache read-only; outputs here.
"""
import sys, os, re, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
ROOT = os.environ.get("CPTAC_DATA_DIR", "."); OUT = os.environ.get("CPTAC_OUT_DIR", ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_regression_ts as P

CANCERS = ['LUAD','COAD','UCEC','CCRCC','OV','BRCA','PDAC']
MIN_N = 10                      # minimum mutated samples to form a stratum
TRUNC = ['Nonsense_Mutation','Frame_Shift_Del','Frame_Shift_Ins','Splice_Site','Splice_Region']
GOF = {'KRAS','NRAS','HRAS','BRAF','PIK3CA','EGFR','ERBB2','FGFR2','FGFR3','IDH1','IDH2','AKT1','MTOR','CTNNB1','MET'}
LOF = {'TP53','PTEN','ARID1A','VHL','CDKN2A','STK11','RB1','SMAD4','NF1','BAP1','PBRM1','SETD2','FBXW7','KEAP1','ATM','KMT2C','KMT2D','APC','CHD4','RBM10'}

def strat_sets(mut, gene, all_patients):
    ns = mut[~mut['Mutation'].isin(['Silent','Intron',"3'UTR","5'UTR","3'Flank","5'Flank",'IGR','RNA'])]
    gm = ns[ns['Gene'] == gene]
    mutated = set(gm.index.unique()) & all_patients
    missense = set(gm[gm['Mutation']=='Missense_Mutation'].index.unique())
    truncating = set(gm[gm['Mutation'].isin(TRUNC)].index.unique()) & all_patients
    missense_only = (missense - truncating) & all_patients
    wt = all_patients - mutated
    # hotspot = missense at recurrently-mutated codons (>=3 samples at same protein position)
    hotspot = set()
    if gene in GOF:
        mm = gm[gm['Mutation']=='Missense_Mutation'].copy()
        pos = mm['Location'].astype(str).str.extract(r'p\.[A-Z*](\d+)')[0]
        mm = mm.assign(codon=pos)
        cc = mm.dropna(subset=['codon']).groupby('codon').apply(lambda d: d.index.nunique())
        hot_codons = set(cc[cc >= 3].index)
        if hot_codons:
            hotspot = set(mm[mm['codon'].isin(hot_codons)].index.unique()) & missense_only
    return dict(mutated=mutated, wt=wt, missense=missense_only, truncating=truncating, hotspot=hotspot)

def fit_vs_wt(trans, gene, mut_set, wt, covars_l1):
    """regression_layer1 for `mut_set` (=1) vs `wt` (=0), covariate-adjusted."""
    if len(mut_set) < MIN_N or len(wt) < MIN_N:
        return None
    combined = list(mut_set) + list(wt)
    mrna = trans.loc[trans.index.intersection(pd.Index(combined)), gene].dropna()
    ms = pd.Series(0, index=mrna.index)
    for p in mut_set:
        if p in ms.index: ms[p] = 1
    if int(ms.sum()) < MIN_N:
        return None
    return P.regression_layer1(mrna, ms, covars_l1)

def main():
    pub = pd.read_csv(f"{ROOT}/pancancer_ts_continuous.csv")[['cancer','gene','beta_L1','R2_L1','R2_L2','TS_R2_2L']]
    long_rows, pair_rows = [], []
    for cancer in CANCERS:
        print(f"\n=== {cancer} ===")
        prot, trans, mut, phos, clin, pur, ds = P.load_cancer_data(cancer)
        available = set(prot.columns) & set(trans.columns)
        drivers = [g for g in P.CANCER_DRIVERS[cancer] if g in available]
        covars, cov_names = P.get_covariates(clin, pur, prot.index)
        covars_l1 = covars[[c for c in cov_names if c != 'tumor_purity']] if cov_names else None
        all_patients = set(trans.index)
        pubc = pub[pub.cancer==cancer].set_index('gene')
        for gene in drivers:
            S = strat_sets(mut, gene, all_patients)
            pr = {'cancer':cancer,'gene':gene,'n_wt':len(S['wt']),
                  'n_missense':len(S['missense']),'n_truncating':len(S['truncating']),'n_hotspot':len(S['hotspot'])}
            R2_L2 = float(pubc.loc[gene,'R2_L2']) if gene in pubc.index else np.nan
            pr['R2_L2_pub'] = R2_L2
            pr['pub_R2_L1'] = float(pubc.loc[gene,'R2_L1']) if gene in pubc.index else np.nan
            pr['pub_TS_R2'] = float(pubc.loc[gene,'TS_R2_2L']) if gene in pubc.index else np.nan

            strata = {}
            # pooled (= published) only when there is a real WT reference
            if len(S['wt']) >= 5:
                strata['pooled'] = fit_vs_wt(trans, gene, S['mutated'], S['wt'], covars_l1)
                pr['near_saturated'] = False
            else:
                pr['near_saturated'] = True   # WT<5 -> published used trunc-vs-missense fallback; WT-strata N/A
            for name in ['missense','truncating','hotspot']:
                strata[name] = fit_vs_wt(trans, gene, S[name], S['wt'], covars_l1)

            for name, res in strata.items():
                if res is None: continue
                long_rows.append({'cancer':cancer,'gene':gene,'stratum':name,
                    'n':res['n'],'n_mut':res['n_mut'],'beta_L1':res['beta'],
                    'R2_L1':res['partial_R2'],'p_mut':res['p_value'],
                    'R2_L2_pub':R2_L2,'TS_R2':res['partial_R2']*R2_L2 if not np.isnan(R2_L2) else np.nan})
                pr[f'{name}_beta']=res['beta']; pr[f'{name}_R2']=res['partial_R2']
                pr[f'{name}_p']=res['p_value']; pr[f'{name}_nmut']=res['n_mut']
                pr[f'{name}_TS']=res['partial_R2']*R2_L2 if not np.isnan(R2_L2) else np.nan
            pair_rows.append(pr)

    long = pd.DataFrame(long_rows); pairs = pd.DataFrame(pair_rows)
    long.to_csv(f"{OUT}/r1_3a_stratified_L1_long.csv", index=False)

    # ---- dilution & sign-cancellation flags ----
    def best_stratum_R2(r):
        vals = [r.get(f'{s}_R2',np.nan) for s in ['missense','truncating','hotspot']]
        vals = [v for v in vals if pd.notna(v)]
        return max(vals) if vals else np.nan
    pairs['best_stratum_R2'] = pairs.apply(best_stratum_R2, axis=1)
    base = pairs['pooled_R2'].where(pairs['pooled_R2'].notna(), pairs['pub_R2_L1'])
    pairs['pooled_ref_R2'] = base
    pairs['dilution_gain'] = pairs['best_stratum_R2'] - pairs['pooled_ref_R2']
    # dilution: a single class materially beats pooled (>=1.5x AND +0.03 absolute)
    pairs['pooling_diluted'] = (pairs['best_stratum_R2'] >= 1.5*pairs['pooled_ref_R2']) & \
                               (pairs['dilution_gain'] >= 0.03)
    # sign cancellation: missense & truncating betas opposite sign, both nominally sig
    def sign_cancel(r):
        mb,tb,mp,tp = r.get('missense_beta'),r.get('truncating_beta'),r.get('missense_p'),r.get('truncating_p')
        if any(pd.isna(x) for x in [mb,tb,mp,tp]): return False
        return (np.sign(mb)!=np.sign(tb)) and (mp<0.1) and (tp<0.1)
    pairs['sign_cancellation'] = pairs.apply(sign_cancel, axis=1)
    pairs['TS_best'] = pairs['best_stratum_R2']*pairs['R2_L2_pub']
    pairs.to_csv(f"{OUT}/r1_3a_pair_summary.csv", index=False)

    summ = {
        'n_pairs': int(len(pairs)),
        'n_with_missense_stratum': int(pairs['missense_R2'].notna().sum()) if 'missense_R2' in pairs else 0,
        'n_with_truncating_stratum': int(pairs['truncating_R2'].notna().sum()) if 'truncating_R2' in pairs else 0,
        'n_with_hotspot_stratum': int(pairs['hotspot_R2'].notna().sum()) if 'hotspot_R2' in pairs else 0,
        'n_pooling_diluted': int(pairs['pooling_diluted'].sum()),
        'diluted_pairs': pairs[pairs['pooling_diluted']][['cancer','gene']].apply(tuple,axis=1).tolist(),
        'n_sign_cancellation': int(pairs['sign_cancellation'].sum()),
        'sign_cancel_pairs': pairs[pairs['sign_cancellation']][['cancer','gene']].apply(tuple,axis=1).tolist(),
        # post-stratification: best single variant-class TS (TS_best = best_stratum_R2 * published R2_L2)
        'top_pairs_stratified_TS_best_gt_0.05': [
            (c, g, round(float(t), 4))
            for c, g, t in pairs.loc[pairs['TS_best'] > 0.05, ['cancer', 'gene', 'TS_best']].itertuples(index=False)
        ],
    }
    with open(f"{OUT}/r1_3a_summary.json",'w') as f: json.dump(summ,f,indent=2,default=str)
    print("\n"+json.dumps(summ,indent=2,default=str))
    print("\nWROTE r1_3a_stratified_L1_long.csv, r1_3a_pair_summary.csv, r1_3a_summary.json")

if __name__ == "__main__":
    main()
