#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MSI/MMR sensitivity analysis
#
# Manuscript: Pan-cancer quantification of driver alteration transmission across
#             molecular layers reveals limited propagation to protein abundance (IJC-26-1558)
# Author:     Hisashi Nakano, PhD - Department of Health Data Science,
#             Niigata University of Health and Welfare, Niigata, Japan
# ORCID:      0000-0002-9023-880X

"""
MSI/MMR sensitivity analysis: strengthens the MSI definition beyond TMB>10 / MLH1 alone.
Restricted to COAD + UCEC (the MSI-relevant cohorts, by data availability).
  (1) reproduce the published ERBB2 MSI x mRNA interaction (TMB>10)  -> sanity gate
  (2) re-test it under an INDEL-burden MSI definition (MSI-specific, better than total TMB)
  (3) MMR-beyond-MLH1: protein loss of MLH1/MSH2/MSH6/PMS2 and reclassification (UCEC; full MMR)
Cache read-only; outputs in revision/r1_4/. Layer-2 model identical to run_msi_pathway.py.
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cptac
OUT = os.environ.get("CPTAC_OUT_DIR", ".")

def gene_level(df):
    d = df[~df.index.str.endswith('.N')].copy()
    d.columns = d.columns.get_level_values(0)
    return d.T.groupby(level=0).mean().T

NONSILENT_EXCL = ['Silent','Intron',"3'UTR","5'UTR","3'Flank","5'Flank",'IGR','RNA']

def load(cn, cls, msrc):
    ds = cls()
    prot = gene_level(ds.get_proteomics(source='umich'))
    trans = gene_level(ds.get_transcriptomics(source='bcm'))
    common = prot.index.intersection(trans.index)
    prot, trans = prot.loc[common], trans.loc[common]
    mut = ds.get_somatic_mutation(source=msrc)
    tmb = (mut.groupby(mut.index).size() / 30)                       # producer's TMB proxy (all rows)
    ns = mut[~mut['Mutation'].isin(NONSILENT_EXCL)]
    ind = ns[ns['Variant_Type'].isin(['INS','DEL'])]
    indel = ind.groupby(ind.index).size().reindex(tmb.index).fillna(0)
    return prot, trans, tmb, indel, common

def interaction(prot, trans, gene, msi_series, common):
    idx = msi_series.index.intersection(common).intersection(trans[gene].dropna().index).intersection(prot[gene].dropna().index)
    if len(idx) < 20: return None
    df = pd.DataFrame({
        'protein_z': (prot.loc[idx,gene]-prot.loc[idx,gene].mean())/prot.loc[idx,gene].std(),
        'mrna_z': (trans.loc[idx,gene]-trans.loc[idx,gene].mean())/trans.loc[idx,gene].std(),
        'msi': (msi_series.loc[idx]=='MSI-H').astype(float)}).dropna()
    if len(df) < 20 or df['msi'].nunique() < 2: return None
    m = sm.OLS.from_formula('protein_z ~ mrna_z * msi', data=df).fit()
    return {'interaction_beta': m.params.get('mrna_z:msi',np.nan),
            'interaction_p': m.pvalues.get('mrna_z:msi',np.nan),
            'n': int(m.nobs), 'n_msih': int(df['msi'].sum())}

results = {'cohorts':{}, 'erbb2_robustness':[], 'mmr':{}}
TARGETS = ['TP53','PIK3CA','KRAS','PTEN','ARID1A','SMAD4','FBXW7','BRAF','ERBB2']

for cn, cls, msrc in [('COAD',cptac.Coad,'washu'), ('UCEC',cptac.Ucec,'harmonized')]:
    prot, trans, tmb, indel, common = load(cn, cls, msrc)
    msi_tmb = pd.Series('MSS', index=tmb.index); msi_tmb[tmb>10] = 'MSI-H'
    # indel-based MSI definitions at several cuts
    cohort = {'n_common':len(common), 'n_tmb_msih':int((tmb>10).sum()),
              'indel_median':float(indel.median()), 'indel_max':float(indel.max())}
    # concordance TMB>10 vs indel>=cut
    conc = {}
    for cut in [8,10,15]:
        a=set(tmb[tmb>10].index); b=set(indel[indel>=cut].index)
        conc[f'indel>={cut}'] = {'n_indel_msih':len(b),'both':len(a&b),'only_tmb':len(a-b),'only_indel':len(b-a)}
    cohort['concordance']=conc
    results['cohorts'][cn]=cohort

    # ERBB2 interaction: TMB (sanity) vs indel-based, across cuts
    base = interaction(prot, trans, 'ERBB2', msi_tmb, common)
    row = {'cancer':cn,'gene':'ERBB2','def_TMB>10_p':base['interaction_p'] if base else None,
           'def_TMB>10_beta':base['interaction_beta'] if base else None,'n':base['n'] if base else None,
           'n_msih_tmb':base['n_msih'] if base else None}
    for cut in [8,10,15]:
        msi_ind = pd.Series('MSS', index=indel.index); msi_ind[indel>=cut]='MSI-H'
        r = interaction(prot, trans, 'ERBB2', msi_ind, common)
        row[f'def_indel>={cut}_p'] = r['interaction_p'] if r else None
        row[f'def_indel>={cut}_nmsih'] = r['n_msih'] if r else None
    results['erbb2_robustness'].append(row)

# ---- MMR beyond MLH1 (UCEC: all 4 MMR proteins 100% detected) ----
dsu = cptac.Ucec()
protu = gene_level(dsu.get_proteomics(source='umich'))
mutu = dsu.get_somatic_mutation(source='harmonized')
tmbu = (mutu.groupby(mutu.index).size()/30)
nsu = mutu[~mutu['Mutation'].isin(NONSILENT_EXCL)]
indu = nsu[nsu['Variant_Type'].isin(['INS','DEL'])]
indelu = indu.groupby(indu.index).size().reindex(protu.index).fillna(0)
MMR=['MLH1','MSH2','MSH6','PMS2']
z = {g:(protu[g]-protu[g].mean())/protu[g].std() for g in MMR if g in protu.columns}
zdf = pd.DataFrame(z)
LOW = -1.0                       # protein-loss call: z < -1 (~bottom 16%)
low = {g:set(zdf.index[zdf[g] < LOW]) for g in zdf.columns}
mlh1_low = low.get('MLH1', set())
any_low = set().union(*low.values()) if low else set()
beyond_mlh1 = any_low - mlh1_low      # MMR-low by MSH2/MSH6/PMS2 but NOT MLH1
results['mmr']['UCEC'] = {
    'threshold_z': LOW,
    'n_samples': int(len(zdf)),
    'n_low_per_gene': {g:len(s) for g,s in low.items()},
    'n_any_mmr_low': len(any_low),
    'n_mlh1_low': len(mlh1_low),
    'n_low_beyond_mlh1': len(beyond_mlh1),   # extra samples MLH1 alone misses
    'beyond_mlh1_indel_high': int(sum(indelu.get(s,0) >= 10 for s in beyond_mlh1)),
    'any_mmr_low_indel_high': int(sum(indelu.get(s,0) >= 10 for s in any_low)),
    'mlh1_low_indel_high': int(sum(indelu.get(s,0) >= 10 for s in mlh1_low)),
}

with open(f'{OUT}/r1_4_results.json','w') as f: json.dump(results,f,indent=2,default=str)
print(json.dumps(results,indent=2,default=str))
print("\nWROTE r1_4_results.json")
