#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Measurement-quality sensitivity analysis
#
# Manuscript: Pan-cancer quantification of driver alteration transmission across
#             molecular layers reveals limited propagation to protein abundance (IJC-26-1558)
# Author:     Hisashi Nakano, PhD - Department of Health Data Science,
#             Niigata University of Health and Welfare, Niigata, Japan
# ORCID:      0000-0002-9023-880X

"""
Measurement-quality sensitivity analysis: tests whether low TS_R2 is confounded by
measurement quality (gene-dependent noise) rather than reflecting true biology.

Per-gene quality metrics (from cached cptac data; cache read-only):
  protein_detection : fraction of tumor samples with non-missing protein (get_proteomics umich)
  NumberPSM         : peptide-spectrum-match count per protein (umich raw report) = spectral/peptide depth
  ref_intensity     : umich ReferenceIntensity (abundance reference)
  protein_median/sd : abundance & variability from the proteomics matrix
  mrna_detection / mrna_median / mrna_sd : transcriptomics (bcm)
Correlate vs published TS_R2 / R2_L1 / R2_L2; conditioning check on well-measured pairs.
"""
import os, sys, json, warnings, re
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cptac
ROOT=os.environ.get("CPTAC_DATA_DIR", "."); OUT=os.environ.get("CPTAC_OUT_DIR", ".")
CACHE=os.path.join(os.path.dirname(cptac.__file__), "data")

CLS={'LUAD':cptac.Luad,'COAD':cptac.Coad,'UCEC':cptac.Ucec,'CCRCC':cptac.Ccrcc,
     'OV':cptac.Ov,'BRCA':cptac.Brca,'PDAC':cptac.Pdac}
UMICH={'LUAD':'umich-luad','COAD':'umich-coad','UCEC':'umich-ucec','CCRCC':'umich-ccrcc',
       'OV':'umich-ov','BRCA':'umich-brca','PDAC':'umich-pdac'}

def gene_level(df):
    d=df[~df.index.str.endswith('.N')].copy()
    d.columns=d.columns.get_level_values(0)
    return d.T.groupby(level=0).mean().T

def umich_qc(cohort):
    """Parse NumberPSM / MaxPepProb / ReferenceIntensity per gene symbol from raw umich report."""
    f=f"{CACHE}/{UMICH[cohort]}/Report_abundance_groupby=protein_protNorm=MD_gu=2.tsv.gz"
    meta=pd.read_csv(f,sep='\t',usecols=['Index','NumberPSM','MaxPepProb','ReferenceIntensity'])
    meta['symbol']=meta['Index'].str.split('|').str[-2]
    g=meta.groupby('symbol').agg(NumberPSM=('NumberPSM','sum'),
        MaxPepProb=('MaxPepProb','max'), ref_intensity=('ReferenceIntensity','max'))
    return g

def main():
    pub=pd.read_csv(f"{ROOT}/pancancer_ts_continuous.csv")
    pub=pub[['cancer','gene','R2_L1','R2_L2','TS_R2_2L']].rename(columns={'TS_R2_2L':'TS_R2'})
    pub=pub.dropna(subset=['TS_R2'])
    # sanity gate: TS == R2_L1*R2_L2
    pub['ts_check']=(pub.R2_L1*pub.R2_L2 - pub.TS_R2).abs()
    print("SANITY: max |TS_R2 - R2_L1*R2_L2| =", f"{pub.ts_check.max():.2e}", "| n pairs =", len(pub))
    print("EGFR/LUAD TS =", float(pub[(pub.cancer=='LUAD')&(pub.gene=='EGFR')].TS_R2))

    rows=[]
    for cohort, cls in CLS.items():
        ds=cls()
        prot=gene_level(ds.get_proteomics(source='umich'))
        trans=gene_level(ds.get_transcriptomics(source='bcm'))
        qc=umich_qc(cohort)
        npat=prot.shape[0]
        for gene in pub[pub.cancer==cohort].gene.unique():
            r={'cancer':cohort,'gene':gene}
            if gene in prot.columns:
                pv=prot[gene]
                r['protein_detection']=float(pv.notna().mean())
                r['protein_median']=float(pv.median())
                r['protein_sd']=float(pv.std())
            if gene in trans.columns:
                tv=trans[gene]
                r['mrna_detection']=float(tv.notna().mean())
                r['mrna_median']=float(tv.median())
                r['mrna_sd']=float(tv.std())
            if gene in qc.index:
                r['NumberPSM']=int(qc.loc[gene,'NumberPSM'])
                r['ref_intensity']=float(qc.loc[gene,'ref_intensity'])
            rows.append(r)
    q=pd.DataFrame(rows)
    df=pub.merge(q,on=['cancer','gene'],how='left')
    df['log10_PSM']=np.log10(df['NumberPSM'])
    df.to_csv(f"{OUT}/r1_2_quality_table.csv",index=False)

    METRICS=['protein_detection','NumberPSM','log10_PSM','ref_intensity','protein_median',
             'protein_sd','mrna_detection','mrna_median','mrna_sd']
    def corr(y,xs):
        out={}
        for x in xs:
            d=df[[y,x]].dropna()
            if len(d)>=10:
                rho,p=stats.spearmanr(d[y],d[x]); out[x]={'rho':round(rho,3),'p':round(p,4),'n':len(d)}
        return out
    correlations={'TS_R2':corr('TS_R2',METRICS),
                  'R2_L1':corr('R2_L1',['mrna_detection','mrna_median','mrna_sd','NumberPSM','log10_PSM']),
                  'R2_L2':corr('R2_L2',['protein_detection','NumberPSM','log10_PSM','ref_intensity','protein_median','protein_sd'])}

    # ---- well-measured vs poorly-measured + conditioning ----
    LOW_TS=0.05
    res={'n_pairs':int(len(df)),'frac_low_TS_overall':float((df.TS_R2<LOW_TS).mean())}
    # protein detection split
    for metric,thr,lab in [('protein_detection',0.90,'detect>90%'),
                           ('NumberPSM',df.NumberPSM.median(),'PSM>median'),
                           ('ref_intensity',df.ref_intensity.median(),'refInt>median')]:
        well=df[df[metric]>thr]; poor=df[df[metric]<=thr]
        res[lab]={'thr':float(thr),'n_well':int(len(well)),'n_poor':int(len(poor)),
            'median_TS_well':float(well.TS_R2.median()),'median_TS_poor':float(poor.TS_R2.median()),
            'frac_low_TS_well':float((well.TS_R2<LOW_TS).mean()),
            'frac_low_TS_poor':float((poor.TS_R2<LOW_TS).mean()),
            'mannwhitney_p':float(stats.mannwhitneyu(well.TS_R2,poor.TS_R2,alternative='two-sided')[1])}
    # decisive conditioning: very well-measured pairs (high detection AND high PSM)
    wm=df[(df.protein_detection>0.90)&(df.NumberPSM>df.NumberPSM.median())]
    res['decisive_wellmeasured']={'n':int(len(wm)),
        'frac_TS_below_0.05':float((wm.TS_R2<0.05).mean()),
        'n_TS_above_0.05':int((wm.TS_R2>=0.05).sum()),
        'examples_high_PSM_low_TS': wm.sort_values('NumberPSM',ascending=False)
            [['cancer','gene','NumberPSM','protein_detection','TS_R2']].head(8).round(4).values.tolist()}

    # ---- named examples ----
    named=['EGFR','ERBB2','CTNNB1','GAPDH','ACTB','SOX9','TCF7L2','FOXA1','GATA3','CBFB','TP53']
    ex=df[df.gene.isin(named)][['cancer','gene','NumberPSM','protein_detection','ref_intensity','R2_L1','R2_L2','TS_R2']].round(4)
    ex.to_csv(f"{OUT}/r1_2_named_examples.csv",index=False)

    out={'correlations':correlations,'splits_conditioning':res}
    with open(f"{OUT}/r1_2_results.json",'w') as f: json.dump(out,f,indent=2,default=str)
    print(json.dumps(out,indent=2,default=str))
    print("\n=== named examples ==="); print(ex.sort_values('NumberPSM',ascending=False).to_string(index=False))
    print("\nWROTE r1_2_quality_table.csv, r1_2_named_examples.csv, r1_2_results.json")

if __name__=="__main__":
    main()
