#!/usr/bin/env python3
"""Create auditable raw and seed-train-only O13G structured Fifth features."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from graphgps.lrx_add.fifth_semantic_features import semantic_features

AA = {'Val':'V','Leu':'L','Ile':'I','Asp':'D','Asn':'N','Glu':'E','Gln':'Q','Phe':'F','Tyr':'Y','His':'H','Arg':'R','Met':'M','Phg':'OTHER'}
TERMS = ('ester','free_carboxylic_acid','peptide_or_DOPE_related','other','unknown')
def canon(x):
    if pd.isna(x) or str(x) in {'nan','[Fr]'}: return '[Fr]'
    m=Chem.MolFromSmiles(str(x)); return Chem.MolToSmiles(m, canonical=True) if m else '[Fr]'
def raw(source):
    rows=[]
    for row in source[['Fifth','Fifth_SMILE']].drop_duplicates().itertuples(index=False):
        s=semantic_features(row.Fifth_SMILE); name='' if pd.isna(row.Fifth) else str(row.Fifth)
        match=re.match(r'^([A-Za-z]+)-(?:UC|OAm)(\d+)$',name)
        aa=AA.get(match.group(1),'UNKNOWN') if match else ('OTHER' if s.family_type!='UC_series' else s.uc_amino_acid_type)
        tail=int(match.group(2)) if match else int(s.uc_tail_carbon_count)
        if s.family_type=='UC_series': term='ester' if s.uc_terminal_ester else ('free_carboxylic_acid' if s.uc_terminal_carboxyl else 'unknown')
        elif s.family_type=='DOPE_SS_peptide_series': term='peptide_or_DOPE_related'
        else: term='other'
        rows.append({'canonical_smiles':canon(row.Fifth_SMILE),'Fifth name':name,'Fifth_SMILE':row.Fifth_SMILE,'parsed_AA':aa,'tail_length':tail if tail else np.nan,'tail_present':int(tail>0),'terminal_state':term,'family_type':s.family_type})
    return pd.DataFrame(rows).drop_duplicates('canonical_smiles')
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input-csv',type=Path,required=True);p.add_argument('--manifest',type=Path);p.add_argument('--raw-output',type=Path,required=True);p.add_argument('--lookup-output',type=Path);p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
 src=pd.read_csv(a.input_csv,dtype={'ID':str}); table=raw(src); a.raw_output.parent.mkdir(parents=True,exist_ok=True);table.to_csv(a.raw_output,index=False)
 audit={'AA_category_counts':table.parsed_AA.value_counts().to_dict(),'terminal_category_counts':table.terminal_state.value_counts().to_dict(),'tail_length_distribution':table.tail_length.value_counts(dropna=False).to_dict(),'UNKNOWN_counts':int(table.parsed_AA.eq('UNKNOWN').sum()),'single_double_counts':src.Fifth_class.fillna('missing').value_counts().to_dict(),'definitions':'AA parses Fifth name UC/OAm prefix then structure fallback; tail from UC/OAm name then structural fallback; no target or ID used.'}
 if a.manifest:
  man=pd.read_csv(a.manifest,dtype={'sample_id':str}); train=src.set_index('ID').loc[man.query("split=='train'").sample_id]; keys=train.Fifth_SMILE.map(canon); occ=table.set_index('canonical_smiles').loc[keys]
  mean,std=occ.tail_length.dropna().mean(),occ.tail_length.dropna().std(ddof=0);std=std if std and np.isfinite(std) else 1.
  aa_vocab={'UNKNOWN':0,**{x:i+1 for i,x in enumerate(sorted(occ.parsed_AA.unique()))}}; term_vocab={'unknown':0,**{x:i+1 for i,x in enumerate(sorted(occ.terminal_state.unique()))}}
  out=table.copy();out['aa_id']=out.parsed_AA.map(aa_vocab).fillna(0).astype(int);out['terminal_id']=out.terminal_state.map(term_vocab).fillna(0).astype(int);out['tail_length_normalized']=((out.tail_length-mean)/std).fillna(0.);out['tail_length_present_mask']=out.tail_present
  out.loc[out.canonical_smiles=='[Fr]',['aa_id','terminal_id','tail_length_normalized','tail_length_present_mask']]=0
  a.lookup_output.parent.mkdir(parents=True,exist_ok=True);out.rename(columns={'canonical_smiles':'smiles'})[['smiles','aa_id','terminal_id','tail_length_normalized','tail_length_present_mask']].query("smiles!='[Fr]'").to_csv(a.lookup_output,index=False)
  audit.update({'train_only':True,'tail_mean':mean,'tail_population_std':std,'aa_vocab':aa_vocab,'terminal_vocab':term_vocab,'leakage_check':'PASS: vocabulary and tail scaling fitted only on manifest train rows'})
 a.audit.parent.mkdir(parents=True,exist_ok=True);a.audit.write_text(json.dumps(audit,indent=2,default=str)+'\n')
if __name__=='__main__':main()
