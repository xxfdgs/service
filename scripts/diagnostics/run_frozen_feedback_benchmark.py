#!/usr/bin/env python3
"""One locked-weight feedback benchmark for every tree baseline and original GraphGPS."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; sys.path.insert(0,str(HERE))
from audit_deduplicated_dataset import COMPONENTS,TARGETS,enrich,sha256_file
from run_deduplicated_tree_baselines import FEATURES,map_unknown_categories,pipeline_for
from stable_formulation import build_stable_feature_sets

MODELS=("Ridge","ExtraTrees","RandomForest")
def metrics(frame):
    return {"mae":mean_absolute_error(frame.y_true,frame.y_pred),"rmse":mean_squared_error(frame.y_true,frame.y_pred)**.5,"r2":r2_score(frame.y_true,frame.y_pred)}
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output-dir',type=Path,default=ROOT/'results/deduplicated_rebaseline'); p.add_argument('--feedback-csv',type=Path,default=ROOT/'datasets_lrx/raw/feedback/20260703_validation.csv'); p.add_argument('--graphgps-predictions',type=Path,default=ROOT/'predict/direct_layer2_multi4_cat_v0_batch8_lr001/predicted_average_6props.csv'); p.add_argument('--n-jobs',type=int,default=8); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
 root=a.output_dir.resolve(); source=json.loads((root/'data_source.json').read_text()); train=pd.read_csv(root/'data_audit/dataset_with_sample_id.csv',dtype={'sample_id':str}); feedback=pd.read_csv(a.feedback_csv.resolve(),dtype={'ID':str})
 if source.get('audit_status') not in {'PASS','PASS_WITH_WARNINGS'} or len(train)!=700 or feedback.ID.duplicated().any() or feedback[TARGETS].isna().any().any(): raise RuntimeError('Invalid audited train or labelled feedback input.')
 fb=enrich(feedback); schema=SimpleNamespace(components=[{'name_column':n,'smiles_column':s,'ratio_column':r} for n,s,r in COMPONENTS]); train_x,_,_=build_stable_feature_sets(train,schema); fb_x,_,_=build_stable_feature_sets(fb,schema)
 out=root/'frozen_feedback_benchmark'; out.mkdir(parents=True,exist_ok=True); preds=[]; rows=[]
 for target in TARGETS:
  y=feedback[target].astype(float).to_numpy()
  for name,values in [('TrainMean',np.full(len(fb),train[target].mean())),('TrainMedian',np.full(len(fb),train[target].median()))]:
   f=pd.DataFrame({'sample_id':feedback.ID,'target':target,'model':name,'feature_set':'none','weight_status':'fitted_on_700_new_rows_then_frozen','y_true':y,'y_pred':values}); preds.append(f)
  for feature in FEATURES:
   xtr,xfb=map_unknown_categories(train_x[feature],fb_x[feature])
   for name in MODELS:
    pred=pipeline_for(xtr,name,a.seed,a.n_jobs).fit(xtr,train[target].astype(float)).predict(xfb)
    preds.append(pd.DataFrame({'sample_id':feedback.ID,'target':target,'model':name,'feature_set':feature,'weight_status':'fitted_on_700_new_rows_then_frozen','y_true':y,'y_pred':pred}))
 gp=pd.read_csv(a.graphgps_predictions.resolve())
 mapping={'EE_before':('true_EE_before','pred_EE_before_average'),'EE_after':('true_EE_after','pred_EE_after_average'),'Aerosolization_Efficiency':('true_Aero_Efficiency','pred_Aero_Efficiency_average'),'mRNA_Recovery_Efficiency':('true_Recovery_Efficiency','pred_Recovery_Efficiency_average')}
 if len(gp)!=len(feedback): raise RuntimeError('Frozen GraphGPS prediction row count differs from feedback.')
 for target,(truth,prediction) in mapping.items():
  if not np.allclose(gp[truth],feedback[target],atol=.011,rtol=0): raise RuntimeError(f'Frozen GraphGPS labels fail alignment: {target}')
  preds.append(pd.DataFrame({'sample_id':feedback.ID,'target':target,'model':'OriginalGraphGPS_10CheckpointEnsemble','feature_set':'molecular_graphs_and_component_ratios','weight_status':'frozen_existing_weights','y_true':feedback[target].astype(float),'y_pred':gp[prediction].astype(float)}))
 predictions=pd.concat(preds,ignore_index=True); predictions['absolute_error']=(predictions.y_true-predictions.y_pred).abs()
 for key,g in predictions.groupby(['target','model','feature_set','weight_status'],sort=True): rows.append(dict(zip(['target','model','feature_set','weight_status'],key))|{'n':len(g)}|metrics(g))
 table=pd.DataFrame(rows).sort_values(['target','mae','model']); predictions.to_csv(out/'all_models_feedback_predictions.csv',index=False); table.to_csv(out/'all_models_feedback_metrics.csv',index=False)
 (out/'provenance.json').write_text(json.dumps({'new_train_csv':str((root/'data_audit/dataset_with_sample_id.csv').resolve()),'new_train_sha256':source['dataset_sha256'],'feedback_csv':str(a.feedback_csv.resolve()),'feedback_sha256':sha256_file(a.feedback_csv.resolve()),'graphgps_prediction_csv':str(a.graphgps_predictions.resolve()),'graphgps_prediction_sha256':sha256_file(a.graphgps_predictions.resolve()),'graphgps_weights':'results/direct_layer2_multi4_cat_v0_batch8_lr001/{0..9}/ckpt','feedback_labels_used_only_for_metrics':True},indent=2)+'\n')
 print(f'Wrote {out}')
if __name__=='__main__': main()
