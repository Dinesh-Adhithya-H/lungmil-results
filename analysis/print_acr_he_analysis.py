import pandas as pd
import numpy as np

df = pd.read_csv('/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv')
pat = df.drop_duplicates('patient_id')
n_all = len(pat)

he_pat = pat[pat['has_HE']==1]
print(f'has_HE: {len(he_pat)}/{n_all} ({100*len(he_pat)/n_all:.0f}%)')

lab = df[df['label'].notna()]
he_lab = lab[lab['has_HE']==1]
print(f'Labeled biopsies - all:     {len(lab)} ({int(lab["label"].sum())} ACR+)')
print(f'Labeled biopsies - HE pats: {len(he_lab)} ({int(he_lab["label"].sum())} ACR+)')
print(f'HE patients with labels:    {he_lab["patient_id"].nunique()}')

print()
for task in ['acr_cls', 'acr_surv']:
    fmap = {'acr_cls': 'comparison_acr_cls.csv', 'acr_surv': 'comparison_acr_surv.csv'}
    comp = pd.read_csv(f'/home/aih/dinesh.haridoss/chicago_mil/results/predictions/{fmap[task]}')
    he_row = comp[comp['model']=='P1 HE'].iloc[0]
    lm_row = comp[comp['model']=='LongMK-v2'].iloc[0]
    print(f'=== {task} ===')
    print('ABMIL HE  splits:', [f"{he_row[s]:.3f}" for s in ['s0','s1','s2','s3','s4']])
    print(f'          mean={he_row["mean"]:.3f}  std={he_row["std"]:.3f}')
    print('LongMIL   splits:', [f"{lm_row[s]:.3f}" for s in ['s0','s1','s2','s3','s4']])
    print(f'          mean={lm_row["mean"]:.3f}  std={lm_row["std"]:.3f}')
    print()

uni = pd.read_csv('/home/aih/dinesh.haridoss/chicago_mil/results/predictions/unimodal_comparison.csv')
print('HE ABMIL n_test per split (acr_cls):')
print(uni[['split','n_p1_acr_HE']].head(6).to_string(index=False))
