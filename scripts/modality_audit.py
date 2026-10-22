import pandas as pd, numpy as np

splits = pd.read_csv('/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv')
annot  = pd.read_csv('/home/aih/dinesh.haridoss/chicago/ACR_biopsies_clean_dates_fixed (1).csv')

splits['anchor_dt'] = pd.to_datetime(splits['anchor_dt']).dt.date
annot['Biopsy Date'] = pd.to_datetime(annot['Biopsy Date']).dt.date

# The 5: annotation-confirmed but no HE
no_he_annot = splits[splits['acr_annotated'] & ~splits['has_HE']]
print(f"Annotation-confirmed samples WITHOUT HE: {len(no_he_annot)}")
print()
cols = ['file','patient_id','anchor_dt','acr_grade','has_HE','has_BAL','has_CT','has_Clinical']
print(no_he_annot[cols].to_string(index=False))
print()

# Cross-ref with annotation to see what slide name maps to these
annot_visits = annot.drop_duplicates(subset=['Record id','Biopsy Date'])
for _, row in no_he_annot.iterrows():
    match = annot_visits[
        (annot_visits['Record id'] == str(row['patient_id'])) &
        (annot_visits['Biopsy Date'] == row['anchor_dt'])
    ]
    print(f"{row['patient_id']}  {row['anchor_dt']}  {row['file']}  grade={row['acr_grade']}")
    if len(match):
        print(f"  -> annotation: slide={match.iloc[0]['Slide Name']}  grade={match.iloc[0]['ACR Status/Grade']}")
    else:
        print("  -> no annotation match (shouldn't happen)")
