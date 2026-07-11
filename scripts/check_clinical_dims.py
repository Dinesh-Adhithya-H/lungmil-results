import torch
from pathlib import Path

base = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/phase2")

print(f"{'Split':<8} {'Combo':<25} {'Fold':<6} {'File':<30} {'Clinical_in_dim'}")
print("-" * 85)

for split in [0, 1, 2, 3, 4]:
    for fold in [0, 1]:
        fold_dir = base / f"split{split}_fold{fold}"
        if not fold_dir.exists():
            continue
        for combo_dir in sorted(fold_dir.iterdir()):
            if combo_dir.name.startswith("slot"):
                continue
            models = list(combo_dir.glob("model_*.pt"))
            if not models:
                continue
            m = models[0]
            try:
                obj = torch.load(m, map_location="cpu", weights_only=False)
                sd = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj.state_dict()
                w = sd.get("encoders.Clinical.backbone.0.weight")
                dim = w.shape[1] if w is not None else "N/A"
                print(f"  split{split}   {combo_dir.name:<25} fold{fold}  {m.name:<30} {dim}")
            except Exception as e:
                print(f"  split{split}   {combo_dir.name:<25} fold{fold}  {m.name:<30} ERR: {e}")
