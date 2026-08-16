from pathlib import Path
from inference import predict
import json

def process_model(model_id : str):
    checkpoint_path = "checkpoints/exp01_baseline/fold_3/self_contained.pt"

    renders_dir = Path(f"../assets/public/{model_id}/renders")
    render_image_paths = sorted(
        str(p) for p in renders_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )

    print(f"Found {len(render_image_paths)} views")  

    result = predict(
        checkpoint_path=checkpoint_path,
        render_image_paths=render_image_paths,
        geometry_json=None, 
    )

    return result