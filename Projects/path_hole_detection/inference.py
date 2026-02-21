from ultralytics import YOLO
import os

# Load model ONCE
model = YOLO("model/best.pt")

def detect_folder(input_dir="sample_inputs", save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)

    images = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if len(images) == 0:
        print("❌ No images found in", input_dir)
        return

    print(f"✅ Found {len(images)} images")

    results = model.predict(
        source=images,
        conf=0.25,
        save=True,
        project=save_dir,
        name="predict"
    )

    print("🎯 Inference completed")
    print("📁 Results saved in:", os.path.join(save_dir, "predict"))


if __name__ == "__main__":
    detect_folder()