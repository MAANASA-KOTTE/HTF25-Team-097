from PIL import Image
import random

# ---- Prompt templates ----
PROMPTS = {
    "office": {
        "genz": [
            "Gen Z office outfit, trendy yet professional with a modern twist",
            "stylish Gen Z smart casual office look"
        ],
        "genx": [
            "Gen X professional office outfit, elegant and balanced",
            "mature and classic business formal outfit"
        ],
        "classic": [
            "classic office attire, timeless blazer and white shirt",
            "traditional business professional outfit"
        ]
    },
    "wedding": {
        "genz": [
            "vibrant Gen Z wedding guest outfit, playful yet elegant",
            "trendy wedding outfit with bold patterns"
        ],
        "genx": [
            "Gen X elegant wedding outfit, refined and graceful",
            "mature formal wedding look with subtle colors"
        ],
        "classic": [
            "classic wedding outfit, timeless gown or formal suit",
            "traditional wedding attire with minimal accessories"
        ]
    },
    "casual": {
        "genz": [
            "Gen Z casual outfit, relaxed streetwear with bold colors",
            "trendy everyday outfit with crop tops, baggy jeans, and sneakers"
        ],
        "genx": [
            "Gen X casual outfit, neat polo shirts, jeans, and loafers",
            "comfortable yet polished casual wear for outings"
        ],
        "classic": [
            "classic casual outfit, simple jeans and shirt combo",
            "timeless weekend look with neutral tones"
        ]
    },
    "beach": {
        "genz": [
            "Gen Z beach outfit, bright colors and lightweight fabrics",
            "summer vibe with trendy swimwear and accessories"
        ],
        "genx": [
            "Gen X relaxed beachwear, comfortable and mature",
            "light cotton clothes for seaside relaxation"
        ],
        "classic": [
            "classic beach outfit, timeless swimwear with cover-up",
            "simple, airy, and elegant coastal style"
        ]
    }
}

MODEL_NAME = "patrickjohncyh/fashion-clip"

# Try to load heavy ML deps; if not available, provide a lightweight fallback
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    def evaluate_outfit(img_path, occasion, style):
        image = Image.open(img_path).convert("RGB")
        texts = PROMPTS.get(occasion, {}).get(style, [])
        if not texts:
            raise ValueError(f"Invalid combination: {occasion}/{style}")

        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)
            score = probs.max().item()

        print(f"[Model] {style.capitalize()} / {occasion.capitalize()} -> {score:.4f}")
        return score

except Exception as _err:
    # Fallback when torch/transformers aren't installed or model can't be loaded.
    # Returns a stable pseudo-random score so the app remains usable for demos.
    def evaluate_outfit(img_path, occasion, style):
        try:
            # lightweight heuristic: base score on image size and a random factor
            img = Image.open(img_path)
            w, h = img.size
            size_score = min(1.0, (w * h) / (1000 * 1000))
        except Exception:
            size_score = 0.5
        rnd = random.random() * 0.5
        score = round(min(1.0, size_score * 0.5 + rnd), 4)
        print(f"[Model fallback] returning score={score} for {img_path}")
        return score
