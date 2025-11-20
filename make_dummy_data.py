from pathlib import Path
from PIL import Image
import numpy as np

OUT = Path("data")
cats = OUT / "cats"
dogs = OUT / "dogs"
cats.mkdir(parents=True, exist_ok=True)
dogs.mkdir(parents=True, exist_ok=True)

for i in range(10):
    img = (np.random.rand(180,180,3) * 255).astype("uint8")
    Image.fromarray(img).save(cats / f"cat_{i:03d}.jpg", quality=85)

for i in range(10):
    img = (np.random.rand(180,180,3) * 255).astype("uint8")
    Image.fromarray(img).save(dogs / f"dog_{i:03d}.jpg", quality=85)

print("Created dummy dataset: data/cats (10 images), data/dogs (10 images)")
