cat > make_dummy_data.py << 'EOF'
from pathlib import Path
from PIL import Image
import numpy as np

OUT = Path("data")
cats = OUT / "cats"
dogs = OUT / "dogs"
cats.mkdir(parents=True, exist_ok=True)
dogs.mkdir(parents=True, exist_ok=True)

# create 10 small random images each
for i in range(10):
    img = (np.random.rand(180,180,3) * 255).astype("uint8")
    Image.fromarray(img).save(cats / f"cat_{i:03d}.jpg")

for i in range(10):
    img = (np.random.rand(180,180,3) * 255).astype("uint8")
    Image.fromarray(img).save(dogs / f"dog_{i:03d}.jpg")

print("Dummy dataset created in data/cats and data/dogs")
EOF