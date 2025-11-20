# export_cats_dogs.py
import os
import tensorflow_datasets as tfds
from pathlib import Path
import tensorflow as tf

OUT = Path("data")
OUT.mkdir(exist_ok=True)

# load split as tf.data.Dataset
ds = tfds.load("cats_vs_dogs", split="train", as_supervised=True)
print("Total examples will be iterated and saved to data/cats and data/dogs ...")

count = 0
for img, label in tfds.as_numpy(ds):
    # label: 0 = cat, 1 = dog
    sub = "dogs" if int(label) == 1 else "cats"
    d = OUT / sub
    d.mkdir(parents=True, exist_ok=True)
    # write images as jpg
    p = d / f"{sub}_{count:05d}.jpg"
    tf.keras.utils.save_img(str(p), img)
    count += 1
    if count % 500 == 0:
        print(f"Saved {count} images...")
print(f"Done — saved {count} images into {OUT}/cats and {OUT}/dogs")