"""
# Image resizing and scaling
from PIL import Image
import os

def resize_images_in_folder(folder, target_size=(224, 224)):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                path = os.path.join(root, file)
                img = Image.open(path).convert("RGB")
                img = img.resize(target_size, Image.LANCZOS)
                img.save(path)  # Overwrites original (or save to new folder)

resize_images_in_folder(r'C:\Users\himan\Downloads\AI Model\Processed Data', target_size=(224, 224))
"""
"""
# Remove Blurry/Irrelevant Images
import cv2
import numpy as np

def is_blurry(image_path, threshold=100.0):
    img = cv2.imread(image_path)
    if img is None:
        return True  # treat unreadable as "bad"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

def remove_blurry_images(folder, threshold=100.0):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                path = os.path.join(root, file)
                if is_blurry(path, threshold):
                    print(f"Removing blurry image: {path}")
                    os.remove(path)
"""
"""
# Remove corrupt or broken images
from PIL import Image
import os

def remove_corrupt_images(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    img.verify()  # Will throw an exception if corrupt
                except Exception:
                    print(f"Removing corrupt image: {path}")
                    os.remove(path)
remove_corrupt_images(r'C:\Users\himan\Downloads\AI Model\Processed Data')
"""
"""
# Remove duplicates
import hashlib

def remove_duplicate_images(folder):
    hashes = {}
    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            with open(path, 'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash in hashes:
                print(f"Removing duplicate: {path}")
                os.remove(path)
            else:
                hashes[filehash] = path
remove_duplicate_images('C:\Users\himan\Downloads\AI Model\Processed Data')
"""
