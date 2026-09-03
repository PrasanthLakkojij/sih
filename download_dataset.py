"""
Download all S-Dataset CSV files from IO-VNBD GitHub repository.
"""
import urllib.request
import os
import sys

BASE_URL = "https://raw.githubusercontent.com/onyekpeu/IO-VNBD/master/Synchronised%20V%20abd%20S%20datasets/Uncategorised%20IOVNB%20Dataset/S-Dataset/"
OUT_DIR = os.path.join("data", "S-Dataset")

# Complete list of S-Dataset files from the GitHub directory listing
FILES = [
    "S-M.csv",
    "S-S1.csv", "S-S2.csv", "S-S3a.csv", "S-S3b.csv", "S-S3c.csv", "S-S4.csv",
    "S-Vfa01.csv", "S-Vfa02.csv",
    "S-Vta10.csv", "S-Vta11.csv", "S-Vta12.csv", "S-Vta13.csv", "S-Vta14.csv",
    "S-Vta15.csv", "S-Vta16.csv", "S-Vta17.csv", "S-Vta19.csv",
    "S-Vta1a.csv", "S-Vta1b.csv",
    "S-Vta2.csv", "S-Vta20.csv", "S-Vta21.csv", "S-Vta22.csv", "S-Vta23.csv",
    "S-Vta24.csv", "S-Vta25.csv", "S-Vta26.csv", "S-Vta27.csv", "S-Vta28.csv",
    "S-Vta29.csv", "S-Vta3.csv", "S-Vta30.csv",
    "S-Vta4.csv", "S-Vta5.csv", "S-Vta6.csv", "S-Vta7.csv", "S-Vta8.csv", "S-Vta9.csv",
    "S-Vtb1.csv", "S-Vtb10.csv", "S-Vtb11.csv", "S-Vtb12.csv",
    "S-Vtb2.csv", "S-Vtb3.csv", "S-Vtb4.csv", "S-Vtb5.csv", "S-Vtb6.csv",
    "S-Vtb7.csv", "S-Vtb8.csv", "S-Vtb9.csv",
    "S-Vw1.csv", "S-Vw10.csv", "S-Vw11.csv", "S-Vw12.csv", "S-Vw13.csv",
    "S-Vw14a.csv", "S-Vw14b.csv", "S-Vw14c.csv",
    "S-Vw15.csv", "S-Vw16a.csv", "S-Vw16b.csv", "S-Vw17.csv",
    "S-Vw2.csv", "S-Vw3.csv", "S-Vw4.csv", "S-Vw5.csv", "S-Vw6.csv",
    "S-Vw7.csv", "S-Vw8.csv", "S-Vw9.csv",
    "S-Y1.csv",
]

os.makedirs(OUT_DIR, exist_ok=True)

total = len(FILES)
for i, fname in enumerate(FILES, 1):
    dest = os.path.join(OUT_DIR, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[{i}/{total}] SKIP (exists): {fname}")
        continue
    url = BASE_URL + fname
    print(f"[{i}/{total}] Downloading: {fname} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = os.path.getsize(dest) / 1024
        print(f"OK ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"FAILED: {e}")

print(f"\nDone. Files in {OUT_DIR}:")
for f in sorted(os.listdir(OUT_DIR)):
    sz = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
    print(f"  {f:20s}  {sz:>10.1f} KB")
