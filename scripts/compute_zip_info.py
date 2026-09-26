#!/usr/bin/env python3
"""
Compute sizes and hash values (SHA-256, MD5) for all supplied data ZIP files.
"""

import os
import hashlib
from pathlib import Path

def compute_file_hash(file_path: Path):
    sha256_hash = hashlib.sha256()
    md5_hash = hashlib.md5()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256_hash.update(chunk)
            md5_hash.update(chunk)
            
    return sha256_hash.hexdigest(), md5_hash.hexdigest()

def main():
    root_dir = Path(__file__).resolve().parents[1]
    datasets_dir = root_dir / "datasets"
    
    zip_files = sorted(datasets_dir.rglob("*.zip"))
    
    if not zip_files:
        print("No .zip files found in datasets directory.")
        return

    print("=" * 90)
    print(f"{'Supplied Data ZIP File':<55} | {'Size (Bytes)':<12} | {'SHA-256 Digest'}")
    print("=" * 90)

    total_bytes = 0
    
    for zip_path in zip_files:
        rel_path = zip_path.relative_to(root_dir)
        size = zip_path.stat().st_size
        total_bytes += size
        sha256, md5 = compute_file_hash(zip_path)
        
        print(f"{str(rel_path):<55} | {size:<12,d} | {sha256}")
        print(f"  └─ MD5: {md5}")
        print("-" * 90)

    print("=" * 90)
    print(f"Total Supplied ZIP Files Count: {len(zip_files)}")
    print(f"Overall Supplied Data ZIP Size: {total_bytes:,d} bytes ({total_bytes / (1024 * 1024):.2f} MB)")
    print("=" * 90)

if __name__ == "__main__":
    main()
