#!/usr/bin/env python3
"""
NASA PCOE Dataset No.12 Download Script

This script downloads the NASA PCOE Capacitor Electrical Stress Dataset No.12
from the official repository and places it in the correct directory structure.
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Optional


def download_with_progress(url: str, filepath: Path, expected_size: Optional[int] = None) -> bool:
    """
    Download a file with progress indication.
    
    Args:
        url: URL to download from
        filepath: Local path to save the file
        expected_size: Expected file size in bytes (optional)
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                downloaded = min(block_num * block_size, total_size)
                print(f"\rダウンロード中... {percent:3d}% ({downloaded:,} / {total_size:,} bytes)", end="")
            else:
                downloaded = block_num * block_size
                print(f"\rダウンロード中... {downloaded:,} bytes", end="")
        
        print(f"ダウンロード開始: {url}")
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print("\nダウンロード完了!")
        
        # Verify file size if expected
        if expected_size:
            actual_size = filepath.stat().st_size
            if actual_size != expected_size:
                print(f"警告: ファイルサイズが期待値と異なります (期待: {expected_size:,}, 実際: {actual_size:,})")
                return False
        
        return True
        
    except Exception as e:
        print(f"\nダウンロードエラー: {e}")
        return False


def verify_checksum(filepath: Path, expected_checksum: str, algorithm: str = "sha256") -> bool:
    """
    Verify file checksum.
    
    Args:
        filepath: Path to the file to verify
        expected_checksum: Expected checksum value
        algorithm: Hash algorithm to use
        
    Returns:
        True if checksum matches, False otherwise
    """
    try:
        hash_obj = hashlib.new(algorithm)
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        actual_checksum = hash_obj.hexdigest()
        if actual_checksum.lower() == expected_checksum.lower():
            print("✓ チェックサム検証成功")
            return True
        else:
            print(f"✗ チェックサム検証失敗")
            print(f"  期待値: {expected_checksum}")
            print(f"  実際値: {actual_checksum}")
            return False
            
    except Exception as e:
        print(f"チェックサム検証エラー: {e}")
        return False


def main():
    """Main function to download NASA PCOE Dataset No.12."""
    parser = argparse.ArgumentParser(
        description="NASA PCOE Dataset No.12 ダウンロードスクリプト"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("data/raw"),
        help="出力ディレクトリ (デフォルト: data/raw)"
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="既存ファイルを上書き"
    )
    parser.add_argument(
        "--verify-checksum",
        action="store_true",
        help="ダウンロード後にチェックサムを検証"
    )
    
    args = parser.parse_args()
    
    # Dataset information
    dataset_info = {
        "name": "ES12.mat",
        "url": "https://ti.arc.nasa.gov/c/3/",  # Placeholder URL - actual URL needs to be updated
        "size": 1_200_000_000,  # Approximately 1.2GB
        "checksum": "placeholder_checksum",  # Actual checksum needs to be provided
        "description": "NASA PCOE Capacitor Electrical Stress Dataset No.12"
    }
    
    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Target file path
    target_file = output_dir / dataset_info["name"]
    
    # Check if file already exists
    if target_file.exists() and not args.force:
        print(f"ファイルが既に存在します: {target_file}")
        print("上書きする場合は --force オプションを使用してください")
        return 0
    
    print("=" * 60)
    print("NASA PCOE Dataset No.12 ダウンロード")
    print("=" * 60)
    print(f"データセット: {dataset_info['description']}")
    print(f"ファイル名: {dataset_info['name']}")
    print(f"サイズ: {dataset_info['size']:,} bytes (~{dataset_info['size'] / (1024**3):.1f} GB)")
    print(f"保存先: {target_file}")
    print()
    
    # Important notice about manual download
    print("⚠️  重要な注意事項:")
    print("現在、NASA PCOE データセットは手動でのダウンロードが必要です。")
    print()
    print("手動ダウンロード手順:")
    print("1. ブラウザで以下のURLにアクセス:")
    print("   https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
    print()
    print("2. 'Capacitor Electrical Stress' セクションを探す")
    print()
    print("3. 'Dataset No.12' または '12.CapacitorElectricalStress' をクリック")
    print()
    print("4. ES12.mat ファイルをダウンロード")
    print()
    print("5. ダウンロードしたファイルを以下の場所に配置:")
    print(f"   {target_file.absolute()}")
    print()
    
    # Check if user wants to continue with automatic download attempt
    try:
        response = input("自動ダウンロードを試行しますか？ (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("手動ダウンロードを行ってください。")
            return 0
    except KeyboardInterrupt:
        print("\n中断されました。")
        return 1
    
    # Attempt automatic download (this will likely fail due to access restrictions)
    print("\n自動ダウンロードを試行中...")
    success = download_with_progress(
        dataset_info["url"], 
        target_file, 
        dataset_info["size"]
    )
    
    if not success:
        print("\n自動ダウンロードに失敗しました。")
        print("上記の手動ダウンロード手順に従ってください。")
        return 1
    
    # Verify checksum if requested
    if args.verify_checksum and dataset_info["checksum"] != "placeholder_checksum":
        print("\nチェックサム検証中...")
        if not verify_checksum(target_file, dataset_info["checksum"]):
            print("チェックサム検証に失敗しました。ファイルが破損している可能性があります。")
            return 1
    
    print(f"\n✓ データセットのダウンロードが完了しました: {target_file}")
    print("\n次のステップ:")
    print("1. EDA分析を実行:")
    print(f"   uv run nasa-pcoe-eda --data {target_file} --output output/")
    print()
    print("2. Jupyter Notebookで探索:")
    print("   uv run jupyter notebook notebooks/exploratory_analysis.ipynb")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())