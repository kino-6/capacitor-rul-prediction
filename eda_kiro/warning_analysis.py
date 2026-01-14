#!/usr/bin/env python3
"""
警告分析スクリプト
test_warnings_analysis.logファイルを読み込んで警告を分類・集計する
"""

import re
from collections import defaultdict, Counter

def analyze_warnings():
    """警告ログファイルを分析して分類・集計する"""
    
    warning_categories = defaultdict(int)
    warning_details = defaultdict(list)
    
    try:
        with open('test_warnings_analysis.log', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("test_warnings_analysis.logファイルが見つかりません")
        return
    
    lines = content.split('\n')
    
    # 警告パターンの定義
    patterns = {
        'japanese_font_missing': r'UserWarning: Glyph.*missing from font\(s\) DejaVu Sans',
        'deprecation_warning': r'DeprecationWarning',
        'future_warning': r'FutureWarning',
        'runtime_warning': r'RuntimeWarning',
        'pending_deprecation': r'PendingDeprecationWarning',
        'scipy_warning': r'scipy.*Warning',
        'statsmodels_warning': r'statsmodels.*Warning',
        'pandas_warning': r'pandas.*Warning',
        'japanize_matplotlib_warning': r'japanize.*Warning',
        'setuptools_warning': r'setuptools.*Warning',
        'user_warning': r'UserWarning',
    }
    
    # 各行を分析
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 日本語フォント不足警告（最も多い）
        if re.search(patterns['japanese_font_missing'], line):
            warning_categories['日本語フォント不足 (matplotlib)'] += 1
            # 具体的な文字を抽出
            glyph_match = re.search(r'Glyph (\d+) \(\\N\{([^}]+)\}\)', line)
            if glyph_match:
                char_code, char_name = glyph_match.groups()
                warning_details['日本語フォント不足'].append(f"{char_name} (U+{int(char_code):04X})")
        
        # その他の警告タイプ
        elif 'DeprecationWarning' in line:
            warning_categories['非推奨警告 (DeprecationWarning)'] += 1
            if 'japanize' in line.lower():
                warning_details['非推奨警告'].append('japanize-matplotlib関連')
            elif 'setuptools' in line.lower():
                warning_details['非推奨警告'].append('setuptools関連')
            elif 'pandas' in line.lower():
                warning_details['非推奨警告'].append('pandas関連')
            else:
                warning_details['非推奨警告'].append('その他')
                
        elif 'FutureWarning' in line:
            warning_categories['将来警告 (FutureWarning)'] += 1
            warning_details['将来警告'].append(line[:100] + '...' if len(line) > 100 else line)
            
        elif 'RuntimeWarning' in line:
            warning_categories['実行時警告 (RuntimeWarning)'] += 1
            warning_details['実行時警告'].append(line[:100] + '...' if len(line) > 100 else line)
            
        elif 'PendingDeprecationWarning' in line:
            warning_categories['保留非推奨警告'] += 1
            
        elif 'UserWarning' in line and 'scipy' in line.lower():
            warning_categories['scipy警告'] += 1
            warning_details['scipy警告'].append('MATLAB形式ファイル読み込み関連')
            
        elif 'UserWarning' in line and 'statsmodels' in line.lower():
            warning_categories['statsmodels警告'] += 1
            warning_details['statsmodels警告'].append('統計計算関連')
    
    # 結果の出力
    print("=" * 60)
    print("NASA PCOE EDA テストスイート警告分析レポート")
    print("=" * 60)
    print()
    
    print("📊 警告カテゴリ別集計:")
    print("-" * 40)
    total_warnings = sum(warning_categories.values())
    
    for category, count in sorted(warning_categories.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_warnings * 100) if total_warnings > 0 else 0
        print(f"{category}: {count:,} 件 ({percentage:.1f}%)")
    
    print(f"\n合計警告数: {total_warnings:,} 件")
    print()
    
    # 詳細分析
    print("🔍 詳細分析:")
    print("-" * 40)
    
    # 日本語フォント不足の詳細
    if '日本語フォント不足' in warning_details:
        font_chars = Counter(warning_details['日本語フォント不足'])
        print(f"\n📝 日本語フォント不足の詳細 (上位10文字):")
        for char, count in font_chars.most_common(10):
            print(f"  {char}: {count} 回")
    
    # 非推奨警告の詳細
    if '非推奨警告' in warning_details:
        dep_sources = Counter(warning_details['非推奨警告'])
        print(f"\n⚠️  非推奨警告の内訳:")
        for source, count in dep_sources.items():
            print(f"  {source}: {count} 件")
    
    print()
    print("💡 推奨対応:")
    print("-" * 40)
    print("1. 日本語フォント不足警告:")
    print("   - 機能に影響なし（表示のみの問題）")
    print("   - 日本語フォントをシステムにインストールすることで解決可能")
    print("   - 本番環境では英語ラベルの使用を検討")
    print()
    print("2. 非推奨警告:")
    print("   - 依存関係の更新時に対応を検討")
    print("   - 現在は機能に影響なし")
    print()
    print("3. その他の警告:")
    print("   - 統計計算やファイル読み込み関連の情報警告")
    print("   - 機能に影響なし")

if __name__ == "__main__":
    analyze_warnings()