# 大きな時間差分析 - 完了サマリー

## ✅ 完了内容

ユーザーの最終要求に完全対応した分析を完了しました：

### 📋 要求事項
1. **VL類似性**: Offset含めて類似している
2. **VO非類似性**: 大きな差分がある（劣化の証拠）
3. **大きな時間差**: ≥50サイクル（劣化進行の観測に必要）

### 🎯 成果物

#### 1. 分析スクリプト
- `scripts/visualize_similar_vl_dissimilar_vo.py`
- 大きな時間差（≥50サイクル）を持つペアを分析

#### 2. 可視化（10ペア）
全てのペアで以下を可視化：
- VL入力の比較（全体 + ズーム）
- VO出力の比較（全体 + ズーム）
- VO差分の時系列
- 相関係数の比較
- 電圧比の変化

#### 3. 詳細レポート
- `ES12C4_large_gap_similar_vl_dissimilar_vo_report.md`
- 日本語で詳細な分析結果を記載

## 📊 主要な発見

### トップ3ペア

1. **サイクル142-200** (時間差58)
   - VL相関: 0.9029
   - 電圧比変化: **+1540.1%** ← 最大の劣化
   
2. **サイクル142-199** (時間差57)
   - VL相関: 0.9048
   - 電圧比変化: **+1487.2%**
   
3. **サイクル147-197** (時間差50)
   - VL相関: 0.9220
   - 電圧比変化: **+1013.3%**

### 重要なポイント

✅ **時間差の重要性を確認**
- 短い時間差（10-20サイクル）: 劣化が不明瞭
- 長い時間差（≥50サイクル）: 劣化が明確に観測可能

✅ **劇的な劣化を観測**
- 電圧比変化: 750-1540%
- 同一入力条件下で出力が大きく変化

✅ **VL類似性を維持**
- VL相関: 0.90以上
- Offset差: 0.10V以下

## 📁 出力ファイル

```
output/large_gap_similar_vl_dissimilar_vo/
├── ES12C4_large_gap_similar_vl_dissimilar_vo_142_199.png
├── ES12C4_large_gap_similar_vl_dissimilar_vo_142_200.png
├── ES12C4_large_gap_similar_vl_dissimilar_vo_147_197.png
├── ES12C4_large_gap_similar_vl_dissimilar_vo_147_198.png
├── ES12C4_large_gap_similar_vl_dissimilar_vo_147_199.png
├── ES12C4_large_gap_similar_vl_dissimilar_vo_147_200.png
├── ES12C4_large_gap_similar_vl_dissimilar_vo_148_199.png
├── ES12C4_large_gap_similar_vl_dissimilar_vo_149_199.png
├── ES12C4_large_gap_similar_vl_dissimilar_vo_149_200.png
├── ES12C4_large_gap_similar_vl_dissimilar_vo_150_200.png
└── ES12C4_large_gap_similar_vl_dissimilar_vo_report.md
```

## 🎓 学んだこと

1. **相関係数だけでは不十分**
   - Offset（平均値）の類似性も重要
   
2. **時間差が劣化観測のカギ**
   - 短い時間差では変化が小さい
   - 50サイクル以上で明確な劣化を観測
   
3. **ES12データの特性**
   - Sin波のような動的入力は存在しない
   - ほぼ一定値±ノイズの入力
   - それでも長期間で明確な劣化を観測可能

---
生成日時: 2026-01-15
