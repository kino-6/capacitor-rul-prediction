# NASA PCOE コンデンサ劣化予測プロジェクト

NASA PCOE Dataset No.12 (ES12) を用いたコンデンサ劣化予測システムの開発

---

## 🚀 メインプロジェクト: RUL予測モデル

**本プロジェクトのメイン成果物は [rul_modeling/](rul_modeling/) ディレクトリにあります。**

### 📊 主要な成果

- **異常検知**: Training FP 5.0%, Late FN 5.2%
- **劣化度予測**: Test MAE 0.0036, R² 0.9996
- **誤報率削減**: FPR 41.4% → 13.1% (68.2%改善)
- **段階的アラートシステム**: 4レベル (INFO/WARNING/ALERT/CRITICAL)

### 📖 詳細ドキュメント

**👉 [rul_modeling/README.md](rul_modeling/README.md) - メインプロジェクトの入り口**

- [最終プロジェクトレポート](rul_modeling/docs/FINAL_PROJECT_REPORT.md)
- [統合デモスクリプト](rul_modeling/scripts/end_to_end_inference_demo.py)
- [タスク管理](rul_modeling/.kiro/specs/rul_model_spec/tasks.md)

---

## 📁 このディレクトリについて

このトップレベルディレクトリには、初期の探索的データ分析（EDA）の成果物が含まれています。
**実用的なRUL予測システムは [rul_modeling/](rul_modeling/) を参照してください。**

## 🎯 初期EDAの概要

本プロジェクトは、NASA PCOE（Prognostics Center of Excellence）が提供するコンデンサ電気ストレスデータセット（ES12）を用いて、**同一入力条件下での出力応答の変化（劣化）**を分析します。

## 📊 主要な成果

### ✅ 最終結論

**[📈 詳細分析レポート（メイン）](output/large_gap_similar_vl_dissimilar_vo/ES12C4_large_gap_similar_vl_dissimilar_vo_report.md)**

**発見事項**:
- VL入力が類似（Offset含む）しているサイクルペアを特定
- VO出力に大きな差分（劣化の証拠）を観測
- **時間差50サイクル以上**で明確な劣化進行を確認
- 電圧比変化: **750-1540%**の劇的な変化

**トップ3ペア**:
1. サイクル142-200（時間差58）: 電圧比変化 **+1540%**
2. サイクル142-199（時間差57）: 電圧比変化 **+1487%**
3. サイクル147-197（時間差50）: 電圧比変化 **+1013%**

### 📁 出力ファイル

```
output/
├── large_gap_similar_vl_dissimilar_vo/    # メイン分析結果
│   ├── ES12C4_large_gap_similar_vl_dissimilar_vo_report.md  # 詳細レポート
│   ├── SUMMARY.md                          # サマリー
│   └── *.png                               # 可視化（10ペア）
└── archive/                                # 参考資料
    ├── best_identical_vl/                  # VL類似性分析
    ├── truly_identical_vl/                 # 厳密なVL類似性
    ├── similar_vl_dissimilar_vo/           # 初期分析（小さな時間差）
    └── ...                                 # その他の探索的分析
```

## 🔍 分析の経緯

### データの現実

ES12データセットの特徴:
- ❌ **Sin波のような周期的波形は存在しない**（FFT分析で周期性比率0.003-0.004）
- ✅ **ほぼ一定値 ± ノイズ**のパターンが主
- ✅ 実運用環境の実データ（制御された実験データではない）

### 分析の進化

1. **初期**: 相関係数のみで類似性を判定 → Offset差を見逃す問題
2. **改善**: Offset（平均値）を含めた類似性判定
3. **課題**: 短い時間差（10-20サイクル）では劣化が不明瞭
4. **最終**: **時間差≥50サイクル**で明確な劣化を観測 ✅

### 重要な学び

| 要素 | 発見 |
|------|------|
| **相関係数** | 形状類似性を示すが、Offset差を見逃す |
| **Offset** | 平均値の類似性も重要（VL類似性の判定に必須） |
| **時間差** | 50サイクル以上で劣化が明確に観測可能 |
| **VO差分** | 同一入力条件下での出力変化が劣化の証拠 |

## 🚀 クイックスタート

### 必要要件

- Python 3.10以上
- UV（Pythonパッケージマネージャー）
- ES12.matデータファイル

### セットアップ

```bash
# 依存関係のインストール
uv sync

# データセットのダウンロード（自動）
uv run python scripts/download_dataset.py

# または手動でES12.matをdata/raw/に配置
```

### 分析の実行

```bash
# メイン分析の再実行
uv run python scripts/visualize_similar_vl_dissimilar_vo.py

# 他の探索的分析
uv run python scripts/find_visually_identical_vl_cycles.py
uv run python scripts/find_truly_identical_vl_with_vo_analysis.py
```

## 📖 ドキュメント

### メインレポート

- **[詳細分析レポート](output/large_gap_similar_vl_dissimilar_vo/ES12C4_large_gap_similar_vl_dissimilar_vo_report.md)** - 最終結論と可視化
- **[サマリー](output/large_gap_similar_vl_dissimilar_vo/SUMMARY.md)** - 簡潔なまとめ

### 参考資料（アーカイブ）

- [VL類似性分析](output/archive/best_identical_vl/ES12C4_best_identical_vl_report.md) - 時間差とVL類似性のトレードオフ
- [厳密なVL類似性](output/archive/truly_identical_vl/) - Offset含む厳密な類似性
- [初期分析](output/archive/similar_vl_dissimilar_vo/) - 小さな時間差での分析

### 技術ドキュメント

- [ES12データ構造ガイド](docs/es12_data_structure_guide.md)
- [実データ分析ガイド](docs/real_data_analysis_guide.md)
- [警告管理ガイド](docs/warning_management_guide.md)

## 🧪 テスト

```bash
# 全テストの実行
uv run pytest

# カバレッジ付き
uv run pytest --cov=src --cov-report=html

# プロパティベーステストのみ
uv run pytest tests/property/
```

## 📊 データセット情報

**NASA PCOE Dataset No.12: Capacitor Electrical Stress**

- **ファイル**: ES12.mat（約1.2GB）
- **形式**: MATLAB v7.3 HDF5
- **内容**: 8個のコンデンサ（ES12C1～ES12C8）のEISデータ
- **測定項目**: VL（入力電圧）、VO（出力電圧）、容量、ESR
- **サイクル数**: 各コンデンサ約200サイクル

**データ取得**:
- [NASA PCOE Dataset Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

## 🛠️ プロジェクト構造

```
.
├── README.md                    # このファイル（EDAプロジェクト）
├── pyproject.toml              # プロジェクト設定
├── data/
│   └── raw/                    # 生データ（ES12.mat）
├── output/
│   ├── large_gap_similar_vl_dissimilar_vo/  # メイン結果
│   └── archive/                # 参考資料
├── scripts/                    # 分析スクリプト
├── src/                        # ソースコード
├── tests/                      # テスト
├── docs/                       # ドキュメント
└── rul_modeling/               # 🆕 RUL予測モデル開発
    ├── README.md               # RULモデリングプロジェクト
    ├── .kiro/specs/            # Spec-driven development
    ├── src/                    # モデルソースコード
    ├── tests/                  # モデルテスト
    ├── notebooks/              # 実験用ノートブック
    ├── output/                 # モデル出力
    └── docs/                   # モデルドキュメント
```

## 🤝 貢献

貢献を歓迎します！以下の手順でお願いします：

1. このリポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📝 ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照

## 🙏 謝辞

- NASA PCOE（Prognostics Center of Excellence）によるデータセット提供
- オープンソースコミュニティ

## 📧 連絡先

問題や質問がある場合は、GitHubのIssueトラッカーで報告してください。

---

## 🗂️ アーカイブファイルについて

以下のファイルは過去の開発・デバッグ過程で生成された一時ファイルです（参考用に保持）:

- `warning_analysis.py` - テスト警告分析スクリプト（デバッグ用）
- `warning_config.yaml` - 警告管理設定（デバッグ用）
- `test_output_with_warnings.log` - 旧テストログ
- `test_warnings_analysis.log` - 旧警告分析ログ
- `test_output/` - 一時テスト出力

これらのファイルは現在のプロジェクトでは使用されていません。

---

**最終更新**: 2026-01-28
