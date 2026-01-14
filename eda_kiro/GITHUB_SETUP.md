# GitHub リポジトリ作成とPush手順

## 📝 手順

### 1. GitHubでリポジトリを作成

1. [GitHub](https://github.com)にアクセスしてログイン
2. 右上の「+」→「New repository」をクリック
3. リポジトリ情報を入力：
   - **Repository name**: `es12-capacitor-degradation-analysis`（または任意の名前）
   - **Description**: `ES12 Capacitor Degradation Analysis - NASA PCOE Dataset`
   - **Public** または **Private** を選択
   - ✅ **Add a README file**: チェックを**外す**（既にREADME.mdがあるため）
   - ✅ **Add .gitignore**: チェックを**外す**（既に.gitignoreがあるため）
   - ✅ **Choose a license**: MIT License を選択（推奨）
4. 「Create repository」をクリック

### 2. リモートリポジトリを追加してPush

GitHubでリポジトリを作成したら、以下のコマンドを実行：

```bash
# リモートリポジトリを追加（URLは自分のリポジトリに置き換える）
git remote add origin https://github.com/YOUR_USERNAME/es12-capacitor-degradation-analysis.git

# ブランチ名をmainに変更（GitHubのデフォルトに合わせる）
git branch -M main

# 初回Push
git push -u origin main
```

### 3. 認証

Pushする際に認証が求められます：

#### Personal Access Token（推奨）

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 「Generate new token」→「Generate new token (classic)」
3. スコープで「repo」にチェック
4. トークンを生成してコピー
5. Pushする際、パスワードの代わりにトークンを入力

#### SSH Key（代替方法）

```bash
# SSH keyを生成（まだない場合）
ssh-keygen -t ed25519 -C "your_email@example.com"

# SSH keyをGitHubに追加
# 1. ~/.ssh/id_ed25519.pub の内容をコピー
# 2. GitHub → Settings → SSH and GPG keys → New SSH key
# 3. コピーした内容を貼り付け

# リモートURLをSSHに変更
git remote set-url origin git@github.com:YOUR_USERNAME/es12-capacitor-degradation-analysis.git

# Push
git push -u origin main
```

## ✅ 完了確認

Pushが成功したら、GitHubのリポジトリページで以下を確認：

- ✅ README.mdが表示されている
- ✅ output/large_gap_similar_vl_dissimilar_vo/ に画像とレポートがある
- ✅ .gitignoreが機能している（data/raw/*.matが除外されている）

## 📊 リポジトリの構成

```
es12-capacitor-degradation-analysis/
├── README.md                    # プロジェクト概要と結論へのリンク
├── .gitignore                   # Git除外設定
├── output/
│   ├── large_gap_similar_vl_dissimilar_vo/  # メイン分析結果
│   └── archive/                 # 参考資料
├── scripts/                     # 分析スクリプト
├── src/                         # ソースコード
├── tests/                       # テスト
└── docs/                        # ドキュメント
```

## 🔄 今後の更新

変更をPushする場合：

```bash
# 変更をステージング
git add .

# コミット
git commit -m "feat: add new analysis"

# Push
git push
```

## 📝 注意事項

- **大きなファイル**: ES12.matは.gitignoreで除外されています
- **画像ファイル**: PNGファイルは含まれます（可視化のため）
- **プライベートデータ**: 機密データがないことを確認してください

---

**作成日**: 2026-01-15
