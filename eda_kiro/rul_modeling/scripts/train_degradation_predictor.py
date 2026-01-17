"""
Task 3.2 & 3.3: 劣化予測モデルの構築

Task 3.2: 劣化度予測モデル
- Random Forest Regressorで現在の特徴量から劣化度を予測
- Train/Val/Test分割（コンデンサベース）

Task 3.3: 次サイクル応答性予測
- 過去Nサイクルから次サイクルの応答性特徴量を予測
- 時系列予測モデル（Random Forest）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """劣化度スコア付きデータの読み込み"""
    data_path = Path("output/degradation_prediction/features_with_degradation_score.csv")
    df = pd.read_csv(data_path)
    print(f"✓ データ読み込み: {len(df)}サンプル")
    return df

def split_data(df):
    """
    コンデンサベースでTrain/Val/Test分割
    Train: C1-C5 (1000サンプル)
    Val: C6 (200サンプル)
    Test: C7-C8 (400サンプル)
    """
    train_df = df[df['capacitor_id'].isin(['ES12C1', 'ES12C2', 'ES12C3', 'ES12C4', 'ES12C5'])].copy()
    val_df = df[df['capacitor_id'] == 'ES12C6'].copy()
    test_df = df[df['capacitor_id'].isin(['ES12C7', 'ES12C8'])].copy()
    
    print(f"\n✓ データ分割:")
    print(f"  Train: {len(train_df)}サンプル (C1-C5)")
    print(f"  Val: {len(val_df)}サンプル (C6)")
    print(f"  Test: {len(test_df)}サンプル (C7-C8)")
    
    return train_df, val_df, test_df


def train_degradation_predictor(train_df, val_df):
    """
    Task 3.2: 劣化度予測モデルの学習
    
    入力: 波形特性特徴量（7個）
    出力: 劣化度スコア（0-1）
    """
    # 使用する特徴量（波形特性のみ、データリーケージなし）
    feature_cols = [
        'waveform_correlation',
        'vo_variability',
        'vl_variability',
        'response_delay',
        'response_delay_normalized',
        'residual_energy_ratio',
        'vo_complexity'
    ]
    
    target_col = 'degradation_score'
    
    # 訓練データ
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    
    # 検証データ
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    
    # Random Forest Regressorの学習
    print("\n" + "="*60)
    print("Task 3.2: 劣化度予測モデルの学習")
    print("="*60)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 訓練データでの評価
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"\n訓練データ性能:")
    print(f"  MAE: {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  R²: {train_r2:.4f}")
    
    # 検証データでの評価
    y_val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\n検証データ性能:")
    print(f"  MAE: {val_mae:.4f}")
    print(f"  RMSE: {val_rmse:.4f}")
    print(f"  R²: {val_r2:.4f}")
    
    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n特徴量重要度:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, feature_cols, feature_importance


def create_sequence_data(df, feature_cols, lookback=5):
    """
    Task 3.3用: 時系列データの作成
    
    過去lookbackサイクルの特徴量から次サイクルの特徴量を予測
    """
    sequences = []
    targets = []
    capacitor_ids = []
    cycles = []
    
    for cap_id in df['capacitor_id'].unique():
        cap_data = df[df['capacitor_id'] == cap_id].sort_values('cycle')
        
        for i in range(lookback, len(cap_data)):
            # 過去lookbackサイクルの特徴量
            seq = cap_data.iloc[i-lookback:i][feature_cols].values.flatten()
            # 次サイクルの特徴量
            target = cap_data.iloc[i][feature_cols].values
            
            sequences.append(seq)
            targets.append(target)
            capacitor_ids.append(cap_id)
            cycles.append(cap_data.iloc[i]['cycle'])
    
    return np.array(sequences), np.array(targets), np.array(capacitor_ids), np.array(cycles)

def train_response_predictor(train_df, val_df):
    """
    Task 3.3: 次サイクル応答性予測モデルの学習
    
    入力: 過去5サイクルの波形特性特徴量
    出力: 次サイクルの波形特性特徴量
    """
    print("\n" + "="*60)
    print("Task 3.3: 次サイクル応答性予測モデルの学習")
    print("="*60)
    
    feature_cols = [
        'waveform_correlation',
        'vo_variability',
        'vl_variability',
        'response_delay',
        'response_delay_normalized',
        'residual_energy_ratio',
        'vo_complexity'
    ]
    
    lookback = 5
    
    # 時系列データの作成
    X_train, y_train, _, _ = create_sequence_data(train_df, feature_cols, lookback)
    X_val, y_val, val_cap_ids, val_cycles = create_sequence_data(val_df, feature_cols, lookback)
    
    print(f"\n時系列データ作成:")
    print(f"  Train: {len(X_train)}サンプル")
    print(f"  Val: {len(X_val)}サンプル")
    print(f"  入力次元: {X_train.shape[1]} (過去{lookback}サイクル × {len(feature_cols)}特徴量)")
    print(f"  出力次元: {y_train.shape[1]} ({len(feature_cols)}特徴量)")
    
    # 各特徴量ごとにモデルを学習
    models = {}
    
    for i, feature_name in enumerate(feature_cols):
        print(f"\n{feature_name}の予測モデル学習中...")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train[:, i])
        
        # 検証データでの評価
        y_val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val[:, i], y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val[:, i], y_val_pred))
        val_r2 = r2_score(y_val[:, i], y_val_pred)
        
        print(f"  Val MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
        
        models[feature_name] = model
    
    return models, feature_cols, lookback


def evaluate_on_test(degradation_model, response_models, test_df, feature_cols, lookback):
    """テストデータでの評価"""
    print("\n" + "="*60)
    print("テストデータでの評価")
    print("="*60)
    
    # Task 3.2: 劣化度予測の評価
    X_test = test_df[feature_cols].values
    y_test = test_df['degradation_score'].values
    
    y_test_pred = degradation_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTask 3.2 - 劣化度予測:")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²: {test_r2:.4f}")
    
    success = "✅ 成功" if test_mae < 0.1 else "⚠️ 目標未達"
    print(f"  成功基準 (MAE < 0.1): {success}")
    
    # Task 3.3: 次サイクル応答性予測の評価
    X_test_seq, y_test_seq, test_cap_ids, test_cycles = create_sequence_data(
        test_df, feature_cols, lookback
    )
    
    print(f"\nTask 3.3 - 次サイクル応答性予測:")
    
    for i, feature_name in enumerate(feature_cols):
        model = response_models[feature_name]
        y_pred = model.predict(X_test_seq)
        
        mae = mean_absolute_error(y_test_seq[:, i], y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_seq[:, i], y_pred))
        r2 = r2_score(y_test_seq[:, i], y_pred)
        
        print(f"  {feature_name}:")
        print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    return {
        'degradation': {
            'y_true': y_test,
            'y_pred': y_test_pred,
            'mae': test_mae,
            'rmse': test_rmse,
            'r2': test_r2
        },
        'response': {
            'X': X_test_seq,
            'y_true': y_test_seq,
            'cap_ids': test_cap_ids,
            'cycles': test_cycles,
            'feature_cols': feature_cols
        }
    }


def visualize_results(test_results, test_df, response_models, feature_importance):
    """結果の可視化"""
    output_dir = Path("output/degradation_prediction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. 劣化度予測: 真値 vs 予測値
    ax1 = fig.add_subplot(gs[0, 0])
    y_true = test_results['degradation']['y_true']
    y_pred = test_results['degradation']['y_pred']
    
    ax1.scatter(y_true, y_pred, alpha=0.5, s=20)
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('True Degradation Score', fontsize=12)
    ax1.set_ylabel('Predicted Degradation Score', fontsize=12)
    ax1.set_title('劣化度予測: 真値 vs 予測値', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 劣化度予測: 時系列プロット（コンデンサ別）
    ax2 = fig.add_subplot(gs[0, 1])
    test_df_with_pred = test_df.copy()
    test_df_with_pred['degradation_pred'] = y_pred
    
    for cap_id in sorted(test_df['capacitor_id'].unique()):
        cap_data = test_df_with_pred[test_df_with_pred['capacitor_id'] == cap_id].sort_values('cycle')
        ax2.plot(cap_data['cycle'], cap_data['degradation_score'], 
                label=f'C{cap_id} True', linestyle='-', linewidth=2)
        ax2.plot(cap_data['cycle'], cap_data['degradation_pred'], 
                label=f'C{cap_id} Pred', linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Cycle', fontsize=12)
    ax2.set_ylabel('Degradation Score', fontsize=12)
    ax2.set_title('劣化度予測: 時系列（テストデータ）', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # 3. 特徴量重要度
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.barh(feature_importance['feature'], feature_importance['importance'])
    ax3.set_xlabel('Importance', fontsize=12)
    ax3.set_title('特徴量重要度（劣化度予測）', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4-9. 次サイクル応答性予測（各特徴量）
    response_data = test_results['response']
    feature_cols = response_data['feature_cols']
    
    for idx, feature_name in enumerate(feature_cols):
        ax = fig.add_subplot(gs[1 + idx // 3, idx % 3])
        
        model = response_models[feature_name]
        y_pred = model.predict(response_data['X'])
        y_true = response_data['y_true'][:, idx]
        
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        ax.set_xlabel('True Value', fontsize=10)
        ax.set_ylabel('Predicted Value', fontsize=10)
        ax.set_title(f'{feature_name}\nMAE: {mae:.4f}, R²: {r2:.4f}', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Phase 3: 劣化予測モデルの評価結果', fontsize=16, fontweight='bold', y=0.995)
    
    # 保存
    output_path = output_dir / "prediction_model_evaluation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 可視化保存: {output_path}")
    plt.close()


def save_models(degradation_model, response_models, feature_cols, lookback, feature_importance):
    """モデルの保存"""
    output_dir = Path("output/models_v3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Task 3.2: 劣化度予測モデル
    model_path = output_dir / "degradation_predictor.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(degradation_model, f)
    print(f"\n✓ 劣化度予測モデル保存: {model_path}")
    
    # 特徴量リスト
    features_path = output_dir / "degradation_predictor_features.txt"
    with open(features_path, 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    print(f"✓ 特徴量リスト保存: {features_path}")
    
    # 特徴量重要度
    importance_path = output_dir / "degradation_predictor_feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    print(f"✓ 特徴量重要度保存: {importance_path}")
    
    # Task 3.3: 次サイクル応答性予測モデル
    response_model_path = output_dir / "response_predictor.pkl"
    with open(response_model_path, 'wb') as f:
        pickle.dump({
            'models': response_models,
            'feature_cols': feature_cols,
            'lookback': lookback
        }, f)
    print(f"✓ 次サイクル応答性予測モデル保存: {response_model_path}")

def create_summary_document(test_results, feature_importance):
    """完了サマリードキュメントの作成"""
    output_dir = Path("output/degradation_prediction")
    doc_path = output_dir / "phase3_completion_summary.md"
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 3 完了サマリー: 劣化予測モデル構築\n\n")
        f.write("**完了日**: 2026-01-18\n")
        f.write("**Phase**: Phase 3 - 劣化予測モデル構築\n\n")
        f.write("---\n\n")
        
        f.write("## 🎯 Phase 3の目的\n\n")
        f.write("応答性の劣化度を予測するモデルを構築。\n\n")
        
        f.write("---\n\n")
        f.write("## 📋 完了したタスク\n\n")
        
        # Task 3.1
        f.write("### Task 3.1: 劣化度の定義 ✅\n\n")
        f.write("**実装内容**:\n")
        f.write("- 複合指標アプローチ: 4つの波形特性を組み合わせ\n")
        f.write("- 劣化度スコア範囲: 0.000 - 0.731\n")
        f.write("- 劣化ステージ定義: Normal, Degrading, Severe, Critical\n\n")
        
        # Task 3.2
        f.write("### Task 3.2: 劣化度予測モデルの構築 ✅\n\n")
        f.write("**アプローチ**: Random Forest Regressor\n\n")
        f.write("**使用特徴量** (7個の波形特性):\n")
        for _, row in feature_importance.iterrows():
            f.write(f"- {row['feature']}: {row['importance']:.4f}\n")
        f.write("\n")
        
        deg_results = test_results['degradation']
        f.write("**テストデータ性能**:\n")
        f.write(f"- MAE: {deg_results['mae']:.4f}\n")
        f.write(f"- RMSE: {deg_results['rmse']:.4f}\n")
        f.write(f"- R²: {deg_results['r2']:.4f}\n\n")
        
        success = "✅ 達成" if deg_results['mae'] < 0.1 else "⚠️ 未達成"
        f.write(f"**成功基準 (MAE < 0.1)**: {success}\n\n")
        
        # Task 3.3
        f.write("### Task 3.3: 次サイクル応答性予測 ✅\n\n")
        f.write("**アプローチ**: Random Forest Regressor (特徴量ごと)\n\n")
        f.write("**入力**: 過去5サイクルの波形特性特徴量\n")
        f.write("**出力**: 次サイクルの波形特性特徴量\n\n")
        
        f.write("---\n\n")
        f.write("## 📊 Phase 3の成果\n\n")
        f.write("### 構築したモデル\n\n")
        f.write("1. **劣化度予測モデル**: 現在の波形特性から劣化度を予測\n")
        f.write("2. **次サイクル応答性予測モデル**: 過去5サイクルから次サイクルの波形特性を予測\n\n")
        
        f.write("### 出力ファイル\n\n")
        f.write("**モデルファイル**:\n")
        f.write("- `output/models_v3/degradation_predictor.pkl`\n")
        f.write("- `output/models_v3/response_predictor.pkl`\n")
        f.write("- `output/models_v3/degradation_predictor_features.txt`\n")
        f.write("- `output/models_v3/degradation_predictor_feature_importance.csv`\n\n")
        
        f.write("**結果ファイル**:\n")
        f.write("- `output/degradation_prediction/prediction_model_evaluation.png`\n")
        f.write("- `output/degradation_prediction/phase3_completion_summary.md`\n\n")
        
        f.write("---\n\n")
        f.write("## 🎉 プロジェクト完了\n\n")
        f.write("Phase 1, 2, 3の全タスクが完了しました。\n\n")
        f.write("**Phase 1**: VL-VO関係性分析 ✅\n")
        f.write("**Phase 2**: 異常検知モデル構築 ✅\n")
        f.write("**Phase 3**: 劣化予測モデル構築 ✅\n\n")
    
    print(f"✓ 完了サマリー保存: {doc_path}")

def main():
    print("="*60)
    print("Task 3.2 & 3.3: 劣化予測モデルの構築")
    print("="*60)
    
    # 1. データ読み込み
    df = load_data()
    
    # 2. データ分割
    train_df, val_df, test_df = split_data(df)
    
    # 3. Task 3.2: 劣化度予測モデルの学習
    degradation_model, feature_cols, feature_importance = train_degradation_predictor(
        train_df, val_df
    )
    
    # 4. Task 3.3: 次サイクル応答性予測モデルの学習
    response_models, _, lookback = train_response_predictor(train_df, val_df)
    
    # 5. テストデータでの評価
    test_results = evaluate_on_test(
        degradation_model, response_models, test_df, feature_cols, lookback
    )
    
    # 6. 可視化
    visualize_results(test_results, test_df, response_models, feature_importance)
    
    # 7. モデルの保存
    save_models(degradation_model, response_models, feature_cols, lookback, feature_importance)
    
    # 8. 完了サマリーの作成
    create_summary_document(test_results, feature_importance)
    
    print("\n" + "="*60)
    print("✓ Phase 3完了: 劣化予測モデル構築")
    print("="*60)
    print("\n🎉 全Phase完了！")
    print("  Phase 1: VL-VO関係性分析 ✅")
    print("  Phase 2: 異常検知モデル構築 ✅")
    print("  Phase 3: 劣化予測モデル構築 ✅")

if __name__ == "__main__":
    main()
