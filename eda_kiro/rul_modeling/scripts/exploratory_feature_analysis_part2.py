"""
Phase 0: 探索的特徴量分析 - Part 2（相関分析）
"""

def analyze_correlations(df_full):
    """特徴量とRULの相関分析"""
    print("\n[Step 4] 特徴量とRULの相関分析...")
    
    # 数値特徴量のみ選択
    numeric_features = df_full.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features.remove('rul')  # RUL自体は除外
    numeric_features.remove('capacitor_id') if 'capacitor_id' in numeric_features else None
    
    # 相関係数を計算
    correlations = {}
    for feature in numeric_features:
        corr, p_value = pearsonr(df_full[feature], df_full['rul'])
        correlations[feature] = {
            'correlation': corr,
            'p_value': p_value,
            'abs_correlation': abs(corr)
        }
    
    # DataFrameに変換してソート
    corr_df = pd.DataFrame(correlations).T
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    
    print("\n特徴量とRULの相関係数（絶対値の降順）:")
    print(corr_df[['correlation', 'p_value']].head(15))
    
    # 高相関特徴量（|r| > 0.5）
    high_corr_features = corr_df[corr_df['abs_correlation'] > 0.5].index.tolist()
    print(f"\n高相関特徴量（|r| > 0.5）: {len(high_corr_features)}個")
    print(high_corr_features)
    
    # 低相関特徴量（|r| < 0.1）
    low_corr_features = corr_df[corr_df['abs_correlation'] < 0.1].index.tolist()
    print(f"\n低相関特徴量（|r| < 0.1）: {len(low_corr_features)}個")
    print(low_corr_features)
    
    return corr_df, high_corr_features, low_corr_features


def visualize_feature_trends(df_full, top_features):
    """特徴量のトレンドを可視化"""
    print("\n[Step 5] 特徴量のトレンドを可視化...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features[:6]):
        ax = axes[idx]
        ax.plot(df_full['cycle_number'], df_full[feature], marker='o', markersize=2)
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel(feature)
        ax.set_title(f'{feature} vs Cycle Number')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent.parent / "output" / "feature_trends.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"保存: {output_path}")
    plt.close()


def analyze_multiple_capacitors(loader, cap_ids):
    """複数コンデンサでの一貫性確認"""
    print("\n[Step 6] 複数コンデンサでの一貫性確認...")
    
    all_correlations = {}
    
    for cap_id in cap_ids:
        print(f"  処理中: {cap_id}...")
        features_list = []
        
        for cycle in range(1, 201):
            vl, vo = loader.get_cycle_data(cap_id, cycle)
            features = extract_basic_features(vl, vo, cycle)
            features['capacitor_id'] = cap_id
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        df['rul'] = 200 - df['cycle_number']
        
        # 相関係数を計算
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features.remove('rul')
        
        correlations = {}
        for feature in numeric_features:
            corr, _ = pearsonr(df[feature], df['rul'])
            correlations[feature] = corr
        
        all_correlations[cap_id] = correlations
    
    # DataFrameに変換
    corr_matrix = pd.DataFrame(all_correlations).T
    
    print("\n各コンデンサでの相関係数:")
    print(corr_matrix.head())
    
    # 平均相関係数
    mean_corr = corr_matrix.mean().sort_values(ascending=False, key=abs)
    print("\n平均相関係数（絶対値の降順）:")
    print(mean_corr.head(10))
    
    # 一貫性の高い特徴量（全コンデンサで|r| > 0.3）
    consistent_features = []
    for feature in corr_matrix.columns:
        if (corr_matrix[feature].abs() > 0.3).all():
            consistent_features.append(feature)
    
    print(f"\n一貫性の高い特徴量（全コンデンサで|r| > 0.3）: {len(consistent_features)}個")
    print(consistent_features)
    
    return corr_matrix, consistent_features


if __name__ == "__main__":
    # Part 1のコードを実行してdf_fullを取得
    # ここでは省略
    pass
