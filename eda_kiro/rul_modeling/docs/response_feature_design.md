# VL-VO応答性特徴量の設計

## 📅 作成日: 2026-01-17

## 🎯 目的

VL（入力電圧）とVO（出力電圧）の関係性を定量化し、コンデンサの劣化を検出・予測するための特徴量を設計する。

## 📊 Task 1.1の発見事項（復習）

### 観察された劣化パターン

1. **Response Efficiency（応答効率）**: 70-85% → 1%（98.5%減少）
2. **Voltage Ratio（電圧比）**: 正の値 → 負の値（極性反転）
3. **Correlation（相関）**: 0.83 → 0.9998（波形単純化）

### 物理的解釈

- コンデンサがエネルギーを伝達できなくなる
- 出力が入力と逆位相になる（異常な応答）
- 波形が単純化（劣化により複雑な応答ができない）

## 🔧 設計する特徴量

### カテゴリ1: エネルギー伝達特徴量

#### 1.1 Response Efficiency（応答効率）

**定義**:
```python
response_efficiency = sum(VO^2) / sum(VL^2)
```

**物理的意味**:
- 入力エネルギーに対する出力エネルギーの比率
- 健全なコンデンサ: 70-85%
- 劣化したコンデンサ: < 1%

**特徴**:
- 劣化に対して非常に敏感
- 0-100%の範囲で正規化可能
- 全コンデンサで一貫した劣化パターン

#### 1.2 Voltage Ratio（電圧比）

**定義**:
```python
voltage_ratio = mean(VO) / mean(VL)
```

**物理的意味**:
- 平均的な電圧変換比
- 健全なコンデンサ: 正の値（5-15程度）
- 劣化したコンデンサ: 負の値（極性反転）

**特徴**:
- 極性反転を検出可能
- 劣化の進行を示す

#### 1.3 Peak Voltage Ratio（ピーク電圧比）

**定義**:
```python
peak_voltage_ratio = max(abs(VO)) / max(abs(VL))
```

**物理的意味**:
- ピーク電圧の変換効率
- 瞬間的な応答能力を示す

**特徴**:
- 平均値よりもノイズに強い
- 瞬間的な劣化を検出

### カテゴリ2: 波形類似度特徴量

#### 2.1 Waveform Correlation（波形相関）

**定義**:
```python
waveform_correlation = pearson_correlation(VL, VO)
```

**物理的意味**:
- VLとVOの波形の類似度
- 健全なコンデンサ: 0.8-0.85
- 劣化したコンデンサ: 0.9998（完全相関に近づく）

**特徴**:
- 波形の複雑さを反映
- 劣化により単純化（相関が上昇）

#### 2.2 Phase Coherence（位相整合性）

**定義**:
```python
# VLとVOのヒルベルト変換による位相差
phase_vl = angle(hilbert(VL))
phase_vo = angle(hilbert(VO))
phase_coherence = 1 - mean(abs(phase_vl - phase_vo)) / pi
```

**物理的意味**:
- VLとVOの位相のずれ
- 健全なコンデンサ: 位相差が小さい
- 劣化したコンデンサ: 位相差が大きい

**特徴**:
- 応答遅延を定量化
- 0-1の範囲で正規化

#### 2.3 Waveform Complexity（波形複雑度）

**定義**:
```python
# サンプルエントロピーまたはフラクタル次元
waveform_complexity = sample_entropy(VO, m=2, r=0.2*std(VO))
```

**物理的意味**:
- 波形の複雑さ・不規則性
- 健全なコンデンサ: 複雑な波形
- 劣化したコンデンサ: 単純な波形

**特徴**:
- 劣化による波形単純化を検出
- ノイズに対してロバスト

### カテゴリ3: 応答遅延特徴量

#### 3.1 Response Delay（応答遅延）

**定義**:
```python
# 相互相関のピーク位置
cross_corr = correlate(VL - mean(VL), VO - mean(VO), mode='full')
response_delay = argmax(cross_corr) - (len(VL) - 1)
```

**物理的意味**:
- VLに対するVOの時間遅れ（サンプル数）
- 健全なコンデンサ: 遅延が小さい
- 劣化したコンデンサ: 遅延が大きい

**特徴**:
- 応答速度を定量化
- 正負の値（位相の進み/遅れ）

#### 3.2 Response Time Constant（応答時定数）

**定義**:
```python
# VOの立ち上がり時間（10%-90%）
vo_normalized = (VO - min(VO)) / (max(VO) - min(VO))
t_10 = find_first(vo_normalized > 0.1)
t_90 = find_first(vo_normalized > 0.9)
response_time_constant = t_90 - t_10
```

**物理的意味**:
- 応答の速さ
- 健全なコンデンサ: 速い応答
- 劣化したコンデンサ: 遅い応答

**特徴**:
- 動的応答特性を捉える
- 時定数の変化を追跡

### カテゴリ4: 初期状態からの偏差特徴量

#### 4.1 Efficiency Degradation Rate（効率劣化率）

**定義**:
```python
# 初期サイクル（1-10）の平均を基準
initial_efficiency = mean(response_efficiency[cycles 1-10])
efficiency_degradation_rate = (initial_efficiency - current_efficiency) / initial_efficiency
```

**物理的意味**:
- 初期状態からの劣化度合い
- 0（劣化なし）から1（完全劣化）

**特徴**:
- 劣化度の直接的な指標
- 0-1の範囲で正規化済み

#### 4.2 Voltage Ratio Deviation（電圧比偏差）

**定義**:
```python
initial_voltage_ratio = mean(voltage_ratio[cycles 1-10])
voltage_ratio_deviation = abs(current_voltage_ratio - initial_voltage_ratio) / abs(initial_voltage_ratio)
```

**物理的意味**:
- 初期電圧比からの変化率
- 大きいほど劣化が進行

**特徴**:
- 相対的な変化を捉える
- コンデンサ間の個体差を吸収

#### 4.3 Correlation Shift（相関シフト）

**定義**:
```python
initial_correlation = mean(waveform_correlation[cycles 1-10])
correlation_shift = current_correlation - initial_correlation
```

**物理的意味**:
- 波形相関の変化
- 正の値: 相関が上昇（波形単純化）

**特徴**:
- 波形の質的変化を捉える
- 劣化の進行を示す

### カテゴリ5: 劣化速度特徴量

#### 5.1 Efficiency Change Rate（効率変化率）

**定義**:
```python
# 過去Nサイクルの効率変化
efficiency_change_rate = (current_efficiency - efficiency_N_cycles_ago) / N
```

**物理的意味**:
- サイクルあたりの効率変化
- 劣化の速度を示す

**特徴**:
- 劣化の加速を検出
- 負の値（効率低下）

#### 5.2 Degradation Acceleration（劣化加速度）

**定義**:
```python
# 効率変化率の変化（2次微分）
degradation_acceleration = d²(response_efficiency) / dt²
```

**物理的意味**:
- 劣化速度の変化
- 急激な劣化を検出

**特徴**:
- 故障の予兆を早期検出
- 正の値: 劣化が加速

### カテゴリ6: 統計的特徴量

#### 6.1 VO Variability（VO変動性）

**定義**:
```python
vo_variability = std(VO) / mean(abs(VO))
```

**物理的意味**:
- 出力電圧の変動係数
- 健全なコンデンサ: 安定した出力
- 劣化したコンデンサ: 不安定な出力

**特徴**:
- 出力の安定性を評価
- ノイズや異常を検出

#### 6.2 VL-VO Residual Energy（残差エネルギー）

**定義**:
```python
# VLからVOを線形予測した残差
vo_predicted = linear_fit(VL) * VL
residual = VO - vo_predicted
residual_energy = sum(residual^2) / sum(VO^2)
```

**物理的意味**:
- 線形関係からのずれ
- 非線形な劣化を検出

**特徴**:
- 複雑な劣化パターンを捉える
- 0-1の範囲

## 📋 特徴量サマリー

### 優先度別リスト

#### 高優先度（必須）

1. **Response Efficiency** - 劣化の主要指標
2. **Efficiency Degradation Rate** - 劣化度の直接指標
3. **Voltage Ratio** - 極性反転を検出
4. **Waveform Correlation** - 波形単純化を検出
5. **Response Delay** - 応答遅延を定量化

#### 中優先度（推奨）

6. **Peak Voltage Ratio** - ピーク応答
7. **Voltage Ratio Deviation** - 初期状態からの変化
8. **Correlation Shift** - 相関の変化
9. **Efficiency Change Rate** - 劣化速度
10. **VO Variability** - 出力安定性

#### 低優先度（オプション）

11. **Phase Coherence** - 位相整合性
12. **Waveform Complexity** - 波形複雑度
13. **Response Time Constant** - 応答時定数
14. **Degradation Acceleration** - 劣化加速度
15. **VL-VO Residual Energy** - 残差エネルギー

## 🔧 実装方針

### Phase 1: 基本特徴量（高優先度）

```python
class ResponseFeatureExtractor:
    def extract_basic_features(self, vl, vo, cycle, initial_stats):
        features = {}
        
        # エネルギー伝達
        features['response_efficiency'] = self._calc_efficiency(vl, vo)
        features['voltage_ratio'] = np.mean(vo) / np.mean(vl)
        features['peak_voltage_ratio'] = np.max(np.abs(vo)) / np.max(np.abs(vl))
        
        # 波形類似度
        features['waveform_correlation'] = np.corrcoef(vl, vo)[0, 1]
        
        # 応答遅延
        features['response_delay'] = self._calc_delay(vl, vo)
        
        # 初期状態からの偏差
        features['efficiency_degradation_rate'] = self._calc_degradation_rate(
            features['response_efficiency'], 
            initial_stats['initial_efficiency']
        )
        features['voltage_ratio_deviation'] = self._calc_deviation(
            features['voltage_ratio'],
            initial_stats['initial_voltage_ratio']
        )
        features['correlation_shift'] = (
            features['waveform_correlation'] - 
            initial_stats['initial_correlation']
        )
        
        return features
```

### Phase 2: 拡張特徴量（中・低優先度）

- 時系列特徴量（過去Nサイクルの統計）
- 高度な信号処理特徴量（ヒルベルト変換、エントロピー等）
- 劣化速度・加速度

## 📊 期待される効果

### 異常検知

- **Normal定義**: Response Efficiency > 50%
- **Abnormal検出**: Efficiency < 50% または Voltage Ratio < 0

### 劣化予測

- **劣化度スコア**: Efficiency Degradation Rate（0-1）
- **予測対象**: 次サイクルのResponse Efficiency

### 特徴量重要度（予想）

1. Response Efficiency（最重要）
2. Efficiency Degradation Rate
3. Voltage Ratio
4. Waveform Correlation
5. Response Delay

## 🚀 次のステップ

1. **Task 1.3**: ResponseFeatureExtractorの実装
2. 全サイクルからの特徴量抽出
3. 特徴量の相関分析
4. 特徴量選択（重要度評価）

---

**作成者**: Kiro AI Agent  
**ステータス**: Task 1.2 Complete - Feature Design  
**次のタスク**: Task 1.3 - Feature Extraction Implementation
