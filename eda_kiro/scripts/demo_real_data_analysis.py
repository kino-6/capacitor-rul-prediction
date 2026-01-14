#!/usr/bin/env python3
"""
Demo script for real data analysis capabilities.

This script demonstrates the comprehensive real data analysis functionality
including specialized ES12 data loading, enhanced analysis, validation, and reporting.
"""

import sys
from pathlib import Path
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from nasa_pcoe_eda.real_data_orchestrator import RealDataOrchestrator
from nasa_pcoe_eda.exceptions import AnalysisError, DataLoadError


def main():
    """Main demonstration function."""
    print("🔬 NASA PCOE ES12 実データ分析デモ")
    print("=" * 50)
    
    # Setup paths
    data_path = Path("data/raw/ES12.mat")
    output_dir = Path("output/real_data_demo")
    
    # Check if data file exists
    if not data_path.exists():
        print(f"❌ データファイルが見つかりません: {data_path}")
        print("   ES12.matファイルをdata/raw/ディレクトリに配置してください")
        return
    
    try:
        # Initialize orchestrator
        print("🚀 実データ分析オーケストレーターを初期化中...")
        orchestrator = RealDataOrchestrator(output_dir=output_dir)
        
        # Run comprehensive analysis
        print("📊 包括的実データ分析を実行中...")
        print("   - ES12データの読み込みと検証")
        print("   - 実データ特有のメトリクス抽出")
        print("   - コア分析（統計、相関、外れ値、時系列）")
        print("   - 劣化パターン分析")
        print("   - 個別コンデンサ分析")
        print("   - 分析手法妥当性検証")
        print("   - 拡張可視化生成")
        print("   - 包括的レポート生成")
        
        results = orchestrator.run_comprehensive_analysis(
            data_path=data_path,
            generate_visualizations=True,
            generate_report=True
        )
        
        # Display summary
        print("\n✅ 分析完了！")
        print("=" * 50)
        
        summary = orchestrator.get_analysis_summary()
        
        # Data overview
        if 'data_overview' in summary:
            overview = summary['data_overview']
            print("📋 データ概要:")
            print(f"   - レコード数: {overview.get('records', 'N/A'):,}")
            print(f"   - 特徴量数: {overview.get('features', 'N/A')}")
            print(f"   - コンデンサ数: {overview.get('capacitors', 'N/A')}")
            print(f"   - 測定サイクル数: {overview.get('cycles', 'N/A')}")
        
        # Key findings
        if 'key_findings' in summary:
            findings = summary['key_findings']
            print("\n🔍 主要な発見:")
            print(f"   - データ品質スコア: {findings.get('data_quality_score', 0):.1%}")
            print(f"   - 検出された劣化パターン数: {findings.get('degradation_patterns_detected', 0)}")
            print(f"   - 手法信頼性レベル: {findings.get('methodology_reliability', 'Unknown')}")
        
        # Quality assessment
        if 'quality_assessment' in summary:
            quality = summary['quality_assessment']
            print("\n📈 品質評価:")
            print(f"   - データ完全性: {quality.get('data_completeness', 0):.1%}")
            print(f"   - 測定精度: {quality.get('measurement_precision', 0):.3f}")
            if quality.get('signal_noise_ratio'):
                print(f"   - S/N比: {quality.get('signal_noise_ratio', 0):.1f} dB")
        
        # Recommendations
        if 'recommendations' in summary:
            recommendations = summary['recommendations']
            print("\n💡 推奨事項:")
            for i, rec in enumerate(recommendations[:5], 1):  # Show first 5
                print(f"   {i}. {rec}")
            if len(recommendations) > 5:
                print(f"   ... 他 {len(recommendations) - 5} 項目")
        
        # Output files
        print(f"\n📁 出力ファイル:")
        print(f"   - 分析ログ: {output_dir / 'real_data_analysis.log'}")
        print(f"   - 包括的レポート: {output_dir / 'real_data_analysis_report.html'}")
        print(f"   - 可視化ファイル: {output_dir / 'visualizations/'}")
        
        print("\n🎉 実データ分析デモが正常に完了しました！")
        print(f"   詳細な結果は {output_dir} ディレクトリをご確認ください。")
        
    except DataLoadError as e:
        print(f"❌ データ読み込みエラー: {e}")
        print("   ES12.matファイルの形式または内容を確認してください")
        
    except AnalysisError as e:
        print(f"❌ 分析エラー: {e}")
        print("   分析パラメータまたはデータ品質を確認してください")
        
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        print("   詳細なエラー情報については分析ログを確認してください")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    main()