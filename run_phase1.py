"""
Main Orchestration Script for Phase 1
Runs the complete knowledge graph construction and evaluation pipeline
"""
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add phase1_kg to path
sys.path.append(str(Path(__file__).parent / "phase1_kg"))

from phase1_kg.config import *
from phase1_kg.utils.helpers import log_step

# Import all modules
from phase1_kg.data_collection.fundamental_data import FundamentalDataCollector
from phase1_kg.data_collection.macro_data import MacroDataCollector
from phase1_kg.data_collection.sentiment_analysis import SentimentAnalyzer
from phase1_kg.preprocessing.feature_engineering import FeatureEngineer
from phase1_kg.knowledge_graphs.temporal_kg import TemporalKG
from phase1_kg.knowledge_graphs.causal_kg import CausalKG
from phase1_kg.knowledge_graphs.kg_integration import IntegratedKG
from phase1_kg.evaluation.intrinsic_metrics import KGEvaluator
from phase1_kg.visualization.kg_viz import KGVisualizer
from phase1_kg.evaluation.report_generator import ReportGenerator


def run_data_collection(args):
    """Step 1: Collect all required data"""
    log_step("STEP 1: DATA COLLECTION")

    # 1.1 Fundamental Data
    if not args.skip_fundamentals:
        print("\n[1.1] Collecting Fundamental Data...")
        print("This will fetch real data from Yahoo Finance using ISIN-ticker mapping")
        collector = FundamentalDataCollector()
        from phase1_kg.utils.helpers import load_master_data
        master_df = load_master_data(MASTER_CSV_PATH)
        df_fundamentals = collector.collect_fundamentals_for_stocks(master_df)

        if df_fundamentals.empty:
            print("❌ Failed to collect fundamental data. Check:")
            print("   1. ISIN-ticker mapping exists (run: python create_isin_ticker_mapping.py)")
            print("   2. Internet connection is available")
            print("   3. Yahoo Finance is accessible")
        else:
            output_path = PROCESSED_DATA_DIR / "fundamental_data.csv"
            df_fundamentals.to_csv(output_path, index=False)
            print(f"✓ Fundamentals saved: {output_path}")
            print(f"  Collected data for {df_fundamentals['ISIN'].nunique()} stocks")
    else:
        print("\n[1.1] Skipping fundamental data collection")

    # 1.2 Macroeconomic Data
    if not args.skip_macro:
        print("\n[1.2] Collecting Macroeconomic Data...")
        print("This will fetch real data from Yahoo Finance for Indian market indices")
        collector = MacroDataCollector()
        df_macro = collector.collect_all_indicators(START_DATE, END_DATE)

        if df_macro.empty:
            print("❌ Failed to collect macroeconomic data. Check:")
            print("   1. Internet connection is available")
            print("   2. Yahoo Finance is accessible")
            print("   3. Ticker symbols in config.py are correct")
        else:
            output_path = PROCESSED_DATA_DIR / "macro_data.csv"
            df_macro.to_csv(output_path, index=False)
            print(f"✓ Macro data saved: {output_path}")
            print(f"  Collected {len(df_macro.columns)-1} indicators")
    else:
        print("\n[1.2] Skipping macroeconomic data collection")

    # 1.3 Sentiment Analysis
    if not args.skip_sentiment:
        print("\n[1.3] Collecting Sentiment Data...")
        print("Note: This may take significant time. Using sector-level aggregation.")
        analyzer = SentimentAnalyzer()
        df_sentiment = analyzer.collect_historical_sentiment(
            sectors=STANDARD_SECTORS,
            start_date=START_DATE,
            end_date=END_DATE,
            frequency='M'
        )

        if df_sentiment.empty:
            print("⚠️  Sentiment data collection returned no results")
        else:
            output_path = PROCESSED_DATA_DIR / "sentiment_data.csv"
            df_sentiment.to_csv(output_path, index=False)
            print(f"✓ Sentiment data saved: {output_path}")
    else:
        print("\n[1.3] Skipping sentiment data collection")

    print("\n✓ Data collection complete!")


def run_feature_engineering(args):
    """Step 2: Feature engineering and data integration"""
    log_step("STEP 2: FEATURE ENGINEERING")

    engineer = FeatureEngineer()
    df_integrated = engineer.create_integrated_dataset()

    print(f"✓ Integrated dataset created: {df_integrated.shape}")
    print(f"  Saved to: {PROCESSED_DATA_DIR / 'integrated_dataset.csv'}")


def run_kg_construction(args):
    """Step 3: Build knowledge graphs"""
    log_step("STEP 3: KNOWLEDGE GRAPH CONSTRUCTION")

    # Load integrated dataset
    import pandas as pd
    df = pd.read_csv(PROCESSED_DATA_DIR / "integrated_dataset.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    # 3.1 Temporal KG
    if not args.skip_temporal:
        print("\n[3.1] Building Temporal Knowledge Graph...")
        tkg = TemporalKG()
        tkg.build_from_dataframe(df)
        tkg.save()
        print("✓ Temporal KG constructed and saved")
    else:
        print("\n[3.1] Skipping Temporal KG construction")

    # 3.2 Causal KG
    if not args.skip_causal:
        print("\n[3.2] Building Causal Knowledge Graph...")
        ckg = CausalKG()
        ckg.build_from_dataframe(df)
        ckg.save()
        print("✓ Causal KG constructed and saved")
    else:
        print("\n[3.2] Skipping Causal KG construction")

    # 3.3 Integration
    if not args.skip_integration:
        print("\n[3.3] Integrating Knowledge Graphs...")
        ikg = IntegratedKG.load_and_integrate()
        ikg.save()
        print("✓ Integrated KG constructed and saved")
    else:
        print("\n[3.3] Skipping KG integration")


def run_evaluation(args):
    """Step 4: Evaluate knowledge graphs"""
    log_step("STEP 4: KNOWLEDGE GRAPH EVALUATION")

    import networkx as nx

    # 4.1 Evaluate Temporal KG
    if not args.skip_temporal:
        print("\n[4.1] Evaluating Temporal KG...")
        temporal_kg = nx.read_gpickle(GRAPHS_DIR / "temporal_kg.gpickle")
        evaluator = KGEvaluator(temporal_kg, "Temporal_KG")
        evaluator.evaluate_all()
        evaluator.save_results()
        print("✓ Temporal KG evaluation complete")
    else:
        print("\n[4.1] Skipping Temporal KG evaluation")

    # 4.2 Evaluate Causal KG
    if not args.skip_causal:
        print("\n[4.2] Evaluating Causal KG...")
        causal_kg = nx.read_gpickle(GRAPHS_DIR / "causal_kg.gpickle")
        evaluator = KGEvaluator(causal_kg, "Causal_KG")
        evaluator.evaluate_all()
        evaluator.save_results()
        print("✓ Causal KG evaluation complete")
    else:
        print("\n[4.2] Skipping Causal KG evaluation")


def run_visualization(args):
    """Step 5: Generate visualizations"""
    log_step("STEP 5: VISUALIZATION")

    import networkx as nx

    # 5.1 Visualize Temporal KG
    if not args.skip_temporal and not args.skip_viz:
        print("\n[5.1] Visualizing Temporal KG...")
        temporal_kg = nx.read_gpickle(GRAPHS_DIR / "temporal_kg.gpickle")
        viz = KGVisualizer(temporal_kg, "Temporal_KG")
        viz.visualize_all()
        print("✓ Temporal KG visualizations complete")
    else:
        print("\n[5.1] Skipping Temporal KG visualization")

    # 5.2 Visualize Causal KG
    if not args.skip_causal and not args.skip_viz:
        print("\n[5.2] Visualizing Causal KG...")
        causal_kg = nx.read_gpickle(GRAPHS_DIR / "causal_kg.gpickle")
        viz = KGVisualizer(causal_kg, "Causal_KG")
        viz.visualize_all()
        print("✓ Causal KG visualizations complete")
    else:
        print("\n[5.2] Skipping Causal KG visualization")


def run_report_generation(args):
    """Step 6: Generate final report"""
    log_step("STEP 6: REPORT GENERATION")

    generator = ReportGenerator()
    generator.generate_full_report()

    print("\n✓ Report generation complete!")


def main():
    """Main execution pipeline"""
    parser = argparse.ArgumentParser(
        description="Run Phase 1: Knowledge Graph Construction and Evaluation"
    )

    # Add arguments for selective execution
    parser.add_argument('--skip-data-collection', action='store_true',
                       help='Skip data collection step')
    parser.add_argument('--skip-fundamentals', action='store_true',
                       help='Skip fundamental data collection')
    parser.add_argument('--skip-macro', action='store_true',
                       help='Skip macroeconomic data collection')
    parser.add_argument('--skip-sentiment', action='store_true',
                       help='Skip sentiment analysis')
    parser.add_argument('--skip-feature-engineering', action='store_true',
                       help='Skip feature engineering')
    parser.add_argument('--skip-temporal', action='store_true',
                       help='Skip temporal KG construction/evaluation')
    parser.add_argument('--skip-causal', action='store_true',
                       help='Skip causal KG construction/evaluation')
    parser.add_argument('--skip-integration', action='store_true',
                       help='Skip KG integration')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation step')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization step')
    parser.add_argument('--skip-report', action='store_true',
                       help='Skip report generation')

    parser.add_argument('--only-step', type=str,
                       choices=['data', 'features', 'kg', 'eval', 'viz', 'report'],
                       help='Run only a specific step')

    args = parser.parse_args()

    print("="*80)
    print("PHASE 1: KNOWLEDGE GRAPH CONSTRUCTION AND EVALUATION")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {BASE_DIR}")
    print("="*80)

    try:
        # Run specific step if requested
        if args.only_step:
            if args.only_step == 'data':
                run_data_collection(args)
            elif args.only_step == 'features':
                run_feature_engineering(args)
            elif args.only_step == 'kg':
                run_kg_construction(args)
            elif args.only_step == 'eval':
                run_evaluation(args)
            elif args.only_step == 'viz':
                run_visualization(args)
            elif args.only_step == 'report':
                run_report_generation(args)
        else:
            # Run full pipeline
            if not args.skip_data_collection:
                run_data_collection(args)

            if not args.skip_feature_engineering:
                run_feature_engineering(args)

            run_kg_construction(args)

            if not args.skip_evaluation:
                run_evaluation(args)

            if not args.skip_viz:
                run_visualization(args)

            if not args.skip_report:
                run_report_generation(args)

        print("\n" + "="*80)
        print("PHASE 1 PIPELINE COMPLETE!")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nOutputs saved in:")
        print(f"  - Graphs: {GRAPHS_DIR}")
        print(f"  - Metrics: {METRICS_DIR}")
        print(f"  - Figures: {FIGURES_DIR}")
        print(f"  - Reports: {REPORTS_DIR}")
        print("\nNext steps:")
        print("  1. Review the evaluation report: outputs/reports/phase1_evaluation_report.md")
        print("  2. Examine visualizations in: outputs/figures/")
        print("  3. Proceed to Phase 2 for downstream applications")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
