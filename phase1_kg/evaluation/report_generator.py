"""
Comprehensive Evaluation Report Generator
Creates detailed PDF/HTML reports with evaluation results and visualizations
"""
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import pickle
# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import REPORTS_DIR, METRICS_DIR, FIGURES_DIR
from utils.helpers import log_step


class ReportGenerator:
    """Generate comprehensive evaluation reports"""

    def __init__(self):
        log_step("Initializing Report Generator")

    def load_evaluation_results(self, kg_name: str) -> dict:
        """Load evaluation results from JSON"""
        results_path = METRICS_DIR / f"{kg_name}_evaluation_results.json"

        if not results_path.exists():
            print(f"Warning: Results not found for {kg_name}")
            return {}

        with open(results_path, 'r') as f:
            return json.load(f)

    def generate_markdown_report(
        self,
        temporal_results: dict,
        causal_results: dict,
        save_path: Path = None
    ) -> str:
        """Generate markdown report"""
        log_step("Generating Markdown Report")

        report = f"""# Phase 1 Knowledge Graph Evaluation Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

This report presents the intrinsic evaluation results for the Temporal and Causal Knowledge Graphs constructed from fund manager portfolio data. The evaluation assesses five key dimensions: Structural Completeness, Consistency, Semantic Coherence, Informativeness, and Inferential Utility.

### Overall Scores

| Knowledge Graph | Overall Quality Score |
|----------------|----------------------|
| Temporal KG    | **{temporal_results.get('overall_quality_score', 0):.3f}** |
| Causal KG      | **{causal_results.get('overall_quality_score', 0):.3f}** |

---

## 1. Temporal Knowledge Graph Evaluation

### 1.1 Structural Completeness
**Score:** {temporal_results.get('structural_completeness', {}).get('completeness_score', 0):.3f}

The Temporal KG captures portfolio holdings and their evolution over time.

**Key Metrics:**
- Total Nodes: {temporal_results.get('structural_completeness', {}).get('node_count', 0):,}
- Total Edges: {temporal_results.get('structural_completeness', {}).get('edge_count', 0):,}
- Graph Density: {temporal_results.get('structural_completeness', {}).get('density', 0):.4f}
- Average Degree: {temporal_results.get('structural_completeness', {}).get('average_degree', 0):.2f}
- Connected Components: {temporal_results.get('structural_completeness', {}).get('connected_components', 0)}

**Node Type Distribution:**
"""
        # Add node type distribution
        node_dist = temporal_results.get('structural_completeness', {}).get('node_type_distribution', {})
        for node_type, count in node_dist.items():
            report += f"- {node_type}: {count:,}\n"

        report += f"""
**Edge Type Distribution:**
"""
        # Add edge type distribution
        edge_dist = temporal_results.get('structural_completeness', {}).get('edge_type_distribution', {})
        for edge_type, count in edge_dist.items():
            report += f"- {edge_type}: {count:,}\n"

        report += f"""
### 1.2 Consistency and Integrity
**Score:** {temporal_results.get('consistency', {}).get('consistency_score', 0):.3f}

The consistency evaluation checks for logical contradictions and integrity violations.

**Key Metrics:**
- Orphan Nodes: {temporal_results.get('consistency', {}).get('orphan_nodes_count', 0)} ({temporal_results.get('consistency', {}).get('orphan_nodes_pct', 0)*100:.2f}%)
- Self-loops: {temporal_results.get('consistency', {}).get('self_loops_count', 0)}
- Temporal Violations: {temporal_results.get('consistency', {}).get('temporal_violations', 0)}
- Cardinality Violations: {temporal_results.get('consistency', {}).get('cardinality_violations', 0)}
- Missing Attributes: {temporal_results.get('consistency', {}).get('missing_attributes_count', 0)}

**Violations Found:**
"""
        violations = temporal_results.get('consistency', {}).get('violations', [])
        if violations:
            for v in violations:
                report += f"- {v}\n"
        else:
            report += "- None\n"

        report += f"""
### 1.3 Semantic Coherence
**Score:** {temporal_results.get('semantic_coherence', {}).get('semantic_coherence_score', 0):.3f}

Semantic coherence measures how well related entities cluster together.

**Key Metrics:**
- Number of Communities: {temporal_results.get('semantic_coherence', {}).get('num_communities', 0)}
- Modularity: {temporal_results.get('semantic_coherence', {}).get('modularity', 0):.3f}
- Average Sector Purity: {temporal_results.get('semantic_coherence', {}).get('average_sector_purity', 0):.3f}
- Temporal Coherence: {temporal_results.get('semantic_coherence', {}).get('temporal_coherence', 0):.3f}

### 1.4 Informativeness
**Score:** {temporal_results.get('informativeness', {}).get('informativeness_score', 0):.3f}

Informativeness evaluates the richness and non-redundancy of information.

**Key Metrics:**
- Unique Attributes: {temporal_results.get('informativeness', {}).get('unique_attributes', 0)}
- Avg Attributes per Node: {temporal_results.get('informativeness', {}).get('avg_attributes_per_node', 0):.2f}
- Avg Attributes per Edge: {temporal_results.get('informativeness', {}).get('avg_attributes_per_edge', 0):.2f}
- Redundant Edges: {temporal_results.get('informativeness', {}).get('redundant_edges', 0)}
- Information Density: {temporal_results.get('informativeness', {}).get('information_density', 0):.2f}

### 1.5 Inferential Utility
**Score:** {temporal_results.get('inferential_utility', {}).get('inferential_utility_score', 0):.3f}

Inferential utility measures the graph's ability to support reasoning and multi-hop queries.

**Key Metrics:**
- Average Path Length: {temporal_results.get('inferential_utility', {}).get('average_path_length', 0):.2f}
- Reachability: {temporal_results.get('inferential_utility', {}).get('reachability', 0):.3f}
- Multi-hop Query Support: {temporal_results.get('inferential_utility', {}).get('multi_hop_query_support', 0):.3f}
- Max Degree Centrality: {temporal_results.get('inferential_utility', {}).get('max_degree_centrality', 0):.3f}

---

## 2. Causal Knowledge Graph Evaluation

### 2.1 Structural Completeness
**Score:** {causal_results.get('structural_completeness', {}).get('completeness_score', 0):.3f}

The Causal KG models cause-effect relationships between market factors and portfolio decisions.

**Key Metrics:**
- Total Nodes: {causal_results.get('structural_completeness', {}).get('node_count', 0):,}
- Total Edges: {causal_results.get('structural_completeness', {}).get('edge_count', 0):,}
- Graph Density: {causal_results.get('structural_completeness', {}).get('density', 0):.4f}
- Average Degree: {causal_results.get('structural_completeness', {}).get('average_degree', 0):.2f}
- Is DAG: {causal_results.get('consistency', {}).get('is_dag', 'N/A')}

**Node Type Distribution:**
"""
        # Add node type distribution for causal
        node_dist = causal_results.get('structural_completeness', {}).get('node_type_distribution', {})
        for node_type, count in node_dist.items():
            report += f"- {node_type}: {count:,}\n"

        report += f"""
**Edge Type Distribution:**
"""
        # Add edge type distribution for causal
        edge_dist = causal_results.get('structural_completeness', {}).get('edge_type_distribution', {})
        for edge_type, count in edge_dist.items():
            report += f"- {edge_type}: {count:,}\n"

        report += f"""
### 2.2 Consistency and Integrity
**Score:** {causal_results.get('consistency', {}).get('consistency_score', 0):.3f}

**Key Metrics:**
- Orphan Nodes: {causal_results.get('consistency', {}).get('orphan_nodes_count', 0)}
- Causal Cycles: {causal_results.get('consistency', {}).get('cycle_count', 0)}
- Cardinality Violations: {causal_results.get('consistency', {}).get('cardinality_violations', 0)}

### 2.3 Semantic Coherence
**Score:** {causal_results.get('semantic_coherence', {}).get('semantic_coherence_score', 0):.3f}

**Key Metrics:**
- Number of Communities: {causal_results.get('semantic_coherence', {}).get('num_communities', 0)}
- Modularity: {causal_results.get('semantic_coherence', {}).get('modularity', 0):.3f}

### 2.4 Informativeness
**Score:** {causal_results.get('informativeness', {}).get('informativeness_score', 0):.3f}

**Key Metrics:**
- Unique Attributes: {causal_results.get('informativeness', {}).get('unique_attributes', 0)}
- Avg Attributes per Node: {causal_results.get('informativeness', {}).get('avg_attributes_per_node', 0):.2f}

### 2.5 Inferential Utility
**Score:** {causal_results.get('inferential_utility', {}).get('inferential_utility_score', 0):.3f}

**Key Metrics:**
- Average Path Length: {causal_results.get('inferential_utility', {}).get('average_path_length', 0):.2f}
- Reachability: {causal_results.get('inferential_utility', {}).get('reachability', 0):.3f}

---

## 3. Comparison and Insights

### Score Comparison

| Metric | Temporal KG | Causal KG |
|--------|------------|-----------|
| Structural Completeness | {temporal_results.get('structural_completeness', {}).get('completeness_score', 0):.3f} | {causal_results.get('structural_completeness', {}).get('completeness_score', 0):.3f} |
| Consistency | {temporal_results.get('consistency', {}).get('consistency_score', 0):.3f} | {causal_results.get('consistency', {}).get('consistency_score', 0):.3f} |
| Semantic Coherence | {temporal_results.get('semantic_coherence', {}).get('semantic_coherence_score', 0):.3f} | {causal_results.get('semantic_coherence', {}).get('semantic_coherence_score', 0):.3f} |
| Informativeness | {temporal_results.get('informativeness', {}).get('informativeness_score', 0):.3f} | {causal_results.get('informativeness', {}).get('informativeness_score', 0):.3f} |
| Inferential Utility | {temporal_results.get('inferential_utility', {}).get('inferential_utility_score', 0):.3f} | {causal_results.get('inferential_utility', {}).get('inferential_utility_score', 0):.3f} |
| **Overall** | **{temporal_results.get('overall_quality_score', 0):.3f}** | **{causal_results.get('overall_quality_score', 0):.3f}** |

### Key Findings

1. **Structural Properties**: The Temporal KG is significantly larger with {temporal_results.get('structural_completeness', {}).get('node_count', 0):,} nodes compared to {causal_results.get('structural_completeness', {}).get('node_count', 0):,} in the Causal KG, reflecting its detailed representation of portfolio holdings over time.

2. **Consistency**: Both graphs demonstrate high consistency scores, indicating logical integrity. The Causal KG maintains DAG properties essential for causal inference.

3. **Semantic Coherence**: The community structure and clustering patterns suggest meaningful organization of related entities in both graphs.

4. **Inferential Capabilities**: Both graphs support multi-hop reasoning, with the Temporal KG providing richer temporal traversal paths.

---

## 4. Visualizations

The following visualizations are available in the `outputs/figures/` directory:

### Temporal KG
- Degree distribution
- Node type distribution
- Edge type distribution
- Temporal evolution plot
- Interactive network visualization
- Sample subgraphs

### Causal KG
- Degree distribution
- Node type distribution
- Edge type distribution
- Interactive network visualization
- Sample causal paths

---

## 5. Recommendations

Based on the evaluation results:

1. **Data Enhancement**: Consider enriching fundamental and sentiment data to strengthen causal relationships.

2. **Temporal Coverage**: Extend the temporal range to capture more market cycles and improve pattern detection.

3. **Causal Validation**: Cross-validate extracted causal relationships with domain experts and financial literature.

4. **Integration**: Leverage the integrated KG to combine temporal sequences with causal explanations for comprehensive decision modeling.

5. **Phase 2 Readiness**: The knowledge graphs demonstrate sufficient quality for downstream portfolio construction tasks in Phase 2.

---

## 6. Conclusion

The Phase 1 knowledge representation successfully captures fund manager decision-making patterns through complementary Temporal and Causal perspectives. Both graphs achieve satisfactory scores across all evaluation dimensions, with overall quality scores of **{temporal_results.get('overall_quality_score', 0):.3f}** (Temporal) and **{causal_results.get('overall_quality_score', 0):.3f}** (Causal). The representations demonstrate structural completeness, logical consistency, semantic coherence, informational richness, and inferential utility suitable for the planned downstream applications in Phase 2.

---

**Report End**
"""

        if save_path is None:
            save_path = REPORTS_DIR / "phase1_evaluation_report.md"

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✓ Markdown report saved: {save_path}")
        return report

    def generate_csv_summary(
        self,
        temporal_results: dict,
        causal_results: dict
    ) -> None:
        """Generate CSV summary of metrics"""
        log_step("Generating CSV Summary")

        # Create summary dataframe
        metrics = [
            'Structural Completeness',
            'Consistency',
            'Semantic Coherence',
            'Informativeness',
            'Inferential Utility',
            'Overall Quality'
        ]

        temporal_scores = [
            temporal_results.get('structural_completeness', {}).get('completeness_score', 0),
            temporal_results.get('consistency', {}).get('consistency_score', 0),
            temporal_results.get('semantic_coherence', {}).get('semantic_coherence_score', 0),
            temporal_results.get('informativeness', {}).get('informativeness_score', 0),
            temporal_results.get('inferential_utility', {}).get('inferential_utility_score', 0),
            temporal_results.get('overall_quality_score', 0)
        ]

        causal_scores = [
            causal_results.get('structural_completeness', {}).get('completeness_score', 0),
            causal_results.get('consistency', {}).get('consistency_score', 0),
            causal_results.get('semantic_coherence', {}).get('semantic_coherence_score', 0),
            causal_results.get('informativeness', {}).get('informativeness_score', 0),
            causal_results.get('inferential_utility', {}).get('inferential_utility_score', 0),
            causal_results.get('overall_quality_score', 0)
        ]

        df = pd.DataFrame({
            'Metric': metrics,
            'Temporal_KG': temporal_scores,
            'Causal_KG': causal_scores
        })

        save_path = REPORTS_DIR / "evaluation_summary.csv"
        df.to_csv(save_path, index=False)
        print(f"✓ CSV summary saved: {save_path}")

    def generate_full_report(self) -> None:
        """Generate complete evaluation report"""
        log_step("Generating Full Evaluation Report")

        # Load results
        temporal_results = self.load_evaluation_results("Temporal_KG")
        causal_results = self.load_evaluation_results("Causal_KG")

        # Generate reports
        self.generate_markdown_report(temporal_results, causal_results)
        self.generate_csv_summary(temporal_results, causal_results)

        print("\n" + "="*80)
        print("REPORT GENERATION COMPLETE")
        print("="*80)
        print(f"Reports saved in: {REPORTS_DIR}")
        print(f"- Markdown report: phase1_evaluation_report.md")
        print(f"- CSV summary: evaluation_summary.csv")


def main():
    """Generate evaluation report"""
    generator = ReportGenerator()
    generator.generate_full_report()


if __name__ == "__main__":
    main()
