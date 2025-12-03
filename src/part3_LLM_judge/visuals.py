import json
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np


class Part3Visualizer:
    
    def __init__(self, output_dir: str = "data/part3_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def plot_all(self, validation_path: str = None, judge_path: str = None):
        if validation_path is None:
            validation_path = self.output_dir / "validation_results.json"
        if judge_path is None:
            judge_path = self.output_dir / "judge_results.json"
        
        with open(validation_path, 'r') as f:
            validation = json.load(f)
        with open(judge_path, 'r') as f:
            judge_results = json.load(f)
        
        self.plot_ablation(validation.get("ablation_study", {}))
        self.plot_self_consistency(validation.get("self_consistency", []))
        self.plot_cohens_kappa(validation.get("cohens_kappa", {}))
        self.plot_judge_summary(judge_results)
        
        print(f"All plots saved to {self.output_dir}")
    
    def plot_ablation(self, ablation: Dict):
        if not ablation:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        strategies = list(ablation.keys())
        means = [ablation[s]["mean"] for s in strategies]
        stds = [ablation[s]["std"] for s in strategies]
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = ax.bar(strategies, means, yerr=stds, capsize=8, color=colors[:len(strategies)], 
                      edgecolor='white', linewidth=2)
        
        ax.set_ylabel('Mean Consistency Score', fontsize=12)
        ax.set_xlabel('Prompt Strategy', fontsize=12)
        ax.set_title('Ablation Study: Prompt Strategy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Threshold')
        
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 3,
                   f'{mean:.1f}', ha='center', fontsize=11, fontweight='bold')
        
        for i, (strategy, std) in enumerate(zip(strategies, stds)):
            ax.text(i, 5, f'σ={std:.1f}', ha='center', fontsize=10, color='white', fontweight='bold')
        
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(self.output_dir / "ablation_study.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_self_consistency(self, consistency: List[Dict]):
        if not consistency:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        events = [r["event"][:20] for r in consistency]
        stds = [r["std"] for r in consistency]
        colors = ['#2ecc71' if r["is_stable"] else '#e74c3c' for r in consistency]
        
        bars = ax1.barh(events, stds, color=colors, edgecolor='white', linewidth=2)
        ax1.axvline(x=10, color='gray', linestyle='--', linewidth=2, label='Stability Threshold')
        ax1.set_xlabel('Standard Deviation', fontsize=12)
        ax1.set_title('Score Variance by Event', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        
        for i, (event, std, stable) in enumerate(zip(events, stds, [r["is_stable"] for r in consistency])):
            label = "STABLE" if stable else "NOISY"
            ax1.text(std + 0.5, i, label, va='center', fontsize=9, fontweight='bold')
        
        for i, result in enumerate(consistency):
            scores = result["scores"]
            ax2.scatter([i] * len(scores), scores, alpha=0.6, s=80, c='#3498db')
            ax2.plot([i], [result["mean"]], 'k_', markersize=25, markeredgewidth=3)
        
        ax2.set_xticks(range(len(events)))
        ax2.set_xticklabels(events, rotation=45, ha='right')
        ax2.set_ylabel('Consistency Score', fontsize=12)
        ax2.set_title('Score Distribution (5 runs each)', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "self_consistency.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_cohens_kappa(self, kappa: Dict):
        if not kappa or "confusion" not in kappa:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        cm = kappa.get("confusion", {})
        matrix = np.array([
            [cm.get("TP", 0), cm.get("FP", 0)],
            [cm.get("FN", 0), cm.get("TN", 0)]
        ])
        
        im = ax1.imshow(matrix, cmap='Blues')
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Consistent', 'Contradictory'], fontsize=11)
        ax1.set_yticklabels(['Consistent', 'Contradictory'], fontsize=11)
        ax1.set_xlabel('Human Labels', fontsize=12)
        ax1.set_ylabel('LLM Labels', fontsize=12)
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        for i in range(2):
            for j in range(2):
                color = 'white' if matrix[i, j] > matrix.max()/2 else 'black'
                ax1.text(j, i, str(int(matrix[i, j])), ha='center', va='center', 
                        fontsize=18, fontweight='bold', color=color)
        
        plt.colorbar(im, ax=ax1)
        
        kappa_val = kappa.get("kappa", 0)
        ranges = [(-0.2, 0), (0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        labels = ['Poor', 'Slight', 'Fair', 'Moderate', 'Substantial', 'Perfect']
        colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#2ecc71', '#27ae60']
        
        for i, ((low, high), label, color) in enumerate(zip(ranges, labels, colors)):
            alpha = 1.0 if low <= kappa_val < high else 0.3
            ax2.barh(i, high - low, left=low, color=color, alpha=alpha, edgecolor='white', linewidth=2)
            ax2.text((low + high) / 2, i, label, ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax2.axvline(x=kappa_val, color='red', linewidth=3, label=f'k = {kappa_val:.3f}')
        ax2.set_xlim(-0.2, 1.0)
        ax2.set_yticks([])
        ax2.set_xlabel("Cohen's Kappa Value", fontsize=12)
        ax2.set_title("Kappa Interpretation Scale", fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "cohens_kappa.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_judge_summary(self, judge_results: List[Dict]):
        if not judge_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        scores = [r["consistency_score"] for r in judge_results]
        axes[0, 0].hist(scores, bins=10, range=(0, 100), color='#3498db', edgecolor='white', linewidth=2)
        axes[0, 0].axvline(x=50, color='red', linestyle='--', linewidth=2, label='Threshold')
        axes[0, 0].axvline(x=np.mean(scores), color='green', linestyle='-', linewidth=2, label=f'Mean={np.mean(scores):.1f}')
        axes[0, 0].set_xlabel('Consistency Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Score Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        
        events = list(set(r["event"] for r in judge_results))
        event_scores = {e: [r["consistency_score"] for r in judge_results if r["event"] == e] for e in events}
        event_means = [np.mean(event_scores[e]) for e in events]
        
        colors = ['#2ecc71' if m >= 50 else '#e74c3c' for m in event_means]
        bars = axes[0, 1].barh([e[:25] for e in events], event_means, color=colors, edgecolor='white')
        axes[0, 1].axvline(x=50, color='gray', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Mean Consistency Score')
        axes[0, 1].set_title('Consistency by Event', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlim(0, 100)
        
        contradiction_types = {"factual": 0, "interpretive": 0, "omission": 0}
        for r in judge_results:
            for c in r.get("contradictions", []):
                ctype = c.get("type", "factual").lower()
                if ctype in contradiction_types:
                    contradiction_types[ctype] += 1
        
        if sum(contradiction_types.values()) > 0:
            axes[1, 0].pie(
                contradiction_types.values(), 
                labels=[t.title() for t in contradiction_types.keys()],
                autopct='%1.1f%%',
                colors=['#e74c3c', '#3498db', '#f39c12'],
                explode=(0.05, 0.05, 0.05)
            )
            axes[1, 0].set_title('Contradiction Types', fontsize=12, fontweight='bold')
        
        authors = list(set(r["other_author"] for r in judge_results))
        author_scores = {a: [r["consistency_score"] for r in judge_results if r["other_author"] == a] for a in authors}
        author_means = [np.mean(author_scores[a]) for a in authors]
        author_stds = [np.std(author_scores[a]) for a in authors]
        
        x = np.arange(len(authors))
        axes[1, 1].bar(x, author_means, yerr=author_stds, capsize=5, color='#9b59b6', edgecolor='white')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([a[:12] for a in authors], rotation=45, ha='right')
        axes[1, 1].set_ylabel('Consistency Score')
        axes[1, 1].set_title('Consistency by Author', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "judge_summary.png", dpi=150, bbox_inches='tight')
        plt.close()


def generate_plots():
    viz = Part3Visualizer()
    viz.plot_all()
    print("All visualizations generated")


if __name__ == "__main__":
    generate_plots()