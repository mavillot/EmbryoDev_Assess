import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


class MethodComparator:
    """Compare performance across different training methods."""
    
    def __init__(self, comparison_dir: str = "output/comparison"):
        self.comparison_dir = Path(comparison_dir)
        self.results = {}
    
    def load_fragmentation_results(self):
        """Load fragmentation segmentation results from all methods."""
        fragmentation_results = {}
        
        # Multi-task method
        multitask_path = self.comparison_dir / "multitask" / "fragmentation_metrics.csv"
        if multitask_path.exists():
            df = pd.read_csv(multitask_path)
            fragmentation_results['Multi-Task'] = df.to_dict('records')[0]
        
        # Single-task method
        singletask_path = self.comparison_dir / "fragmentation" / "metrics_summary.csv"
        if singletask_path.exists():
            df = pd.read_csv(singletask_path)
            fragmentation_results['Single-Task'] = df.to_dict('records')[0]
        
        self.results['fragmentation'] = fragmentation_results
        return fragmentation_results
    
    def load_classification_results(self, task: str):
        """Load classification results for a specific task (exp, icm, te, ploidy)."""
        classification_results = {}
        
        # Multi-task method
        multitask_path = self.comparison_dir / "multitask" / f"{task}_classification_metrics.csv"
        if multitask_path.exists():
            df = pd.read_csv(multitask_path)
            classification_results['Multi-Task'] = df.to_dict('records')[0]
        
        # Single-task method
        singletask_path = self.comparison_dir / task / "metrics_summary.csv"
        if singletask_path.exists():
            df = pd.read_csv(singletask_path)
            classification_results['Single-Task'] = df.to_dict('records')[0]
        
        self.results[task] = classification_results
        return classification_results
    
    def compare_fragmentation(self, save_dir: Path = None):
        """Generate comparison for fragmentation segmentation."""
        if 'fragmentation' not in self.results:
            self.load_fragmentation_results()
        
        if not self.results['fragmentation']:
            print("No fragmentation results found")
            return
        
        if save_dir is None:
            save_dir = self.comparison_dir / "visualizations"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for visualization
        methods = list(self.results['fragmentation'].keys())
        metrics = ['accuracy', 'precision', 'recall', 'specificity', 'jaccard', 'dice']
        
        # Create comparison table
        comparison_data = []
        for method in methods:
            row = {'Method': method}
            for metric in metrics:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                
                if mean_key in self.results['fragmentation'][method]:
                    mean = self.results['fragmentation'][method][mean_key]
                    std = self.results['fragmentation'][method].get(std_key, 0)
                    row[metric] = f"{mean:.4f} ± {std:.4f}"
                    row[f'{metric}_value'] = mean
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Save table
        df[['Method'] + metrics].to_csv(save_dir / "fragmentation_comparison.csv", index=False)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, method in enumerate(methods):
            values = [comparison_data[i][f'{m}_value'] for m in metrics]
            offset = width * (i - len(methods)/2 + 0.5)
            ax.bar(x + offset, values, width, label=method)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Fragmentation Segmentation: Method Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(save_dir / "fragmentation_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print results
        print("\n" + "="*80)
        print("FRAGMENTATION SEGMENTATION COMPARISON")
        print("="*80)
        print(df[['Method'] + metrics].to_string(index=False))
        print("="*80)
    
    def compare_classification(self, task: str, save_dir: Path = None):
        """Generate comparison for classification task."""
        if task not in self.results:
            self.load_classification_results(task)
        
        if not self.results[task]:
            print(f"No {task} classification results found")
            return
        
        if save_dir is None:
            save_dir = self.comparison_dir / "visualizations"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        methods = list(self.results[task].keys())
        
        # Overall metrics
        overall_metrics = ['accuracy', 'f1_score', 'macro_precision', 'macro_recall', 'macro_specificity']
        
        # Create comparison table
        comparison_data = []
        for method in methods:
            row = {'Method': method}
            for metric in overall_metrics:
                if metric in self.results[task][method]:
                    value = self.results[task][method][metric]
                    row[metric] = f"{value:.4f}"
                    row[f'{metric}_value'] = value
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Save table
        df[['Method'] + overall_metrics].to_csv(save_dir / f"{task}_comparison.csv", index=False)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(overall_metrics))
        width = 0.35
        
        for i, method in enumerate(methods):
            values = [comparison_data[i].get(f'{m}_value', 0) for m in overall_metrics]
            offset = width * (i - len(methods)/2 + 0.5)
            ax.bar(x + offset, values, width, label=method)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{task.upper()} Classification: Method Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in overall_metrics], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(save_dir / f"{task}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print results
        print("\n" + "="*80)
        print(f"{task.upper()} CLASSIFICATION COMPARISON")
        print("="*80)
        print(df[['Method'] + overall_metrics].to_string(index=False))
        print("="*80)
    
    def generate_full_report(self):
        """Generate comprehensive comparison report."""
        save_dir = self.comparison_dir / "visualizations"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = save_dir / "comparison_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE METHOD COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Fragmentation
            if 'fragmentation' not in self.results:
                self.load_fragmentation_results()
            
            if self.results.get('fragmentation'):
                f.write("\nFRAGMENTATION SEGMENTATION\n")
                f.write("-"*80 + "\n")
                
                metrics = ['accuracy', 'precision', 'recall', 'specificity', 'jaccard', 'dice']
                for method, data in self.results['fragmentation'].items():
                    f.write(f"\n{method}:\n")
                    for metric in metrics:
                        mean_key = f'{metric}_mean'
                        std_key = f'{metric}_std'
                        if mean_key in data:
                            mean = data[mean_key]
                            std = data.get(std_key, 0)
                            f.write(f"  {metric.capitalize()}: {mean:.4f} ± {std:.4f}\n")
            
            # Classification tasks
            for task in ['exp', 'icm', 'te', 'ploidy']:
                if task not in self.results:
                    self.load_classification_results(task)
                
                if self.results.get(task):
                    f.write(f"\n{task.upper()} CLASSIFICATION\n")
                    f.write("-"*80 + "\n")
                    
                    for method, data in self.results[task].items():
                        f.write(f"\n{method}:\n")
                        f.write(f"  Accuracy: {data.get('accuracy', 0):.4f}\n")
                        f.write(f"  F1 Score: {data.get('f1_score', 0):.4f}\n")
                        f.write(f"  Macro Precision: {data.get('macro_precision', 0):.4f}\n")
                        f.write(f"  Macro Recall: {data.get('macro_recall', 0):.4f}\n")
                        f.write(f"  Macro Specificity: {data.get('macro_specificity', 0):.4f}\n")
        
        print(f"\nFull comparison report saved to {report_path}")


def main(args):
    """Main comparison routine."""
    comparator = MethodComparator(comparison_dir=args.comparison_dir)
    
    if args.task == 'all':
        # Compare all tasks
        print("Comparing all tasks...")
        
        comparator.compare_fragmentation()
        
        for task in ['exp', 'icm', 'te', 'ploidy']:
            try:
                comparator.compare_classification(task)
            except Exception as e:
                print(f"Could not compare {task}: {e}")
        
        comparator.generate_full_report()
    
    elif args.task == 'fragmentation':
        comparator.compare_fragmentation()
    
    elif args.task in ['exp', 'icm', 'te', 'ploidy']:
        comparator.compare_classification(args.task)
    
    else:
        print(f"Unknown task: {args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare performance across different training methods"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default='all',
        choices=['all', 'fragmentation', 'exp', 'icm', 'te', 'ploidy'],
        help="Task to compare (default: all)"
    )
    
    parser.add_argument(
        "--comparison_dir",
        type=str,
        default="output/comparison",
        help="Directory containing comparison results"
    )
    
    args = parser.parse_args()
    main(args)