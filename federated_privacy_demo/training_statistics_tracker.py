#!/usr/bin/env python3
"""
Training Statistics Tracker for Privacy-Preserving Federated Learning
Handles comprehensive logging and visualization of training metrics
"""

import os
import json
import csv
from collections import defaultdict
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class TrainingStatisticsTracker:
    """Tracks and saves training statistics in multiple formats for thesis reporting"""
    def __init__(self, base_dir="training_statistics"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = f"{base_dir}_{self.timestamp}"
        
        # Create directory structure
        self.create_directory_structure()
        
        # Initialize storage
        self.server_stats = defaultdict(list)
        self.client_stats = defaultdict(lambda: defaultdict(list))
        self.federated_stats = defaultdict(list)
        self.privacy_stats = defaultdict(lambda: defaultdict(list))
        
        # Setup CSV writers
        self.setup_csv_files()
        
        # Initialize JSON storage
        self.json_data = {
            "server_training": {},
            "client_training": {},
            "federated_learning": {},
            "privacy_metrics": {},
            "configuration": {},
            "final_results": {}
        }
        
        print(f"📊 Statistics will be saved to: {self.base_dir}")

    def create_directory_structure(self):
        """Create necessary directories for storing statistics"""
        # Main directory
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Subdirectories
        subdirs = ['csv', 'json', 'plots', 'latex']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.base_dir, subdir), exist_ok=True)

    def setup_csv_files(self):
        """Setup CSV files for continuous logging"""
        csv_dir = os.path.join(self.base_dir, 'csv')
        
        self.csv_files = {
            "server": open(os.path.join(csv_dir, 'server_training.csv'), 'w', newline=''),
            "clients": open(os.path.join(csv_dir, 'client_training.csv'), 'w', newline=''),
            "federated": open(os.path.join(csv_dir, 'federated_learning.csv'), 'w', newline=''),
            "privacy": open(os.path.join(csv_dir, 'privacy_metrics.csv'), 'w', newline='')
        }
        
        self.csv_writers = {
            "server": csv.writer(self.csv_files["server"]),
            "clients": csv.writer(self.csv_files["clients"]),
            "federated": csv.writer(self.csv_files["federated"]),
            "privacy": csv.writer(self.csv_files["privacy"])
        }
        
        # Write headers
        self.csv_writers["server"].writerow([
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc", 
            "lr", "time_elapsed"
        ])
        self.csv_writers["clients"].writerow([
            "client_id", "epoch", "train_loss", "train_acc", "val_loss", 
            "val_acc", "privacy_budget_used"
        ])
        self.csv_writers["federated"].writerow([
            "round", "global_loss", "global_acc", "avg_client_loss", 
            "avg_client_acc", "active_clients"
        ])
        self.csv_writers["privacy"].writerow([
            "round", "client_id", "epsilon_used", "delta", "noise_multiplier",
            "grad_norm", "sample_rate"
        ])

    def log_server_stats(self, epoch, stats):
        """Log server training statistics"""
        self.server_stats["epochs"].append(epoch)
        for key, value in stats.items():
            self.server_stats[key].append(value)
            
        # Write to CSV
        self.csv_writers["server"].writerow([
            epoch, stats.get("train_loss"), stats.get("train_acc"),
            stats.get("val_loss"), stats.get("val_acc"),
            stats.get("lr"), stats.get("time_elapsed")
        ])
        
        # Update JSON
        self.json_data["server_training"][f"epoch_{epoch}"] = stats

    def log_client_batch(self, round_num: int, client_id: str, epoch: int, batch_idx: int, stats: dict):
        """Log client batch training statistics
        
        Args:
            round_num: Current federated round number
            client_id: Client identifier
            epoch: Current epoch number
            batch_idx: Current batch index
            stats: Dictionary containing batch statistics
        """
        # Create a unique key for this batch
        batch_key = f"round_{round_num}_epoch_{epoch}_batch_{batch_idx}"
        
        # Update JSON storage
        if client_id not in self.json_data["client_training"]:
            self.json_data["client_training"][client_id] = {}
        if "batches" not in self.json_data["client_training"][client_id]:
            self.json_data["client_training"][client_id]["batches"] = {}
        self.json_data["client_training"][client_id]["batches"][batch_key] = {
            "round": round_num,
            "epoch": epoch,
            "batch": batch_idx,
            **stats
        }
        
        # Update client stats
        if "batches" not in self.client_stats[client_id]:
            self.client_stats[client_id]["batches"] = defaultdict(list)
        self.client_stats[client_id]["batches"]["rounds"].append(round_num)
        self.client_stats[client_id]["batches"]["epochs"].append(epoch)
        self.client_stats[client_id]["batches"]["batch_indices"].append(batch_idx)
        for key, value in stats.items():
            self.client_stats[client_id]["batches"][key].append(value)

    def log_client_epoch(self, round_num: int, client_id: str, epoch: int, stats: dict):
        """Log client epoch training statistics
        
        Args:
            round_num: Current federated round number
            client_id: Client identifier
            epoch: Current epoch number
            stats: Dictionary containing epoch statistics
        """
        # Update JSON storage
        if client_id not in self.json_data["client_training"]:
            self.json_data["client_training"][client_id] = {}
        if "epochs" not in self.json_data["client_training"][client_id]:
            self.json_data["client_training"][client_id]["epochs"] = {}
        self.json_data["client_training"][client_id]["epochs"][f"round_{round_num}_epoch_{epoch}"] = {
            "round": round_num,
            "epoch": epoch,
            **stats
        }
        
        # Update client stats
        if "epochs" not in self.client_stats[client_id]:
            self.client_stats[client_id]["epochs"] = defaultdict(list)
        self.client_stats[client_id]["epochs"]["rounds"].append(round_num)
        self.client_stats[client_id]["epochs"]["epochs"].append(epoch)
        for key, value in stats.items():
            self.client_stats[client_id]["epochs"][key].append(value)
        
        # Write to CSV (using existing log_client_stats method)
        self.log_client_stats(client_id, epoch, {
            "round": round_num,
            **stats
        })

    def log_client_stats(self, round_num: int, client_id: str, stats: dict):
        """Log final client statistics for a federated round
        
        Args:
            round_num: Current federated round number
            client_id: Client identifier
            stats: Dictionary containing final client statistics
        """
        # Update JSON storage
        if client_id not in self.json_data["client_training"]:
            self.json_data["client_training"][client_id] = {}
        if "rounds" not in self.json_data["client_training"][client_id]:
            self.json_data["client_training"][client_id]["rounds"] = {}
        self.json_data["client_training"][client_id]["rounds"][f"round_{round_num}"] = {
            "round": round_num,
            **stats
        }
        
        # Update client stats
        if "rounds" not in self.client_stats[client_id]:
            self.client_stats[client_id]["rounds"] = defaultdict(list)
        self.client_stats[client_id]["rounds"]["rounds"].append(round_num)
        for key, value in stats.items():
            self.client_stats[client_id]["rounds"][key].append(value)
        
        # Write to CSV
        self.csv_writers["clients"].writerow([
            client_id, round_num, stats.get("loss"), stats.get("accuracy"),
            stats.get("val_loss", 0), stats.get("val_acc", 0),
            stats.get("privacy_budget_used", 0)
        ])

    def log_federated_stats(self, round_num, stats):
        """Log federated learning round statistics"""
        self.federated_stats["rounds"].append(round_num)
        for key, value in stats.items():
            self.federated_stats[key].append(value)
            
        # Write to CSV
        self.csv_writers["federated"].writerow([
            round_num, stats.get("global_loss"), stats.get("global_acc"),
            stats.get("avg_client_loss"), stats.get("avg_client_acc"),
            stats.get("active_clients")
        ])
        
        # Update JSON
        self.json_data["federated_learning"][f"round_{round_num}"] = stats

    def log_privacy_metrics(self, round_num, client_id, metrics):
        """Log privacy-related metrics"""
        self.privacy_stats[client_id]["rounds"].append(round_num)
        for key, value in metrics.items():
            self.privacy_stats[client_id][key].append(value)
            
        # Write to CSV
        self.csv_writers["privacy"].writerow([
            round_num, client_id, metrics.get("epsilon_used"),
            metrics.get("delta"), metrics.get("noise_multiplier"),
            metrics.get("grad_norm"), metrics.get("sample_rate")
        ])
        
        # Update JSON
        if client_id not in self.json_data["privacy_metrics"]:
            self.json_data["privacy_metrics"][client_id] = {}
        self.json_data["privacy_metrics"][client_id][f"round_{round_num}"] = metrics

    def save_final_results(self, results):
        """Save final training results"""
        self.json_data["final_results"] = results
        
        # Save as separate JSON for quick access
        with open(os.path.join(self.base_dir, 'json', 'final_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    def save_configuration(self, config):
        """Save training configuration"""
        self.json_data["configuration"] = config
        
        # Save as separate JSON for quick access
        with open(os.path.join(self.base_dir, 'json', 'configuration.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def generate_plots(self):
        """Generate visualization plots"""
        plots_dir = os.path.join(self.base_dir, 'plots')
        
        # Server training plots
        if self.server_stats["epochs"]:
            plt.figure(figsize=(12, 6))
            plt.plot(self.server_stats["epochs"], self.server_stats["train_acc"], label="Train Acc")
            if "val_acc" in self.server_stats:
                plt.plot(self.server_stats["epochs"], self.server_stats["val_acc"], label="Val Acc")
            plt.title("Server Training Progress")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.legend()
            plt.savefig(os.path.join(plots_dir, 'server_training.png'))
            plt.close()
        
        # Federated learning plots
        if self.federated_stats["rounds"]:
            plt.figure(figsize=(12, 6))
            plt.plot(self.federated_stats["rounds"], self.federated_stats["global_acc"], label="Global Acc")
            plt.title("Federated Learning Progress")
            plt.xlabel("Round")
            plt.ylabel("Accuracy (%)")
            plt.legend()
            plt.savefig(os.path.join(plots_dir, 'federated_learning.png'))
            plt.close()
        
        # Privacy budget usage
        if self.privacy_stats:
            plt.figure(figsize=(12, 6))
            for client_id in self.privacy_stats.keys():
                if "epsilon_used" in self.privacy_stats[client_id]:
                    plt.plot(self.privacy_stats[client_id]["rounds"], 
                            self.privacy_stats[client_id]["epsilon_used"],
                            label=f"{client_id}")
            plt.title("Privacy Budget Usage")
            plt.xlabel("Round")
            plt.ylabel("Epsilon Used")
            plt.legend()
            plt.savefig(os.path.join(plots_dir, 'privacy_budget.png'))
            plt.close()

    def generate_latex_tables(self):
        """Generate LaTeX tables for thesis"""
        latex_dir = os.path.join(self.base_dir, 'latex')
        
        # Final results table
        with open(os.path.join(latex_dir, 'final_results.tex'), 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Final Training Results}\n")
            f.write("\\begin{tabular}{lc}\n")
            f.write("\\toprule\n")
            f.write("Metric & Value \\\\\n")
            f.write("\\midrule\n")
            for key, value in self.json_data["final_results"].items():
                if isinstance(value, float):
                    f.write(f"{key} & {value:.4f} \\\\\n")
                else:
                    f.write(f"{key} & {value} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}")

    def save_all(self):
        """Save all statistics and close files"""
        # Save complete JSON data
        with open(os.path.join(self.base_dir, 'json', 'complete_statistics.json'), 'w') as f:
            json.dump(self.json_data, f, indent=2)
        
        # Save pandas DataFrames for easy analysis
        if self.server_stats:
            pd.DataFrame(self.server_stats).to_csv(
                os.path.join(self.base_dir, 'csv', 'server_stats_processed.csv'))
        
        for client_id, stats in self.client_stats.items():
            if stats:
                pd.DataFrame(stats).to_csv(
                    os.path.join(self.base_dir, 'csv', f'client_{client_id}_stats_processed.csv'))
        
        if self.federated_stats:
            pd.DataFrame(self.federated_stats).to_csv(
                os.path.join(self.base_dir, 'csv', 'federated_stats_processed.csv'))
        
        # Generate plots
        self.generate_plots()
        
        # Generate LaTeX tables
        self.generate_latex_tables()
        
        # Close CSV files
        for file in self.csv_files.values():
            file.close()
        
        # Create README
        with open(os.path.join(self.base_dir, 'README.md'), 'w') as f:
            f.write("# Training Statistics\n\n")
            f.write(f"Generated on: {self.timestamp}\n\n")
            f.write("## Contents\n\n")
            f.write("1. CSV Files (in `csv/`)\n")
            f.write("   - server_training.csv: Server training metrics\n")
            f.write("   - client_training.csv: Client training metrics\n")
            f.write("   - federated_learning.csv: Federated learning metrics\n")
            f.write("   - privacy_metrics.csv: Privacy-related metrics\n")
            f.write("   - *_processed.csv: Processed statistics for analysis\n\n")
            f.write("2. JSON Files (in `json/`)\n")
            f.write("   - complete_statistics.json: All metrics\n")
            f.write("   - configuration.json: Training configuration\n")
            f.write("   - final_results.json: Final results summary\n\n")
            f.write("3. Visualizations (in `plots/`)\n")
            f.write("   - server_training.png: Server training progress\n")
            f.write("   - federated_learning.png: Federated learning progress\n")
            f.write("   - privacy_budget.png: Privacy budget usage\n\n")
            f.write("4. LaTeX Tables (in `latex/`)\n")
            f.write("   - final_results.tex: Results in LaTeX format\n\n")
            f.write("## Quick Start\n\n")
            f.write("1. For quick overview: Check plots in `plots/`\n")
            f.write("2. For detailed analysis: Use processed CSV files\n")
            f.write("3. For thesis tables: Use LaTeX files in `latex/`\n")
            f.write("4. For complete data: Use JSON files in `json/`\n") 