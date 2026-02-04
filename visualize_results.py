import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import os

def comparison(entity, project, plot_keys=["test/loss", "test/acc"], run_names=["VIT_CIFAR100_FullPrec", "VIT_CIFAR100_Quant_[2, 4, 8, 16]"], run_id=None, output_dir="plot_comparisons"):
    # if not run_id:
    #     raise ValueError("At least one run ID must be provided for loss plotting.")
    api = wandb.Api()
        
    if run_id:
        runs = [api.run(f"{entity}/{project}/{run_id}")]
    else:
        runs = api.runs(f"{entity}/{project}")
        
    if run_names:
        runs = [run for run in runs if run.name in run_names]
    
    data = {}
    # get loss history for each run 
    for run in runs:
        for plot_key in plot_keys:
            if plot_key not in run.history().columns:
                print(f"Warning: {plot_key} not found in history for run {run.name}. Available keys: {run.history().columns}")
                continue
            print("Fetching history for run:", run.name)
            history = run.history(samples=10000) 
            
            if history.empty:
                print(f"No history found for run {run.name}")
                continue

            print("History ", history)
            train_metric = history[plot_key]
            print(f"{plot_key}:", train_metric)
            cols = history.columns
            print("Available columns:", cols)
            test_loss_cols = [c for c in cols if plot_key in c]
            data[run.name] = test_loss_cols
            # loss_cols = [c for c in cols if "loss" in c]
            # print(f"Run: {run.name} ({run.id}) - State: {run.state}, ")

        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        for run_name in data.keys():
            run = next(r for r in runs if r.name == run_name)
            history = run.history(samples=10000) 
            cols = history.columns
            test_loss_cols = [c for c in cols if plot_key in c]
            for c in test_loss_cols:
                sns.lineplot(data=history, x='_step', y=c, label=f"{run.name} - {c}")
        plt.title(f"{plot_key} over Epochs - Comparison")
        plt.xlabel("Step")
        plt.ylabel(plot_key)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison_{plot_key.replace('/', '_')}.png")

def visualize(entity, project, run_id=None, output_dir="plots"):
    api = wandb.Api()
    
    if run_id:
        runs = [api.run(f"{entity}/{project}/{run_id}")]
    else:
        # Get all runs from the project
        runs = api.runs(f"{entity}/{project}")
    
    for run in runs:
        print(f"Run: {run.name} ({run.id}) - State: {run.state}")
    exit()

    os.makedirs(output_dir, exist_ok=True)
    
    for run in runs:
        print(f"Processing run: {run.name} ({run.id})")
        
        # 1. Fetch history
        # Using a large samples value to get all steps
        history = run.history(samples=10000) 
        
        if history.empty:
            print(f"No history found for run {run.name}")
            continue

        # Filter relevant columns
        cols = history.columns
        loss_cols = [c for c in cols if "loss" in c]
        bit_cols = [c for c in cols if "bits/" in c]
        
        # 2. Plot Loss
        plt.figure(figsize=(10, 6))
        for c in loss_cols:
            sns.lineplot(data=history, x='_step', y=c, label=c)
        plt.title(f"Loss over Epochs - {run.name}")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(f"{output_dir}/{run.name}_loss.png")
        plt.close()
        
        # 3. Plot Bit Widths
        if not bit_cols:
            print(f"No bit width data found for {run.name}")
            continue
            
        # Structure data for heatmap
        # bits/layer_name -> value
        # We want: Index=Layer, Columns=Epoch/Step
        
        # Clean layer names: bits/layer.w_q -> layer.w_q
        
        # Extract bit data
        bit_data = history[["_step"] + bit_cols].copy()
        
        # Melt to long format: Step, Layer, Bits
        bit_data_melted = bit_data.melt(id_vars=["_step"], var_name="Layer", value_name="Bits")
        bit_data_melted["Layer"] = bit_data_melted["Layer"].apply(lambda x: x.replace("bits/", ""))
        
        # Separate into Weights and Activations if possible
        # Assumes naming convention ending in .w_q or .a_q based on LinearFQ/ConvFQ
        w_data = bit_data_melted[bit_data_melted["Layer"].str.contains(r"\.w_q|weights?|_w")]
        a_data = bit_data_melted[bit_data_melted["Layer"].str.contains(r"\.a_q|activations?|_a")]
        
        # Helper to plot heatmap
        def plot_heatmap(data, title_suffix, filename_suffix):
            if data.empty:
                return

            heatmap_data = data.dropna().pivot(index="Layer", columns="_step", values="Bits")
            if heatmap_data.empty:
                return

            # Reduce heatmap size if too large
            if heatmap_data.shape[1] > 20: # User wanted "smaller", downsample more aggressively
                indices = np.linspace(0, heatmap_data.shape[1]-1, 20, dtype=int)
                heatmap_data = heatmap_data.iloc[:, indices]

            plt.figure(figsize=(10, 6)) # Smaller figure size
            sns.heatmap(heatmap_data, cmap="viridis", annot=True, fmt=".0f", cbar_kws={'label': 'Bits'})
            plt.title(f"Bit Width Evolution ({title_suffix}) - {run.name}")
            plt.xlabel("Step")
            plt.ylabel("Layer Module")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{run.name}_bits_heatmap_{filename_suffix}.png")
            plt.close()

        # If strict naming (.w_q, .a_q) is found, split plots. Otherwise plot all.
        if not w_data.empty and not a_data.empty:
            plot_heatmap(w_data, "Weights", "weights")
            plot_heatmap(a_data, "Activations", "activations")
        else:
            # Fallback to plotting everything in one if regex didn't match cleanly
            plot_heatmap(bit_data_melted, "All", "all")

        # 4. Generate Bar Plots for Average Final Bit-Width per Layer
        # "visulaize the average bit over each layer instead og steps"
        
        def plot_avg_bits_per_layer(data, title_suffix, filename_suffix):
            if data.empty:
                return
            
            # Use last ~20% of steps or last 10 steps to determine "converged" average
            # If run is short, use all.
            steps = data["_step"].unique()
            if len(steps) > 5:
                cutoff = steps[int(len(steps) * 0.8)]
                recent_data = data[data["_step"] >= cutoff]
            else:
                recent_data = data
            
            # Calculate mean per layer
            layer_means = recent_data.groupby("Layer")["Bits"].mean().reset_index()
            
            def clean_name(name):
                print("name: ", name)
                n = name
                n = n.replace("encoder.block.", "enc.")
                n = n.replace("decoder.block.", "dec.")
                n = n.replace("model.", "")
                n = n.replace("backbone.", "")
                n = n.replace("features.", "")
                n = n.replace("SelfAttention", "SA")
                n = n.replace("DenseReluDense", "FFN")
                n = n.replace("vit.vit.encoder.", "encoder.")
                n = n.replace("attention.", "")
                # n = n.replace(".layer.", ".l.") # Keep somewhat verbose to avoid confusion if needed
                return n

            layer_means["CleanLayer"] = layer_means["Layer"].apply(clean_name)
            layer_means = layer_means.sort_values("Layer") # Sort by original full name to keep logical order
            
            plt.figure(figsize=(max(8, len(layer_means)*0.4), 7)) # Dynamic width
            
            # 2) Color bars by bit depth
            # Create a colormap
            norm = plt.Normalize(2, 16) # Assuming typical bit range 2-16
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
            sm.set_array([])
            
            # Create palette dict based on the values
            palette = {l: sm.to_rgba(b) for l, b in zip(layer_means["CleanLayer"], layer_means["Bits"])}

            # hue=CleanLayer is required to apply palette properly map-wise in recent seaborn
            sns.barplot(data=layer_means, x="CleanLayer", y="Bits", palette=palette, hue="CleanLayer", legend=False)
            
            plt.title(f"Average Final Bit-Width per Layer ({title_suffix})")
            plt.xlabel("Layer")
            plt.ylabel("Avg Bits")
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.ylim(0, 17) # Ensure room for labels
            
            # Add colorbar to show scale
            cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
            cbar.set_label('Bits')

            # Add value labels
            for i, v in enumerate(layer_means["Bits"]):
                plt.text(i, v + 0.2, f"{v:.1f}", ha='center', va='bottom', fontsize=9, rotation=90)
                
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{run.name}_avg_bits_bar_{filename_suffix}.png")
            plt.close()

        if not w_data.empty and not a_data.empty:
            plot_avg_bits_per_layer(w_data, "Weights", "weights")
            plot_avg_bits_per_layer(a_data, "Activations", "activations")
        else:
            plot_avg_bits_per_layer(bit_data_melted, "All", "all")

        print(f"Saved plots for {run.name}")
        
        # 4. Generate LaTeX Table for Average Final Bit-Widths
        generate_latex_table(run.name, bit_data, output_dir)

def generate_latex_table(run_name, bit_data, output_dir):
    # bit_data has columns: _step, [bits/layer1, bits/layer2...]
    
    # Get last 5 steps (or all if < 5) to average over convergence
    # (Assuming the run has converged)
    last_steps = bit_data.sort_values("_step").tail(5)
    
    # Calculate mean per layer
    # Note: bit_data contains _step and bits/..., so we filter numeric only
    means = last_steps.mean(numeric_only=True)
    
    # Filter for bit columns only
    bit_means = means[[c for c in means.index if "bits/" in c]]
    
    if bit_means.empty:
        return

    # Create DataFrame for table
    df = pd.DataFrame(bit_means, columns=["Avg Bit Width"])
    df.index.name = "Layer"
    
    # Clean up index: remove 'bits/' and replace common naming patterns for clarity
    df.index = df.index.str.replace("bits/", "")
    df.index = df.index.str.replace("model.", "") # clean pytorch model prefix if any
    
    df = df.sort_index()
    
    # Generate LaTeX
    # format float to 2 decimal places
    latex_str = df.to_latex(
        float_format="%.2f",
        caption=f"Average Final Bit-Widths per Layer ({run_name})",
        label=f"tab:bits_{run_name.replace('-', '_')}",
        column_format="|l|c|"
    )
    
    # Wrap in a full document structure for easy copy-pasting/testing if needed
    # or just the table snippet. Usually snippet is better for inclusion.
    # But adding a small note in the file.
    
    file_path = f"{output_dir}/{run_name}_bits_table.tex"
    with open(file_path, "w") as f:
        f.write("% Auto-generated table\n")
        f.write(latex_str)
        
    print(f"Saved LaTeX table for {run_name} to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default="user_entity_here", help="WandB entity/username")
    parser.add_argument("--project", type=str, default="frankenstein-quant", help="WandB project name")
    parser.add_argument("--run-id", type=str, help="Specific run ID to visualize")
    parser.add_argument("--compare", type=list, default=[], help="Generate comparison plots between runs")
    args = parser.parse_args()

    # # Try to guess entity if not provided? user needs to provide it or configure wandb
    if args.entity == "user_entity_here":
         # try to get default entity
         try:
             args.entity = wandb.Api().default_entity
         except:
             pass
    visualize(args.entity, args.project, args.run_id)
    if not args.run_id:
        print("\nGenerating Comparison Plots...")
        visualize_comparison(args.entity, args.project)
    if args.compare:
        print("\nGenerating Specified Comparison Plots...")
        comparison(args.entity, args.project, run_id=args.run_id)  