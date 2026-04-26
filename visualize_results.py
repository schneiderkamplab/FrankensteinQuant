import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import os
import yaml

def comparison(entity, project, plot_name, plot_config, plot_key=[], run_names=[], labels={}, run_id=None, name=None, output_dir="plot_comparisons"):
    api = wandb.Api()
        
    if run_id:
        runs = [api.run(f"{entity}/{project}/{run_id}")]
    else:
        runs = api.runs(f"{entity}/{project}")
        
    if run_names:
        runs = [run for run in runs if run.name in run_names]

    data = {}
    for run in runs:
        for key in plot_key :
            if key not in run.history().columns:
                print(f"Warning: {key} not found in history for run {run.name}. Available keys: {run.history().columns}")
                continue
            history = run.history(samples=10000)             
            if history.empty:
                print(f"No history found for run {run.name}")
                continue
            train_metric = history[key]
            print(f"{key}:", train_metric)
            cols = history.columns
            print("Available columns:", cols)
            test_loss_cols = [c for c in cols if key in c]
            data[run.name] = test_loss_cols

        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        for run_name in data.keys():
            run = next(r for r in runs if r.name == run_name)
            history = run.history(samples=10000) 
            cols = history.columns
            test_loss_cols = [c for c in cols if key in c]
            for c in test_loss_cols:
                display_label = labels.get(run.name, run.name)    
                sns.lineplot(data=history, x='_step', y=c, label=display_label)
        plt.title(plot_config['title'])
        plt.xlabel(plot_config['x_label'])
        plt.ylabel(plot_config['y_label'])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{plot_name}.png")
        plt.close()

def visualize(entity, project, run_id=None, output_dir="plots"):
    api = wandb.Api()

    if run_id:
        runs = [api.run(f"{entity}/{project}/{run_id}")]
    else:
        runs = api.runs(f"{entity}/{project}")

    os.makedirs(output_dir, exist_ok=True)

    for run in runs:
        print(f"Processing run: {run.name} ({run.id})")

        history = run.history(samples=10000)
        if history.empty:
            print(f"No history found for run {run.name}")
            continue

        cols = history.columns
        bit_cols = [c for c in cols if "bits/" in c]
        if not bit_cols:
            print(f"No bit width data found for {run.name}")
            continue

        bit_data = history[["_step"] + bit_cols].copy()
        bit_data_melted = bit_data.melt(id_vars=["_step"], var_name="Layer", value_name="Bits")
        bit_data_melted["Layer"] = bit_data_melted["Layer"].apply(lambda x: x.replace("bits/", ""))

        w_data = bit_data_melted[bit_data_melted["Layer"].str.contains(r"\.w_q|weights?|_w")]
        a_data = bit_data_melted[bit_data_melted["Layer"].str.contains(r"\.a_q|activations?|_a")]

        def plot_final_bits_per_layer(data, title_suffix, filename_suffix):
            if data.empty:
                return
            
            # Use the final bit-width value (from the last logged step)
            # Sort by step and take the last entry for each layer
            layer_values = data.sort_values("_step").groupby("Layer")[["Bits"]].last().reset_index()
            
            # Clean up x-axis labels to keep only unique identifiers
            def clean_name(name):
                n = name
                # Remove common root prefixes
                for prefix in ["model.", "backbone.", "features.", "encoder.", "decoder.", "vit."]:
                     if n.startswith(prefix):
                         n = n[len(prefix):]
                
                # Remove quantization suffixes (since title/filename indicates weights/activations)
                for suffix in [".w_q", "_w", ".a_q", "_a"]:
                    if n.endswith(suffix):
                        n = n[:-len(suffix)]

                # Shorten structural names
                n = n.replace("block.", "b")
                n = n.replace("layer.", "l")
                n = n.replace("SelfAttention", "SA")
                n = n.replace("DenseReluDense", "FFN")
                n = n.replace("CrossAttention", "CA")
                n = n.replace("encoder", "enc")
                n = n.replace("attention", "attn")
                n = n.replace("vit", "")
                
                # Clean up repeated dots or leading/trailing dots
                n = n.strip(".")
                return n

            layer_values["CleanLayer"] = layer_values["Layer"].apply(clean_name)
            layer_values = layer_values.sort_values("Layer") 
            
            plt.figure(figsize=(max(8, len(layer_values)*0.4), 7))
            
            # Color bars by bit depth
            norm = plt.Normalize(2, 16)
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
            sm.set_array([])
            palette = {l: sm.to_rgba(b) for l, b in zip(layer_values["CleanLayer"], layer_values["Bits"])}

            sns.barplot(data=layer_values, x="CleanLayer", y="Bits", palette=palette, hue="CleanLayer", legend=False)
            
            plt.title(f"Final Bit-Width ({title_suffix}) - {run.name}")
            plt.xlabel("Layer")
            plt.ylabel("Bits")
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.ylim(0, 17)
            
            cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
            cbar.set_label('Bits')

            for i, v in enumerate(layer_values["Bits"]):
                plt.text(i, v + 0.2, f"{v:.1f}", ha='center', va='bottom', fontsize=9, rotation=90)
                
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{run.name}_final_bits_bar_{filename_suffix}.png")
            plt.close()

        if not w_data.empty and not a_data.empty:
            plot_final_bits_per_layer(w_data, "Weights", "weights")
            plot_final_bits_per_layer(a_data, "Activations", "activations")
        else:
            plot_final_bits_per_layer(bit_data_melted, "All", "all")

        print(f"Saved bar plots for {run.name}")

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
    parser.add_argument("--plot-file", type=str, default="plot_configs.yaml", help="YAML file specifying which plots to generate")
    args = parser.parse_args()

    # # Try to guess entity if not provided? user needs to provide it or configure wandb
    if args.entity == "user_entity_here":
         # try to get default entity
         try:
             args.entity = wandb.Api().default_entity
         except:
             pass

    with open(args.plot_file, "r") as f:
        configs = yaml.safe_load(f)
        # print("Loaded plot configurations:", configs)

    for name, config in configs.items():
        print(config)
        if len(config['runs']) > 1:
            print("\nGenerating Specified Comparison Plots...")
            os.makedirs(config.get("output_dir", "plot_comparisons"), exist_ok=True)
            comparison(args.entity, args.project, plot_name=name, plot_config=config['plot_config'], run_names=config["runs"], labels=config.get("labels", {}), plot_key=config["plot_key"], run_id=config.get("run_id", None), output_dir=config.get("output_dir", "plot_comparisons"))
        else:
            print("\nGenerating Plots for Individual Runs...")
            visualize(args.entity, args.project)
        # if not args.run_id:
        #     print("\nGenerating Comparison Plots...")
        # visualize_comparison(args.entity, args.project)