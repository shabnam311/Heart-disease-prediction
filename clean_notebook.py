import json
import sys
import copy

def main():
    log = open('clean_log.txt', 'w', encoding='utf-8')
    try:
        with open('d:/Heart-disease-prediction/ADS_project.ipynb', 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        cells = nb.get('cells', [])
        new_cells = []
        
        for i, cell in enumerate(cells):
            source = "".join(cell.get('source', []))
            
            # 1. Delete Cell 3 (the broken harmonize cell). It conflicts with Cell 4.
            if "Cell 3 — Harmonize Features" in source:
                log.write(f"Skipping cell {i} (Cell 3 — Harmonize Features)\n")
                continue
                
            # 2. Delete the !pip install opacus cell at the end.
            if "!pip install opacus" in source or "!pip install -q opacus" in source:
                log.write(f"Skipping cell {i} (!pip install opacus)\n")
                continue
                
            # 3. Fix Cell 8's main block.
            if "if __name__ == \"__main__\" and 'processed_dfs' in globals():" in source:
                log.write(f"Modifying cell {i} (Cell 8 main block)\n")
                new_source = []
                for line in cell['source']:
                    if "if __name__ == \"__main__\" and 'processed_dfs' in globals():" in line:
                        new_source.append(line.replace("if __name__ == \"__main__\" and 'processed_dfs' in globals():", "if 'processed_dfs' in globals():"))
                    elif "learner = UltraAggressiveFederatedLearner(processed_dfs, num_rounds=15)" in line:
                        new_source.append(line)
                    elif "personalized_models = learner.run()" in line:
                        new_source.append(line)
                        # We also need to add the else block as requested:
                        new_source.append("else:\n")
                        new_source.append("    print(\"Run Cell 7/7.5 first to load processed_dfs\")\n")
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                
            new_cells.append(cell)
            
        # 4. Add a model saving cell at the end.
        log.write("Adding saving cell at the end\n")
        saving_cell_source = [
            "import pickle, os\n",
            "save_path = '/content/drive/MyDrive/federated_models'\n",
            "os.makedirs(save_path, exist_ok=True)\n",
            "\n",
            "for server_name, result in personalized_models.items():\n",
            "    model_path = os.path.join(save_path, f'{server_name}_model.pkl')\n",
            "    with open(model_path, 'wb') as f:\n",
            "        pickle.dump({'model': result['model'], 'scaler': result['scaler'], 'engineer': result['engineer']}, f)\n",
            "    print(f\"✓ Saved {server_name} → {model_path}\")\n",
            "\n",
            "print(\"All models saved to Drive.\")\n"
        ]
        saving_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": saving_cell_source
        }
        new_cells.append(saving_cell)

        nb['cells'] = new_cells
        
        with open('d:/Heart-disease-prediction/ADS_project_clean.ipynb', 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
            
        log.write("Successfully created ADS_project_clean.ipynb\n")

    except Exception as e:
        log.write(f"Error: {e}\n")
    log.close()

if __name__ == "__main__":
    main()
