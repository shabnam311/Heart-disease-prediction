import json
import os

def convert_to_script():
    try:
        with open('d:/Heart-disease-prediction/ADS_project_clean.ipynb', 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        script_lines = []
        for cell in nb.get('cells', []):
            if cell['cell_type'] == 'code':
                for line in cell.get('source', []):
                    # Filter out colab commands
                    if line.strip().startswith('!'):
                        continue
                    # Modify the drive paths to local paths
                    if '/content/drive/MyDrive/dsa/' in line:
                        line = line.replace('/content/drive/MyDrive/dsa/', 'd:/Heart-disease-prediction/data/')
                    # Also replace the model save path
                    if '/content/drive/MyDrive/federated_models' in line:
                        line = line.replace('/content/drive/MyDrive/federated_models', 'd:/Heart-disease-prediction/federated_models')
                        
                    script_lines.append(line)
                script_lines.append("\n\n")
                
        with open('d:/Heart-disease-prediction/train_local.py', 'w', encoding='utf-8') as f:
            f.writelines(script_lines)
            
        print("Successfully converted to train_local.py")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    convert_to_script()
