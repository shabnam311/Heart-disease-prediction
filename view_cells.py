import json
import sys

def main():
    try:
        with open('d:/Heart-disease-prediction/ADS_project.ipynb', 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = "".join(cell.get('source', []))
                print(f"Cell {i} (Code):")
                print(source[:100] + "..." if len(source) > 100 else source)
                print("-" * 40)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
