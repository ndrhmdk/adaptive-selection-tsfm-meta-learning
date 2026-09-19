import json

from src.data.loader import load_dataset
from src.data.validation import validate_dataset

def main():
    dataset = load_dataset('ETTh1')
    report = validate_dataset(dataset)
    
    print("--- Dataset Validation ---------------------")
    
    for key, value in report.items():
        print(f"{key:20}: {value}")
        
    print(f"\n- First 5 observations:\n{dataset.values.head()}")
    print(f"\n- Last 5 observations:\n{dataset.values.tail()}")
    print(f"\n- Columns:\n{dataset.target_columns}")
    print(f"\n- Primary target:\n{dataset.primary_target}")
    
    output_path = 'data/metadata/ETTh1.json'
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)
    print(f"\nMetdata saved to: {output_path}")
    
if __name__ == "__main__":
    main()