import pandas as pd
import json
from pathlib import Path

def merge_data():
    base_dir = Path(__file__).parent

    # Load parquet
    parquet_path = base_dir / "combined_summary_result.parquet"
    print(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  Rows: {len(df)}")

    # Load train json
    train_path = base_dir / "new_dataset_train_3000.json"
    print(f"Loading train JSON: {train_path}")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print(f"  Items: {len(train_data)}")

    # Load eval json
    eval_path = base_dir / "new_dataset_eval_3000.json"
    print(f"Loading eval JSON: {eval_path}")
    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    print(f"  Items: {len(eval_data)}")

    # Build lookup from parquet: addr_input (matches JSON 'input') -> row
    addr_to_row = {row['addr_input']: row for _, row in df.iterrows()}

    # Merge train data
    print("\nMerging train data...")
    merged_train = []
    matched = 0
    unmatched = 0

    for item in train_data:
        addr = item.get('input', '')
        if addr in addr_to_row:
            row = addr_to_row[addr]
            merged_item = {
                'input': addr,
                'output': {
                    'no': str(row.get('no', '') or ''),
                    'moo': str(row.get('moo', '') or ''),
                    'mooban': str(row.get('mooban', '') or ''),
                    'building': str(row.get('building', '') or ''),
                    'floor': str(row.get('floor', '') or ''),
                    'room': str(row.get('room', '') or ''),
                    'soi': str(row.get('soi', '') or ''),
                    'road': str(row.get('road', '') or ''),
                    'tambol': str(row.get('tambol', '') or ''),
                    'amphoe': str(row.get('amphoe', '') or ''),
                    'province': str(row.get('province', '') or ''),
                }
            }
            merged_train.append(merged_item)
            matched += 1
        else:
            merged_train.append(item)  # keep original if not matched
            unmatched += 1

    print(f"  Matched: {matched}, Unmatched: {unmatched}")

    # Merge eval data
    print("\nMerging eval data...")
    merged_eval = []
    matched = 0
    unmatched = 0

    for item in eval_data:
        addr = item.get('input', '')
        if addr in addr_to_row:
            row = addr_to_row[addr]
            merged_item = {
                'input': addr,
                'output': {
                    'no': str(row.get('no', '') or ''),
                    'moo': str(row.get('moo', '') or ''),
                    'mooban': str(row.get('mooban', '') or ''),
                    'building': str(row.get('building', '') or ''),
                    'floor': str(row.get('floor', '') or ''),
                    'room': str(row.get('room', '') or ''),
                    'soi': str(row.get('soi', '') or ''),
                    'road': str(row.get('road', '') or ''),
                    'tambol': str(row.get('tambol', '') or ''),
                    'amphoe': str(row.get('amphoe', '') or ''),
                    'province': str(row.get('province', '') or ''),
                }
            }
            merged_eval.append(merged_item)
            matched += 1
        else:
            merged_eval.append(item)  # keep original if not matched
            unmatched += 1

    print(f"  Matched: {matched}, Unmatched: {unmatched}")

    # Combine train + eval into single file
    combined = merged_train + merged_eval
    print(f"\nTotal combined items: {len(combined)}")

    # Save as single file
    output_file = base_dir / "data_address.json"
    print(f"Saving data_address.json...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print("\nDone!")
    return combined

if __name__ == "__main__":
    merge_data()
