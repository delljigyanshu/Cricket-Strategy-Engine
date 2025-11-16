import pandas as pd
import json
import os

PARQUET = 'processed/deliveries_processed.parquet'
OUT = 'processed'
TOP_N = 12   

def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(PARQUET)
   
    bowler_counts = df['bowler'].value_counts().index.tolist()
    top_bowlers = bowler_counts[:TOP_N]

    batsman_counts = df['batsman'].value_counts().index.tolist()
    top_batsmen = batsman_counts[:TOP_N]

    bowler_map = {f"bowler_{i}": name for i, name in enumerate(top_bowlers)}
    
    batsman_list = top_batsmen

    mappings = {
        "bowler_map": bowler_map,
        "batsman_list": batsman_list
    }

    with open(os.path.join(OUT, 'mappings.json'), 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)

    print("Saved mappings:", os.path.join(OUT, 'mappings.json'))
    print("Example bowler mapping:", bowler_map)

if __name__ == '__main__':
    main()
