import pandas as pd
import os
import json
import argparse
from tqdm import tqdm

core_required = ['match_id','inning','over','ball','bowler','batsman_runs','date']


def prepare_data(input_csv, out_dir):

    print("Loading CSV:", input_csv)
    df = pd.read_csv(input_csv)

    df.rename(columns={
    'matchId': 'match_id',
    'over_ball': 'over_ball'
}, inplace=True)

    os.makedirs(out_dir, exist_ok=True)

    for c in core_required:
        if c not in df.columns:
            raise ValueError(f"Dataset missing essential column: {c}")

    if 'total_runs' not in df.columns:
        print("⚠ total_runs not found → creating from batsman_runs + extra_runs")
        if 'extra_runs' in df.columns:
            df['total_runs'] = df['batsman_runs'] + df['extra_runs']
        elif 'extras' in df.columns:
            df['total_runs'] = df['batsman_runs'] + df['extras']
        else:
            df['total_runs'] = df['batsman_runs']    

    if 'extras' not in df.columns:
        if 'extra_runs' in df.columns:
            df['extras'] = df['extra_runs']
        else:
            df['extras'] = 0

    if 'player_dismissed' not in df.columns:
        print("⚠ player_dismissed not found → filling with 'none'")
        df['player_dismissed'] = 'none'
    else:
        df['player_dismissed'] = df['player_dismissed'].fillna('none')

    if 'dismissal_kind' not in df.columns:
        print("⚠ dismissal_kind not found → filling with 'not_out'")
        df['dismissal_kind'] = 'not_out'
    else:
        df['dismissal_kind'] = df['dismissal_kind'].fillna('not_out')

    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    except:
        print("⚠ date format could not be parsed")

    df = df.sort_values(['match_id','inning','over','ball']).reset_index(drop=True)

    df['runs_cumulative'] = df.groupby(['match_id','inning'])['total_runs'].cumsum()
    df['wicket'] = df['player_dismissed'].apply(lambda x: x != 'none')

    df['over_ball'] = df['over'].astype(str) + '.' + df['ball'].astype(str)

    df['balls_bowled'] = df.groupby(['match_id','inning']).cumcount() + 1
    df['balls_left'] = 120 - df['balls_bowled']


    parquet_path = os.path.join(out_dir, 'deliveries_processed.parquet')
    df.to_parquet(parquet_path, index=False)
    print("Saved processed deliveries →", parquet_path)

    def over_phase(over):
        if over <= 6:
            return 'powerplay'
        elif over <= 15:
            return 'middle'
        else:
            return 'death'

    df['phase'] = df['over'].apply(over_phase)
    df['runs_bucket'] = df['total_runs'].clip(0, 6)

    emp = {}
    print("Building empirical probability tables...")
    grp = df.groupby(['phase', 'bowler'])

    for (phase, bowler), g in tqdm(grp):
        key = f"{phase}||{bowler}"

        counts = g['runs_bucket'].value_counts().to_dict()
        total = g.shape[0]

        probs = [counts.get(i, 0) / total for i in range(7)]
        wicket_prob = g['wicket'].mean()

        emp[key] = {
            'probs_runs': probs,
            'wicket_prob': float(wicket_prob),
            'sample_count': int(total)
        }

    json_path = os.path.join(out_dir, 'empirical_tables.json')
    with open(json_path, 'w') as f:
        json.dump(emp, f, indent=4)

    print("Saved empirical tables →", json_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/deliveries.csv')
    parser.add_argument('--out', default='processed')
    args = parser.parse_args()

    prepare_data(args.input, args.out)
