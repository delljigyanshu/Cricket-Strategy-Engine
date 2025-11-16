import numpy as np
import json
import os

class EmpiricalSimulator:
    def __init__(self, df, empirical_json_path, mappings_path='processed/mappings.json'):
        self.df = df
        with open(empirical_json_path, 'r', encoding='utf-8') as f:
            self.emp = json.load(f)

        if os.path.exists(mappings_path):
            with open(mappings_path, 'r', encoding='utf-8') as f:
                m = json.load(f)
            self.bowler_map = m.get('bowler_map', {})
            self.batsman_list = m.get('batsman_list', [])
        else:
            detected = set(k.split('||')[1] for k in self.emp.keys() if '||' in k)
            self.batsman_list = list(self.df['batsman'].dropna().unique()[:12])
            self.bowler_map = {f"bowler_{i}": b for i,b in enumerate(list(detected)[:12])}

        self.reset_match()

    def reset_match(self, batting_team_name='TeamA', batting_order=None):
        self.score = 0
        self.wickets = 0
        self.balls_bowled = 0
        self.over = 0
        self.ball = 0
        self.done = False

        if batting_order and isinstance(batting_order, list) and len(batting_order) >= 2:
            self.batting_order = batting_order[:]
        else:
            self.batting_order = list(self.batsman_list[:]) if self.batsman_list else ['Batsman_1','Batsman_2','Batsman_3']

        self.current_batsman = self.batting_order[0] if len(self.batting_order)>0 else 'Batsman_1'
        self.non_striker = self.batting_order[1] if len(self.batting_order)>1 else 'Batsman_2'
        self.next_bat_idx = 2
        self.batsman_scores = {p: 0 for p in self.batting_order}
        self.batting_team = batting_team_name

    def _current_state(self):
        return {
            'score': self.score,
            'wickets': self.wickets,
            'balls_bowled': self.balls_bowled,
            'over': self.over,
            'ball': self.ball,
            'phase': self._phase(),
            'current_batsman': self.current_batsman,
            'non_striker': self.non_striker
        }

    def _phase(self):
        if self.over < 6:
            return 'powerplay'
        elif self.over < 16:
            return 'middle'
        else:
            return 'death'

    def sample_ball(self, bowler_name):
        real_bowler = bowler_name
        if isinstance(bowler_name, str) and bowler_name.startswith('bowler_'):
            real_bowler = self.bowler_map.get(bowler_name, None)
            if real_bowler is None:
                try:
                    idx = int(bowler_name.split('_')[1])
                    keys = list(self.bowler_map.values())
                    if 0 <= idx < len(keys):
                        real_bowler = keys[idx]
                except Exception:
                    real_bowler = None

        if not real_bowler:
            real_bowler = bowler_name

        key = f"{self._phase()}||{real_bowler}"

        if key not in self.emp:
            runs = np.random.choice([0,1,2,3,4,6], p=[0.55,0.25,0.10,0.03,0.06,0.01])
            wicket = np.random.rand() < 0.03
            return runs, wicket, real_bowler

        entry = self.emp[key]
        probs = entry.get('probs_runs', None)
        if not probs or sum(probs) == 0:
            runs = np.random.choice([0,1,2,3,4,6])
        else:
            runs = int(np.random.choice(range(len(probs)), p=probs))
        wicket = np.random.rand() < entry.get('wicket_prob', 0.03)
        return runs, wicket, real_bowler

    def step(self, action):
        """
        action = {
            'bowler': 'bowler_1' or index, 
            'batting_intent': 'defensive'/'normal'/'aggressive'
        }
        or action can be [bowler_idx, intent_idx]
        """
        if isinstance(action, (list, tuple, np.ndarray)):
            bowler_idx = int(action[0])
            intent_idx = int(action[1]) if len(action)>1 else 1
            bowler_name = f"bowler_{bowler_idx}"
            intent_map = {0:'defensive',1:'normal',2:'aggressive'}
            intent = intent_map.get(intent_idx, 'normal')
        elif isinstance(action, dict):
            bowler_name = action.get('bowler', 'bowler_0')
            intent = action.get('batting_intent', 'normal')
        else:
            bowler_name = str(action)
            intent = 'normal'

        runs, wicket, real_bowler = self.sample_ball(bowler_name)

        if intent == 'aggressive':
            if np.random.rand() < 0.15:
                runs = min(6, runs + np.random.choice([0,1,2]))
            if np.random.rand() < 0.03:
                wicket = True
        elif intent == 'defensive':
            if np.random.rand() < 0.6:
                runs = max(0, runs - 1)

        striker = self.current_batsman
        self.batsman_scores.setdefault(striker, 0)
        self.batsman_scores[striker] += runs

        self.score += runs
        if wicket:
            self.wickets += 1
            if self.next_bat_idx < len(self.batting_order):
                new_batsman = self.batting_order[self.next_bat_idx]
                self.next_bat_idx += 1
            else:
                new_batsman = f"sub_{self.next_bat_idx}"
                self.next_bat_idx += 1
            self.current_batsman = new_batsman
            self.batsman_scores.setdefault(new_batsman, 0)
        else:
            if runs % 2 == 1:
                self.current_batsman, self.non_striker = self.non_striker, self.current_batsman

        self.balls_bowled += 1
        self.over = self.balls_bowled // 6
        self.ball = self.balls_bowled % 6

        if self.balls_bowled >= 120 or self.wickets >= 10:
            self.done = True

        return {
            'runs': runs,
            'wicket': wicket,
            'bowler': real_bowler,
            'batsman': striker,
            'batting_intent': intent,
            'next_state': self._current_state(),
            'match_end': self.done
        }
