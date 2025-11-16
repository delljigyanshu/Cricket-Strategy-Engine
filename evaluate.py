import gymnasium as gym
import numpy as np

class CricketEnv(gym.Env):

    def __init__(self, simulator, max_balls=120):
        super().__init__()
        self.sim = simulator
        self.max_balls = max_balls

        self.observation_space = gym.spaces.Box(
            low=0, high=200,
            shape=(5 + 3 + 5,),
            dtype=np.float32
        )

        self.action_space = gym.spaces.MultiDiscrete([10, 3])

        self._current_over_bowler = None

        self.reset()
    def reset(self, *, seed=None, options=None):
   
        super().reset(seed=seed)

        self.sim.reset_match()
        self.last5 = [0] * 5
        self._current_over_bowler = None  

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        s = self.sim._current_state()

        over = s['over']
        ball = s['ball']
        score = s['score']
        wickets = s['wickets']
        balls_left = max(0, self.max_balls - s['balls_bowled'])

        phase = s['phase']
        phase_onehot = [
            1 if phase == 'powerplay' else 0,
            1 if phase == 'middle' else 0,
            1 if phase == 'death' else 0
        ]

        obs = [
            over, ball, score, wickets, balls_left
        ] + phase_onehot + self.last5

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """
        Enforce: one bowler per over.
        - If at start of an over (balls_bowled % 6 == 0) accept proposed bowler.
        - Otherwise keep current over bowler and ignore proposed mid-over changes.
        Returns (obs, reward, done, truncated, info) with info['outcome'] from simulator.
        """

        if isinstance(action, (list, tuple, np.ndarray)):
            bowler_idx = int(action[0])
            intent_idx = int(action[1]) if len(action) > 1 else 1
            proposed_bowler = f"bowler_{bowler_idx}"
            intent_map = {0: 'defensive', 1: 'normal', 2: 'aggressive'}
            intent = intent_map.get(intent_idx, 'normal')
        elif isinstance(action, dict):
            proposed_bowler = action.get('bowler', 'bowler_0')
            intent = action.get('batting_intent', 'normal')
        else:
            proposed_bowler = str(action)
            intent = 'normal'

        balls_bowled = getattr(self.sim, 'balls_bowled', 0)
        ball_in_over = balls_bowled % 6

        if ball_in_over == 0 or self._current_over_bowler is None:
            self._current_over_bowler = proposed_bowler
        else:
            proposed_bowler = self._current_over_bowler

        outcome = self.sim.step({
            'bowler': proposed_bowler,
            'batting_intent': intent
        })

        self.last5 = self.last5[1:] + [outcome.get('runs', 0)]

        obs = self._get_obs()
        reward = self._compute_reward(outcome)

        done = bool(outcome.get('match_end', False))
        truncated = False

        info = {
            'outcome': outcome,
            'bowler_used': self._current_over_bowler
        }

        return obs, reward, done, truncated, info

    def _compute_reward(self, outcome):
        r = -outcome.get('runs', 0)
        if outcome.get('wicket', False):
            r += 6
        if outcome.get('runs', 0) == 0:
            r += 1
        return r

    def render(self, mode='human'):
        s = self.sim._current_state()
        print(
            f"Over: {s['over']}.{s['ball']}   "
            f"Score: {s['score']}/{s['wickets']}   "
            f"Balls bowled: {s['balls_bowled']}   "
            f"Bowler this over: {self._current_over_bowler}"
        )
