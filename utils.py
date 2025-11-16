import os
import json
import math
import random
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class SimpleEmbed:
    def __init__(self):
        self.map = {}
        self.next_id = 0

    def get(self, key):
        if key not in self.map:
            self.map[key] = self.next_id
            self.next_id += 1
        return self.map[key]
