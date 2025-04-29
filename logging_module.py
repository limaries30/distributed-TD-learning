import numpy as np  
import os

class Logger:
    def __init__(self, save_dir):
    
        self.save_dir= save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.logs = {}
        
    def add(self, key, value):
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append(value)
    
    def save_logs(self):
        for key, values in self.logs.items():
          np.save(os.path.join(self.save_dir, f"{key}.npy"), np.array(values))
          