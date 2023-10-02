import sys
from os import path

sys.path.append(path.join(path.dirname(path.abspath(__file__)), '..')) # Adds parent directory to find other repos next to this one
sys.path.append(path.dirname(path.abspath(__file__))) # Adds the repo directory to find modules within this repo