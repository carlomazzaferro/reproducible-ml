import os

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir) # This is your Project Root
CONFIG_PATH = os.path.join(ROOT_DIR, 'configuration.conf')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
RESUTLS_DIR = os.path.join(ROOT_DIR, 'reports')