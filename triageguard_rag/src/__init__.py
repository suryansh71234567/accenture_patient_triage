import logging
import os
import yaml

def init_logger():
    # Load logging level from config if available
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    level = logging.INFO
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
                lvl_str = cfg.get('logging', {}).get('level', 'INFO').upper()
                level = getattr(logging, lvl_str, logging.INFO)
        except Exception:
            pass
    logging.basicConfig(level=level, format='[%(asctime)s] %(levelname)s: %(message)s')
    return logging.getLogger('triageguard_rag')

logger = init_logger()
