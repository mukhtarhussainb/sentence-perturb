import logging
# from config import MODELS_CONFIG
from pipeline import run_pipeline
# from uitls import load_models_config
import os

# from sentence_perturb_create_ds import WordReplacer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # model_config_file = "models_config.yaml"
    # model_config_file = os.path.join(os.path.dirname(__file__), model_config_file)
    # if not os.path.exists(model_config_file):
    #     logger.error(f"Model config file {model_config_file} does not exist")
    #     return
    # MODELS_CONFIG = load_models_config(model_config_file)
    # MODELS_CONFIG = config.get("MODELS_CONFIG")

    # if MODELS_CONFIG is None:
    #     logger.error("Failed to load models config")
    #     return
    dataset_name = "paws-x"
    # model_type = "classic"
    lang_list = ["en", "es", "de", "fr", "ja", "ko", "zh"]
    # lang_list = ["es", "de", "fr", "ja", "ko", "zh"]
    # lang_list = ["ja", "ko", "zh","fr",]
    for lang in lang_list:
        logger.info(f"Processing language: {lang}")
        # replacer = WordReplacer(language=lang)
        run_pipeline(dataset_name, lang)

if __name__ == "__main__":
    main()
