import logging
from pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():

    dataset_name = "paws-x"
    lang_list = ["en", "es", "de", "fr", "ja", "ko", "zh"]
    # lang_list = ["es", "de", "fr", "ja", "ko", "zh"]
    # lang_list = ["ja", "ko", "zh","fr",]
    for lang in lang_list:
        logger.info(f"Processing language: {lang}")
        run_pipeline(dataset_name, lang, batch_size=32)
        logger.info(f"Finished processing language: {lang}")

if __name__ == "__main__":
    main()
