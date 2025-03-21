import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORGANIZATION")
)

# Language dictionary with all seven languages
languages = {
    "en": {"name": "english", "verb": "verb", "adj": "adjective"},
    "es": {"name": "spanish", "verb": "verbo", "adj": "adjetivo"},
    "de": {"name": "german", "verb": "Verb", "adj": "Adjektiv"},
    "fr": {"name": "french", "verb": "verbe", "adj": "adjectif"},
    "ja": {"name": "japanese", "verb": "動詞 (dōshi)", "adj": "形容詞 (keiyōshi)"},
    "ko": {"name": "korean", "verb": "동사 (dongsa)", "adj": "형용사 (hyeongyongsa)"},
    "zh": {"name": "chinese", "verb": "动词 (dòngcí)", "adj": "形容词 (xíngróngcí)"},
}

# Language-specific prompts
prompts = {
    "en": (
        "For the following list of sentences in English, replace exactly {n} word(s) in each sentence that are either a verb (action word) or an adjective (quality word, not in adverbial form like 'brightly') with {replacement_type}. "
        "Do not replace adverbs (e.g., 'brightly', 'quickly'), nouns (e.g., 'season', 'seat'), or particles, connectors, or auxiliary words. Replace only a verb or adjective, keeping sentence structure and tense unchanged. "
        "Do not use negation (e.g., 'not') for antonyms; use a direct opposite word instead. Ensure synonyms preserve the sentence’s core meaning. Examples:\n"
        "- 'The sun shines brightly' → 'The sun glows brightly' (synonym, verb) or 'The sun dims brightly' (antonym, verb).\n"
        "- 'The sky is blue today' → 'The sky is azure today' (synonym, adjective) or 'The sky is gray today' (antonym, adjective).\n"
        "Sentences:\n{numbered_sentences}\n"
        "Return only the modified sentences as a numbered list in the same order, no extra text or explanations. If no suitable verb or adjective exists, return the original sentence unchanged."
    ),
    "es": (
        "For the following list of sentences in Spanish, replace exactly {n} word(s) in each sentence that are either a verbo (action word) or an adjetivo (quality word, not in adverbial form like 'brillantemente') with {replacement_type}. "
        "Do not replace adverbs (e.g., 'brillantemente', 'rápidamente'), nouns (e.g., 'temporada', 'asiento'), or particles, connectors, or auxiliary words. Replace only a verbo or adjetivo, keeping sentence structure and tense unchanged. "
        "Do not use negation (e.g., 'no') for antonyms; use a direct opposite word instead. Ensure synonyms preserve the sentence’s core meaning. Examples:\n"
        "- 'El sol brilla intensamente' → 'El sol resplandece intensamente' (synonym, verbo) or 'El sol se apaga intensamente' (antonym, verbo).\n"
        "- 'El cielo es azul hoy' → 'El cielo es celeste hoy' (synonym, adjetivo) or 'El cielo es gris hoy' (antonym, adjetivo).\n"
        "Sentences:\n{numbered_sentences}\n"
        "Return only the modified sentences as a numbered list in the same order, no extra text or explanations. If no suitable verbo or adjetivo exists, return the original sentence unchanged."
    ),
    "de": (
        "For the following list of sentences in German, replace exactly {n} word(s) in each sentence that are either a Verb (action word) or an Adjektiv (quality word, not in adverbial form like 'hell') with {replacement_type}. "
        "Do not replace adverbs (e.g., 'hell', 'schnell'), nouns (e.g., 'Jahreszeit', 'Sitz'), or particles, connectors, or auxiliary words. Replace only a Verb or Adjektiv, keeping sentence structure and tense unchanged. "
        "Do not use negation (e.g., 'nicht') for antonyms; use a direct opposite word instead. Ensure synonyms preserve the sentence’s core meaning. Examples:\n"
        "- 'Die Sonne scheint hell' → 'Die Sonne leuchtet hell' (synonym, Verb) or 'Die Sonne verdunkelt hell' (antonym, Verb).\n"
        "- 'Der Himmel ist blau heute' → 'Der Himmel ist azurblau heute' (synonym, Adjektiv) or 'Der Himmel ist grau heute' (antonym, Adjektiv).\n"
        "Sentences:\n{numbered_sentences}\n"
        "Return only the modified sentences as a numbered list in the same order, no extra text or explanations. If no suitable Verb or Adjektiv exists, return the original sentence unchanged."
    ),
    "fr": (
        "For the following list of sentences in French, replace exactly {n} word(s) in each sentence that are either a verbe (action word) or an adjectif (quality word, not in adverbial form like 'intensément') with {replacement_type}. "
        "Do not replace adverbs (e.g., 'intensément', 'rapidement'), nouns (e.g., 'saison', 'siège'), or particles, connectors, or auxiliary words. Replace only a verbe or adjectif, keeping sentence structure and tense unchanged. "
        "Do not use negation (e.g., 'pas') for antonyms; use a direct opposite word instead. Ensure synonyms preserve the sentence’s core meaning. Examples:\n"
        "- 'Le soleil brille intensément' → 'Le soleil rayonne intensément' (synonym, verbe) or 'Le soleil s’éteint intensément' (antonym, verbe).\n"
        "- 'Le ciel est bleu aujourd’hui' → 'Le ciel est azur aujourd’hui' (synonym, adjectif) or 'Le ciel est gris aujourd’hui' (antonym, adjectif).\n"
        "Sentences:\n{numbered_sentences}\n"
        "Return only the modified sentences as a numbered list in the same order, no extra text or explanations. If no suitable verbe or adjectif exists, return the original sentence unchanged."
    ),
    "ja": (
        "For the following list of sentences in Japanese, replace exactly {n} word(s) in each sentence that are either a 動詞 (dōshi, action word) or a 形容詞 (keiyōshi, quality word, not in adverbial form like '明るく') with {replacement_type}. "
        "Do not replace adverbs (e.g., '明るく', '速く'), nouns (e.g., '季節', '席'), or particles, connectors, or auxiliary words. Replace only a 動詞 or 形容詞, keeping sentence structure and tense unchanged. "
        "Do not use negation (e.g., 'ない') for antonyms; use a direct opposite word instead. Ensure synonyms preserve the sentence’s core meaning. Examples:\n"
        "- '太陽が明るく輝く' → '太陽が明るく煌めく' (synonym, 動詞) or '太陽が明るく暗くなる' (antonym, 動詞).\n"
        "- '空が青い今日' → '空が蒼い今日' (synonym, 形容詞) or '空が灰色の今日' (antonym, 形容詞).\n"
        "Sentences:\n{numbered_sentences}\n"
        "Return only the modified sentences as a numbered list in the same order, no extra text or explanations. If no suitable 動詞 or 形容詞 exists, return the original sentence unchanged."
    ),
    "ko": (
        "For the following list of sentences in Korean, replace exactly {n} word(s) in each sentence that are either a 동사 (dongsa, action word) or a 형용사 (hyeongyongsa, quality word, not in adverbial form like '밝게') with {replacement_type}. "
        "Do not replace adverbs (e.g., '밝게', '빨리'), nouns (e.g., '시즌', '자리'), or particles, connectors, or auxiliary words. Replace only a 동사 or 형용사, keeping sentence structure and tense unchanged. "
        "Do not use negation (e.g., '아니었다') for antonyms; use a direct opposite word instead. Ensure synonyms preserve the sentence’s core meaning. Examples:\n"
        "- '태양이 밝게 빛난다' → '태양이 밝게 반짝인다' (synonym, 동사) or '태양이 밝게 어두워진다' (antonym, 동사).\n"
        "- '하늘이 파랗다 오늘' → '하늘이 푸르다 오늘' (synonym, 형용사) or '하늘이 회색이다 오늘' (antonym, 형용사).\n"
        "Sentences:\n{numbered_sentences}\n"
        "Return only the modified sentences as a numbered list in the same order, no extra text or explanations. If no suitable 동사 or 형용사 exists, return the original sentence unchanged."
    ),
    "zh": (
        "For the following list of sentences in Chinese, replace exactly {n} word(s) in each sentence that are either a 动词 (dòngcí, action word) or a 形容词 (xíngróngcí, quality word, not in adverbial form like '明亮地') with {replacement_type}. "
        "Do not replace adverbs (e.g., '明亮地', '快地'), nouns (e.g., '赛季', '座位'), or particles, connectors, or auxiliary words. Replace only a 动词 or 形容词, keeping sentence structure and tense unchanged. "
        "Do not use negation (e.g., '否') for antonyms; use a direct opposite word instead. Ensure synonyms preserve the sentence’s core meaning. Examples:\n"
        "- '太阳明亮地照耀' → '太阳明亮地普照' (synonym, 动词) or '太阳明亮地熄灭' (antonym, 动词).\n"
        "- '天空是蓝色的今天' → '天空是青色的今天' (synonym, 形容词) or '天空是灰色的今天' (antonym, 形容词).\n"
        "Sentences:\n{numbered_sentences}\n"
        "Return only the modified sentences as a numbered list in the same order, no extra text or explanations. If no suitable 动词 or 形容词 exists, return the original sentence unchanged."
    )
}

class WordReplacer:
    def __init__(self, language=languages["en"], llm_model="gpt-4o"):
        self.language = language
        self.client = client
        self.llm_model = llm_model
        if self.language not in languages.values():
            raise ValueError(f"Unsupported language: {self.language['name']}")

    def sentence_replacement(self, sentences, n, types=""):
        types = types.lower()
        if types not in ["synonyms", "antonyms"]:
            return sentences  # Return original list if type is invalid

        # Create a numbered list of sentences for the prompt
        numbered_sentences = [f"{i+1}. {sentence}" for i, sentence in enumerate(sentences)]
        sentence_list = "\n".join(numbered_sentences)

        # Determine replacement type
        replacement_type = "a synonym" if types == "synonyms" else "an antonym"

        # Use the dictionary key directly instead of deriving from name
        lang_code = [key for key, value in languages.items() if value == self.language][0]
        prompt_template = prompts[lang_code]
        prompt = prompt_template.format(n=n, numbered_sentences=sentence_list, replacement_type=replacement_type)

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1856  # 32 sentences (4096 context), adjustable
            )
            response_lines = response.choices[0].message.content.strip().split("\n")
            modified_sentences = [line.split(". ", 1)[1] if ". " in line else line for line in response_lines]
            if len(modified_sentences) != len(sentences):
                logger.error(f"Warning: Mismatched sentence count in {self.language['name']} response. Returning original sentences.")
                # return sentences
                exit(1)
            return modified_sentences
        except Exception as e:
            logger.error(f"Error with API for {self.language['name']}: {e}")
            # return sentences  # Fallback to original list
            exit(1)

