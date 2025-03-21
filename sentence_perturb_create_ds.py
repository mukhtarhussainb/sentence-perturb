import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

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
    def __init__(self, language=languages["en"]):
        self.language = language
        self.client = client

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
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1856  # 32 sentences (4096 context), adjustable
            )
            response_lines = response.choices[0].message.content.strip().split("\n")
            modified_sentences = [line.split(". ", 1)[1] if ". " in line else line for line in response_lines]
            if len(modified_sentences) != len(sentences):
                print(f"Warning: Mismatched sentence count in {self.language['name']} response. Returning original sentences.")
                return sentences
            return modified_sentences
        except Exception as e:
            print(f"Error with API for {self.language['name']}: {e}")
            return sentences  # Fallback to original list


# if __name__ == "__main__":
#     all_sentences = {
#         "en": [
#             "The sun shines brightly over the fields.",
#             "Friends meet in the park every afternoon."
#         ],
#         "ko": [
#             "태양이 밝게 빛난다.",
#             "친구들이 공원에서 만난다."
#         ],
#         # Add other languages as needed
#     }

#     output_file = "sentence_replacements_gpt4o.xlsx"
#     with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
#         for lang_code, sentences in all_sentences.items():
#             replacer = WordReplacer(language=languages[lang_code])
#             synonyms = replacer.sentence_replacement(sentences, n=1, types="synonyms")
#             antonyms = replacer.sentence_replacement(sentences, n=1, types="antonyms")

#             df = pd.DataFrame({
#                 "Original Sentence": sentences,
#                 "Synonym Replacement": synonyms,
#                 "Antonym Replacement": antonyms
#             })

#             sheet_name = languages[lang_code]["name"].capitalize()
#             df.to_excel(writer, sheet_name=sheet_name, index=False)
#             print(f"Processed {sheet_name} sentences")

#     print(f"\nResults saved to {output_file}")
# import os
# from dotenv import load_dotenv
# from openai import OpenAI
# import pandas as pd

# load_dotenv()

# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     organization=os.getenv("OPENAI_ORGANIZATION")
# )

# # Language dictionary with all seven languages
# languages = {
#     "en": {"name": "english", "verb": "verb", "adj": "adjective"},
#     "es": {"name": "spanish", "verb": "verbo", "adj": "adjetivo"},
#     "de": {"name": "german", "verb": "Verb", "adj": "Adjektiv"},
#     "fr": {"name": "french", "verb": "verbe", "adj": "adjectif"},
#     "ja": {"name": "japanese", "verb": "動詞 (dōshi)", "adj": "形容詞 (keiyōshi)"},
#     "ko": {"name": "korean", "verb": "동사 (dongsa)", "adj": "형용사 (hyeongyongsa)"},
#     "zh": {"name": "chinese", "verb": "动词 (dòngcí)", "adj": "形容词 (xíngróngcí)"},
# }

# class WordReplacer:
#     def __init__(self, language=languages["en"]):
#         self.language = language
#         self.client = client

#     def sentence_replacement(self, sentences, n, types=""):
#         types = types.lower()
#         if types not in ["synonyms", "antonyms"]:
#             return sentences  # Return original list if type is invalid

#         # Create a numbered list of sentences for the prompt
#         numbered_sentences = [f"{i+1}. {sentence}" for i, sentence in enumerate(sentences)]
#         sentence_list = "\n".join(numbered_sentences)

#         # Language-specific adverb examples to avoid
#         adverb_examples = {
#             "en": "brightly", "es": "brillantemente", "de": "hell", "fr": "intensément",
#             "ja": "明るく (akaruku)", "ko": "밝게 (balge)", "zh": "明亮地 (míngliàngdì)"
#         }
#         lang_code = self.language["name"][0:2]  # e.g., "en" from "english"

#         prompt = (
#             f"For the following list of sentences in {self.language['name']}, replace exactly {n} word(s) in each sentence "
#             f"that are either a {self.language['verb']} (action word) or a {self.language['adj']} (quality word, not in adverbial form like '{adverb_examples[lang_code]}') "
#             f"with {'a synonym' if types == 'synonyms' else 'an antonym'}. "
#             f"Do not replace adverbs (e.g., '{adverb_examples[lang_code]}'), nouns (objects, people, places like 'season', 'seat', '시즌', '赛季', 'saison'), "
#             f"particles, connectors, or auxiliary words. Replace only a {self.language['verb']} or {self.language['adj']}, "
#             f"keeping sentence structure and tense unchanged. Do not use negation (e.g., 'not', '아니었다', '否', 'pas') for antonyms; use a direct opposite word instead. "
#             f"Ensure synonyms preserve the sentence’s core meaning. Examples:\n"
#             f"- English: 'The sun shines brightly' → 'The sun glows brightly' (synonym, verb) or 'The sun dims brightly' (antonym, verb).\n"
#             f"- Spanish: 'El sol brilla intensamente' → 'El sol resplandece intensamente' (synonym, verb) or 'El sol se apaga intensamente' (antonym, verb).\n"
#             f"- German: 'Die Sonne scheint hell' → 'Die Sonne leuchtet hell' (synonym, verb) or 'Die Sonne verdunkelt hell' (antonym, verb).\n"
#             f"- French: 'Le soleil brille intensément' → 'Le soleil rayonne intensément' (synonym, verb) or 'Le soleil s’éteint intensément' (antonym, verb).\n"
#             f"- Japanese: '太陽が明るく輝く' → '太陽が明るく煌めく' (synonym, verb) or '太陽が明るく暗くなる' (antonym, verb).\n"
#             f"- Korean: '태양이 밝게 빛난다' → '태양이 밝게 반짝인다' (synonym, verb) or '태양이 밝게 어두워진다' (antonym, verb).\n"
#             f"- Chinese: '太阳明亮地照耀' → '太阳明亮地普照' (synonym, verb) or '太阳明亮地熄灭' (antonym, verb).\n"
#             f"Sentences:\n{sentence_list}\n"
#             f"Return only the modified sentences as a numbered list in the same order, no extra text or explanations. "
#             f"If no suitable {self.language['verb']} or {self.language['adj']} exists, return the original sentence unchanged."
#         )
#         try:
#             response = self.client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.3,
#                 max_tokens=1856  # 32 sentences (4096 context), adjustable
#             )
#             # Split the response into lines and extract the modified sentences
#             response_lines = response.choices[0].message.content.strip().split("\n")
#             # Remove the numbering (e.g., "1. ") from each line
#             modified_sentences = [line.split(". ", 1)[1] if ". " in line else line for line in response_lines]
#             # Ensure the output length matches input to avoid index errors
#             if len(modified_sentences) != len(sentences):
#                 print(f"Warning: Mismatched sentence count in {self.language['name']} response. Returning original sentences.")
#                 raise ValueError("Mismatched sentence count")
#                 # return sentences
#             return modified_sentences
#         except Exception as e:
#             print(f"Error with API for {self.language['name']}: {e}")
#             raise ValueError("API error")  # Fallback to original list
#             # return sentences  # Fallback to original list

# if __name__ == "__main__":
#     # Example usage with your sentences
#     all_sentences = {
#         "en": [
#             "The sun shines brightly over the fields.",
#             "Friends meet in the park every afternoon."
#         ],
#         "ko": [
#             "태양이 밝게 빛난다.",
#             "친구들이 공원에서 만난다."
#         ],
#         # Add other languages as needed
#     }

#     output_file = "sentence_replacements_gpt4o.xlsx"
#     with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
#         for lang_code, sentences in all_sentences.items():
#             replacer = WordReplacer(language=languages[lang_code])
#             synonyms = replacer.sentence_replacement(sentences, n=1, types="synonyms")
#             antonyms = replacer.sentence_replacement(sentences, n=1, types="antonyms")

#             df = pd.DataFrame({
#                 "Original Sentence": sentences,
#                 "Synonym Replacement": synonyms,
#                 "Antonym Replacement": antonyms
#             })

#             sheet_name = languages[lang_code]["name"].capitalize()
#             df.to_excel(writer, sheet_name=sheet_name, index=False)
#             print(f"Processed {sheet_name} sentences")

#     print(f"\nResults saved to {output_file}")

# import os
# from dotenv import load_dotenv
# from openai import OpenAI
# import pandas as pd

# load_dotenv()
# # Point to the local server
# # client = OpenAI(base_url="http://localhost:8000/v1",
# #                 api_key="sk-no-key-required")  # Dummy key for local use

# client = OpenAI(
# api_key=os.getenv("OPENAI_API_KEY"),
# organization=os.getenv("OPENAI_ORGANIZATION")
# )

# # Language dictionary with all seven languages
# languages = {
#     "en": {"name": "english", "verb": "verb", "adj": "adjective"},
#     "es": {"name": "spanish", "verb": "verbo", "adj": "adjetivo"},
#     "de": {"name": "german", "verb": "Verb", "adj": "Adjektiv"},
#     "fr": {"name": "french", "verb": "verbe", "adj": "adjectif"},
#     "ja": {"name": "japanese", "verb": "動詞 (dōshi)", "adj": "形容詞 (keiyōshi)"},
#     "ko": {"name": "korean", "verb": "동사 (dongsa)", "adj": "형용사 (hyeongyongsa)"},
#     "zh": {"name": "chinese", "verb": "动词 (dòngcí)", "adj": "形容词 (xíngróngcí)"},
# }

# class WordReplacer:
#     def __init__(self, language=languages["en"]):
#         self.language = language
#         self.client = client

#     def sentence_replacement(self, sentences, n, types=""):
#         types = types.lower()
#         if types not in ["synonyms", "antonyms"]:
#             return sentences  # Return original list if type is invalid

#         # Create a numbered list of sentences for the prompt
#         numbered_sentences = [f"{i+1}. {sentence}" for i, sentence in enumerate(sentences)]
#         sentence_list = "\n".join(numbered_sentences)

#         prompt = (
#             f"For the following list of sentences in {self.language['name']}, replace {n} word(s) in each sentence "
#             f"that are either a {self.language['verb']} (action word) or a {self.language['adj']} (quality word, not in adverbial form like 'brightly') "
#             f"with {'a synonym' if types == 'synonyms' else 'an antonym'}. "
#             f"Do not replace adverbs (e.g., 'quickly,' '明るく,' '明亮地'), even if derived from adjectives, "
#             f"nouns (objects, people, places), or particles, connectors, or auxiliary words. "
#             f"Replace only the target {self.language['verb']} or {self.language['adj']}, keeping the sentence structure and tense unchanged. "
#             f"Do not use negation (e.g., 'not') for antonyms; use a direct opposite word instead. "
#             f"Ensure the replacement preserves the sentence’s core meaning for synonyms. "
#             f"For example: "
#             f"- In English 'The sun shines brightly', replace 'shines' (verb) with 'glows', not 'brightly' (adverb). "
#             f"- In Japanese '太陽が明るく輝く', replace '輝く' (動詞) with '煌めく', not '明るく' (adverb). "
#             f"- In Korean '친구들이 공원에서 만난다', replace '만난다' (동사) with '모인다' (gather), not '보낸다' (spend time). "
#             f"- In Chinese '太阳明亮地照耀着田野', replace '照耀' (动词) with '普照' or '明亮' (形容词) with '昏暗', not '明亮地' (adverb). "
#             f"Here are the sentences:\n{sentence_list}\n\n"
#             f"Return only the modified sentences as a numbered list in the same order, with no extra text or explanations."
#         )
#         try:
#             response = self.client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.3,
#                 max_tokens=1856  # 32 sentences (4096 context)
#             )
#             # Split the response into lines and extract the modified sentences
#             response_lines = response.choices[0].message.content.strip().split("\n")
#             # Remove the numbering (e.g., "1. ") from each line
#             modified_sentences = [line.split(". ", 1)[1] if ". " in line else line for line in response_lines]
#             return modified_sentences
#         except Exception as e:
#             print(f"Error with API for {self.language['name']}: {e}")
#             return sentences  # Fallback to original list

# if __name__ == "__main__":
#     # Define sentences for all languages
#     all_sentences = {
#         "en": [
#             "The sun shines brightly over the fields.",
#             "Friends meet in the park every afternoon."
#         ],
#         "es": [
#             "El sol brilla intensamente sobre los campos.",
#             "Los amigos se reúnen en el parque por la tarde."
#         ],
#         "de": [
#             "Die Sonne scheint hell über den Feldern.",
#             "Freunde treffen sich im Park am Nachmittag."
#         ],
#         "fr": [
#             "Le soleil brille intensément sur les champs.",
#             "Les amis se retrouvent dans le parc l’après-midi."
#         ],
#         "ja": [
#             "太陽が明るく輝く。",
#             "友達が公園で集まる。"
#         ],
#         "ko": [
#             "태양이 밝게 빛난다。",
#             "친구들이 공원에서 만난다。"
#         ],
#         "zh": [
#             "太阳明亮地照耀着田野。",
#             "朋友们在公园里聚会。"
#         ]
#     }

    # # Create an Excel writer object
    # output_file = "sentence_replacements_llama3.3-70b.xlsx"
    # with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    #     for lang_code, sentences in all_sentences.items():
    #         replacer = WordReplacer(language=languages[lang_code])
    #         # Generate synonyms and antonyms for the entire list in one call
    #         synonyms = replacer.sentence_replacement(sentences, n=1, types="synonyms")
    #         antonyms = replacer.sentence_replacement(sentences, n=1, types="antonyms")

    #         # Create DataFrame
    #         df = pd.DataFrame({
    #             "Original Sentence": sentences,
    #             "Synonym Replacement": synonyms,
    #             "Antonym Replacement": antonyms
    #         })

    #         # Write to Excel sheet named after the language
    #         sheet_name = languages[lang_code]["name"].capitalize()
    #         df.to_excel(writer, sheet_name=sheet_name, index=False)
    #         print(f"Processed {sheet_name} sentences")

    # print(f"\nResults saved to {output_file}")

# # Example: reuse your existing OpenAI setup
# import os
# from dotenv import load_dotenv
# from openai import OpenAI
# import pandas as pd

# load_dotenv()

# # Point to the local server
# client = OpenAI(base_url="http://localhost:8000/v1",
#     api_key="sk-no-key-required"  # Dummy key for local use
#     )

# # completion = client.chat.completions.create(
# #   model="deepseek-r1-distill-qwen-7b",
# #   messages=[
# #     {"role": "system", "content": "Always answer in rhymes."},
# #     {"role": "user", "content": "Introduce yourself."}
# #   ],
# #   temperature=0.7,
# # )

# # client = OpenAI(
# # api_key=os.getenv("OPENAI_API_KEY"),
# # organization=os.getenv("OPENAI_ORGANIZATION")
# # )
# # Language dictionary with all seven languages
# languages = {
#     "en": {"name": "english", "verb": "verb", "adj": "adjective"},
#     "es": {"name": "spanish", "verb": "verbo", "adj": "adjetivo"},
#     "de": {"name": "german", "verb": "Verb", "adj": "Adjektiv"},
#     "fr": {"name": "french", "verb": "verbe", "adj": "adjectif"},
#     "ja": {"name": "japanese", "verb": "動詞 (dōshi)", "adj": "形容詞 (keiyōshi)"},
#     "ko": {"name": "korean", "verb": "동사 (dongsa)", "adj": "형용사 (hyeongyongsa)"},
#     "zh": {"name": "chinese", "verb": "动词 (dòngcí)", "adj": "形容词 (xíngróngcí)"},
# }

# class WordReplacer:
#     def __init__(self, language=languages["en"]):
#         self.language = language
#         self.client = client

#     def sentence_replacement(self, words, n, types=""):
#         types = types.lower()
#         if types not in ["synonyms", "antonyms"]:
#             return words

#         prompt = (
#             f"In the sentence '{words}', replace {n} word(s) that are either a {self.language['verb']} (action word) "
#             f"or a {self.language['adj']} (quality word, not in adverbial form like 'brightly') "
#             f"with {'a synonym' if types == 'synonyms' else 'an antonym'} in {self.language['name']}. "
#             f"Do not replace adverbs (e.g., 'quickly,' '明るく,' '明亮地'), even if derived from adjectives, "
#             f"nouns (objects, people, places), or particles, connectors, or auxiliary words. "
#             f"Replace only the target {self.language['verb']} or {self.language['adj']}, keeping the sentence structure and tense unchanged. "
#             f"Do not use negation (e.g., 'not') for antonyms; use a direct opposite word instead. "
#             f"Ensure the replacement preserves the sentence’s core meaning for synonyms. "
#             f"For example: "
#             f"- In English 'The sun shines brightly', replace 'shines' (verb) with 'glows', not 'brightly' (adverb). "
#             f"- In Japanese '太陽が明るく輝く', replace '輝く' (動詞) with '煌めく', not '明るく' (adverb). "
#             f"- In Korean '친구들이 공원에서 만난다', replace '만난다' (동사) with '모인다' (gather), not '보낸다' (spend time). "
#             f"- In Chinese '太阳明亮地照耀着田野', replace '照耀' (动词) with '普照' or '明亮' (形容词) with '昏暗', not '明亮地' (adverb). "
#             f"Return only the modified sentence, no extra text."
#         )
#         try:
#             response = self.client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.3,
#                 max_tokens=100
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"Error with GPT-4 API for {self.language['name']}: {e}")
#             return words  # Fallback to original

# if __name__ == "__main__":
#     # Define sentences for all languages
#     all_sentences = {
#         "en": [
#             "The sun shines brightly over the fields.",
#             "Friends meet in the park every afternoon."
#         ],
#         # "es": [
#         #     "El sol brilla intensamente sobre los campos.",
#         #     "Los amigos se reúnen en el parque por la tarde."
#         # ],
#         # "de": [
#         #     "Die Sonne scheint hell über den Feldern.",
#         #     "Freunde treffen sich im Park am Nachmittag."
#         # ],
#         # "fr": [
#         #     "Le soleil brille intensément sur les champs.",
#         #     "Les amis se retrouvent dans le parc l’après-midi."
#         # ],
#         # "ja": [
#         #     "太陽が明るく輝く。",
#         #     "友達が公園で集まる。"
#         # ],
#         # "ko": [
#         #     "태양이 밝게 빛난다。",
#         #     "친구들이 공원에서 만난다。"
#         # ],
#         # "zh": [
#         #     "太阳明亮地照耀着田野。",
#         #     "朋友们在公园里聚会。"
#         # ]
#     }

#     # Create an Excel writer object
#     output_file = "sentence_replacements_chat_gpt-4.0.xlsx"
#     with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
#         for lang_code, sentences in all_sentences.items():
#             replacer = WordReplacer(language=languages[lang_code])
#             # Generate synonyms and antonyms
#             synonyms = [replacer.sentence_replacement(sentence, n=1, types="synonyms") for sentence in sentences]
#             antonyms = [replacer.sentence_replacement(sentence, n=1, types="antonyms") for sentence in sentences]

#             # Create DataFrame
#             df = pd.DataFrame({
#                 "Original Sentence": sentences,
#                 "Synonym Replacement": synonyms,
#                 "Antonym Replacement": antonyms
#             })

#             # Write to Excel sheet named after the language
#             sheet_name = languages[lang_code]["name"].capitalize()
#             df.to_excel(writer, sheet_name=sheet_name, index=False)
#             print(f"Processed {sheet_name} sentences")

#     print(f"\nResults saved to {output_file}")