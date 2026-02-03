from langdetect import detect
from googletrans import Translator

translator = Translator()

def translate_to_english(text):
    try:
        lang = detect(text)
        if lang != 'en':
            translated = translator.translate(text, src=lang, dest='en')
            return lang, translated.text
        return lang, text
    except:
        return "unknown", text
