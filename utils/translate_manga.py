"""
This module is used to translate manga from one language to another.
"""
import os
from deep_translator import GoogleTranslator, MicrosoftTranslator
# from ollama import chat
# from ollama import ChatResponse


def translate_manga(text: str, source_lang: str = "auto", target_lang: str = "pt") -> str:
    """
    Translate manga from one language to another.
    """

    if source_lang == target_lang:
        return text

    translated_text = MicrosoftTranslator(api_key=os.environ['MICROSOFT_API_KEY'],
        target=target_lang, region='brazilsouth').translate(text)
    print("Original text:", text)
    print("Translated text:", translated_text)

    return translated_text

def improve_text(translated_text):
    response: ChatResponse = chat(model='dorian2b/vera', messages=[
    {
        'role': 'user',
        'content': f"Please improve the following sexual content text for better coherence and clarity:\n{translated_text}",
    },
    ])

    improved_text = response.message.content

    return improved_text