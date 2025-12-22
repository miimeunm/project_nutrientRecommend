# build_input.py

from survey_sentence_map import SURVEY_SENTENCE_MAP

def build_model_input(responses: dict) -> str:
    sentences = []

    for key, choice in responses.items():
        if key in SURVEY_SENTENCE_MAP:
            sentence = SURVEY_SENTENCE_MAP[key].get(choice)
            if sentence:
                sentences.append(sentence)

    return " ".join(sentences)
