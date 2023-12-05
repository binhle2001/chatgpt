from keras.models import load_model
model = load_model('ai_core/model_add_accent/a_best_weight.h5')

from collections import Counter

import numpy as np

from utils import *
import string
import re

alphabet = set('\x00 _' + string.ascii_lowercase + string.digits + ''.join(ACCENTED_TO_BASE_CHAR_MAP.keys()))

codec = CharacterCodec(alphabet, MAXLEN)

def guess(ngram):
    text = ' '.join(ngram)
    text += '\x00' * (MAXLEN - len(text))
    if INVERT:
        text = text[::-1]
    preds = model.predict(np.array([codec.encode(text)]), verbose = 0)
    rtext = codec.decode(np.argmax(preds, axis=-1)[0], calc_argmax=False).strip('\x00')

    if len(rtext)>0:
        index = rtext.find('\x00')
        if index>-1:
            rtext = rtext[:index]
    return rtext


def add_accent(text):
    # lowercase the input text as we train the model on lowercase text only
    # but we keep the map of uppercase characters to restore cases in output
    is_uppercase_map = [c.isupper() for c in text]
    text = remove_accent(text.lower())

    outputs = []
    words_or_symbols_list = re.findall('\w[\w ]*|\W+', text)

    # print(words_or_symbols_list)

    for words_or_symbols in words_or_symbols_list:
        if is_words(words_or_symbols):
            outputs.append(_add_accent(words_or_symbols))
        else:
            outputs.append(words_or_symbols)
        # print(outputs)
    output_text = ''.join(outputs)

    # restore uppercase characters
    output_text = ''.join(c.upper() if is_upper else c
                            for c, is_upper in zip(output_text, is_uppercase_map))
    return output_text

def _add_accent(phrase):
    grams = list(gen_ngram(phrase.lower(), n=NGRAM, pad_words=PAD_WORDS_INPUT))

    guessed_grams = list(guess(gram) for gram in grams)
    # print("phrase",phrase,'grams',grams,'guessed_grams',guessed_grams)
    candidates = [Counter() for _ in range(len(guessed_grams) + NGRAM - 1)]
    for idx, gram in enumerate(guessed_grams):
        for wid, word in enumerate(re.split(' +', gram)):
            candidates[idx + wid].update([word])
    output = ' '.join(c.most_common(1)[0][0] for c in candidates if c)
    return output.strip('\x00 ')