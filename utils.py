import re
import string
import time
from contextlib import contextmanager
import numpy as np



# maximum string length to train and predict
# this is set based on our ngram length break down below
MAXLEN = 32

# minimum string length to consider
MINLEN = 3

# how many words per ngram to consider in our model
NGRAM = 5

# inverting the input generally help with accuracy
INVERT = True

# mini batch size
BATCH_SIZE = 128

# number of phrases set apart from training set to validate our model
VALIDATION_SIZE = 100000

# using g2.2xl GPU is ~5x faster than a Macbook Pro Core i5 CPU
HAS_GPU = True

PAD_WORDS_INPUT  = True

### Ánh xạ từ không dấu sang có dấu

ACCENTED_CHARS = {
	'a': u'a á à ả ã ạ â ấ ầ ẩ ẫ ậ ă ắ ằ ẳ ẵ ặ',
	'o': u'o ó ò ỏ õ ọ ô ố ồ ổ ỗ ộ ơ ớ ờ ở ỡ ợ',
	'e': u'e é è ẻ ẽ ẹ ê ế ề ể ễ ệ',
	'u': u'u ú ù ủ ũ ụ ư ứ ừ ử ữ ự',
	'i': u'i í ì ỉ ĩ ị',
	'y': u'y ý ỳ ỷ ỹ ỵ',
	'd': u'd đ',
}

### Ánh xạ từ có dấu sang không dấu
ACCENTED_TO_BASE_CHAR_MAP = {}
for c, variants in ACCENTED_CHARS.items():
	for v in variants.split(' '):
		ACCENTED_TO_BASE_CHAR_MAP[v] = c

# \x00 ký tự padding

### Những ký tự cơ bản, bao gồm ký tự padding, các chữ cái và các chữ số
BASE_ALPHABET = set('\x00 _' + string.ascii_lowercase + string.digits)

### Bộ ký tự bao gồm những ký tự cơ bản và những ký tự có dấu
ALPHABET = BASE_ALPHABET.union(set(''.join(ACCENTED_TO_BASE_CHAR_MAP.keys())))


def is_words(text):
	return re.fullmatch('\w[\w ]*', text)

# Hàm bỏ dấu khỏi một câu
def remove_accent(text):
	""" remove accent from text """
	return u''.join(ACCENTED_TO_BASE_CHAR_MAP.get(char, char) for char in text)

#hàm thêm padding vào một câu
def pad(phrase, maxlen):
	""" right pad given string with \x00 to exact "maxlen" length """
	return phrase + u'\x00' * (maxlen - len(phrase))


def gen_ngram(words, n=3, pad_words=True):
	""" gen n-grams from given phrase or list of words """
	if isinstance(words, str):
		words = re.split('\s+', words.strip())

	if len(words) < n:
		if pad_words:
			words += ['\x00'] * (n - len(words))
		yield tuple(words)
	else:
		for i in range(len(words) - n + 1):
			yield tuple(words[i: i + n])

def extract_phrases(text):
	""" extract phrases, i.e. group of continuous words, from text """
	return re.findall(r'\w[\w ]+', text, re.UNICODE)


@contextmanager
def timing(label):
	begin = time.monotonic()
	print(label, end='', flush=True)
	try:
		yield
	finally:
		duration = time.monotonic() - begin
	print(': took {:.2f}s'.format(duration))

class CharacterCodec(object):
    def __init__(self, alphabet, maxlen):
        self.alphabet = list(sorted(set(alphabet)))
        self.index_alphabet = dict((c, i) for i, c in enumerate(self.alphabet))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.alphabet)))
        for i, c in enumerate(C[:maxlen]):
            X[i, self.index_alphabet[c]] = 1
        return X

    def try_encode(self, C, maxlen=None):
        try:
            return self.encode(C, maxlen)
        except KeyError:
            return None

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.alphabet[x] for x in X)
