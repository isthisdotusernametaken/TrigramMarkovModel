from nltk import download
from nltk.tokenize import sent_tokenize, word_tokenize

from Markov import TrigramModel


# Ensure the necessary tools for tokenization are available
print("Checking requirements...")
download("punkt")
print()


# Utility constants

# The number of words to generate in the output
_WORD_COUNT = 2000

# The maximum number of words between random words
# Setting this to 15 seems to avoid long strings of the same sentence
# repeating, but this also causes some long sentences.
_REFRESH_LIMIT = 15

# Text files to train the model on and how many lines to skip at the start of
# each file
_FILES = [("houn.txt", 8), ("sign.txt", 8), ("stud.txt", 8), ("vall.txt", 8)]

# Based on the example linked in the instructions document
# (https://www.geeksforgeeks.org/removing-punctuations-given-string/), but
# periods are excluded so that they are counted as words.
_PUNCTUATION = "!()-[]{};:''``\"\\,<>/?@#$%^&*_~"

# Words to always capitalize
_CAPITALIZE = [
    "i", "mr.", "dr.", "sherlock", "holmes", "john", "watson", "moriarty"
]

# The output file name
_OUTPUT_FILE = "Readme.txt"


# Input parsing

# Find the index of the nth instance of a substring in a given string, or -1 if
# the substring does not have n instances (or if n < 1).
def _nth_index(string, substring, n):
    ind = -1
    for i in range(n):
        ind = string.find(substring, ind + 1)
        if ind == -1:
            return -1 # Signal no more instances
    
    return ind

# Read the specified input file and return the remainder after skipping the
# specified number of lines.
def _read_file(file_and_skip):
    try:
        with open(file_and_skip[0], "r") as input:
            content = input.read()
    except OSError:
        print(f'The contents of "{file_and_skip[0]}" could not be read.')
        return None
    
    # Skip the specified number of initial lines and return the result.
    return content[_nth_index(content, '\n', file_and_skip[1]) + 1:]

# Correct for the punkt word tokenizer's tendency to leave a single quote at
# the start of a word after removing other punctuation.
# This leaves "'s" as-is.
def _remove_starting_single_quote(word):
    return word[1:] if word[0] == "'" and word[1] != "s" else word

def _tokenize(words):
    # Split input into sentences.
    # sent_tokenize successfully treats some periods such as the period in "Mr." as
    # not the end of a sentence, but other cases such as abbreviations (e.g.,
    # "C.C.H.") cause a sentence to end incorrectly. For simplicity, this is not
    # corrected here.
    sentences = sent_tokenize(words)

    # Split sentences into single list of words, excluding punctuation.
    # Words in sentences could be treated differently here for a more accurate
    # language model, but sentence tokenization currently only helps identify
    # periods at the ends of sentences.
    return [_remove_starting_single_quote(word.lower())
        for wordList in [word_tokenize(sent) for sent in sentences] # Split each sentence
        for word in wordList # Extract word from its sentence's list
        if word not in _PUNCTUATION # Exclude punctuation
    ]


# Model training and output generation

# Split the provided text into words and feed the words to the model.
def _consume_text(model, text):
    words = _tokenize(text)

    model.start_input(words[0], words[1])
    for i in range(2, len(words)):
        model.consume_word(words[i])
    model.end_input()

# Parse the provided input file into words and train the provided model.
def _build_model():
    model = TrigramModel()

    for file_and_skip in _FILES:
        content = _read_file(file_and_skip)
        if content is None:
            return None # Signal failure. Error message is already printed.
        
        _consume_text(model, content)

    model.finish()
    return model

# Return a string of the specified number of words based on the model.
# After the specified number of words, continue until a period is generated.
def _generate_words(model, word_count, refresh_limit):
    generator = model.output_generator(refresh_limit)
    words = []

    count = 0
    capitalize = True
    new_word = None
    while count < word_count or new_word != ".":
        new_word = generator.generate_word()

        # If a period is produced, add it to the end of the previous word
        # without a space, capitalize the next word, and do not increment the
        # counter (do not count periods toward the word count).
        if new_word == ".":
            if count == 0: # Do not start with a period
                continue

            words.append(words.pop() + ".")
            capitalize = True
        else:
            words.append(
                new_word.capitalize() if capitalize or new_word in _CAPITALIZE \
                else new_word
            )
            capitalize = False
            count += 1

    # Separate the words with spaces and return a string.
    return " ".join(words)


# File printing

def _print_file(text):
    try:
        with open(_OUTPUT_FILE, "w") as file:
            file.write(text)
    except OSError:
        print(
            f'The results could not be written to "{_OUTPUT_FILE}". Make sure '
            'the program is run with sufficient permissions.'
        )


def main():
    model = _build_model()
    if model is None:
        return -1 # Failure

    # Generate output
    _print_file(_generate_words(model, _WORD_COUNT, _REFRESH_LIMIT))

if __name__ == "__main__":
    main()
