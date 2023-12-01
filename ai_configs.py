import os
import sys

# MODEL PARAMETERS: https://platform.openai.com/docs/models/gpt-3-5
MODEL_NAME = "gpt-3.5-turbo"  # Must select from MODEL_NAMES
MODEL_NAMES = ["gpt-4", "text-davinci-003", "gpt-3.5-turbo"]
EMBEDDING_MODEL = (
    "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
)

# CHATBOT SERVICE
SERVICE = "Teamhub"  # Must select from SERVICES
SERVICES = ["TokyoTechLab", "Teamhub", "Test"]

# DATA FORMATTING
DELIMITER_TOKYOTECHLAB = "Sub Section:"
FILE_TYPE = ".txt"
FILE_ENCODING = "utf-8"
INTRODUCTION_MESSAGE = (
    f"You are a chatbot of {SERVICE}. "
    f"Use the below articles on the {SERVICE} to answer the subsequent question. "  # noqa: E501
    "If an answer cannot be found in the articles, write sorry that I cannot answer your request, please contact our support team for further assistance."  # noqa: E501
    r'If an answer is found, add embedding title in this format "[Title](URL)" to the end of an answer and ignore the same title.'  # noqa: E501
)
SYSTEM_CONTENT = "You answer questions about {SERVICE}"

# CALCULATE EMBEDDING PARAMETERS
MAX_TOKENS = 1600  # maximum tokens for a section
BATCH_SIZE = 1000  # up to 2048 embedding inputs per request
TOKEN_BUDGET = 4096 - 500

# TRAINING PARAMETERS
CONTEXT_WINDOW = 4096  # Context window for the LLM.
NUM_OUTPUTS = 512  # Number of outputs for the LLM.
CHUNK_OVERLAP_RATIO = 0.1  # Chunk overlap as a ratio of chunk size
TEMPERATURE = 0.0  # A parameter that controls the “creativity” or
# randomness of the text generated. A higher temperature (e.g., 0.7)
# results in more diverse and creative output, while a lower temperature
# (e.g., 0.2) makes the output more deterministic and focused.

sys.path.append(os.path.abspath(os.path.join("..", "data")))

# PATH
if SERVICE in SERVICES:
    if MODEL_NAME in MODEL_NAMES:
        # Path to training files:
        FOLDERPATH_DOCUMENTS = os.path.join(
            "data",
            SERVICE,
            "training_files",
        )
        # Path to model
        FOLDERPATH_INDEXES = os.path.join(
            "models",
            SERVICE,
            MODEL_NAME,
        )
        FILEPATH_EMBEDDINGS = os.path.join(
            "models",
            SERVICE,
            "embeddings",
            f"{SERVICE}.csv",
        )
        # For evaluation
        FOLDERPATH_QUESTION = os.path.join(
            "data",
            SERVICE,
            "evaluation",
            "questions",
        )
        FOLDERPATH_QA = os.path.join(
            "data",
            SERVICE,
            "evaluation",
            "QA_" + MODEL_NAME,
        )
    else:
        raise ValueError("MODEL_NAME must be in MODEL_NAMES")
else:
    raise ValueError("SERVICE must be in SERVICES")
