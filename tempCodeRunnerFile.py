import configparser
import os
import warnings

import openai
import pandas as pd
import tiktoken
from ai_configs import (
    BATCH_SIZE,
    DELIMITER_TOKYOTECHLAB,
    EMBEDDING_MODEL,
    FILE_ENCODING,
    FILE_TYPE,
    FILEPATH_EMBEDDINGS,
    FOLDERPATH_DOCUMENTS,
    MAX_TOKENS,
    MODEL_NAME,
    SERVICE,
)

env = configparser.ConfigParser()
env.read(".env")
os.environ["OPENAI_API_KEY"] = env["OpenAI"]["OPENAI_KEY_TTL"]
openai.api_key = os.environ["OPENAI_API_KEY"]


def list_files(directory: str) -> list:
    """List all files that match the FILE_TYPE in a directory"""
    files = []
    for file in os.listdir(directory):
        # check only text files
        if file.endswith(FILE_TYPE):
            files.append(file)
    return files


def read_file(file_path: str) -> str:
    """Read a file"""
    # Open a file: file
    file = open(file_path, encoding=FILE_ENCODING)

    # read all lines at once
    file_content = file.read()

    # close the file
    file.close()
    return file_content


def num_tokens(text: str, model: str = MODEL_NAME) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = True,
) -> str:
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(
            f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.",  # noqa: E501
        )
    return truncated_string


def determine_delimiter(
    strings: str,
    service: str = SERVICE,
) -> str:
    """
    Determine the delimiter of the file
    """
    if service == "TokyoTechLab":
        return DELIMITER_TOKYOTECHLAB
    elif service == "Teamhub":
        if "# " in strings:
            return "# "
        elif "## " in strings:
            return "## "
        elif "### " in strings:
            return "### "
        else:
            raise ValueError(
                f"No delimiter found in Teamhub file: {strings[0:20]}",
            )
    else:
        raise ValueError(f"Unknown service: {service}")


def format_content_Tokyo_Tech_Lab(
    strings: str,
    content: str,
    max_tokens: int = 1000,
    model: str = MODEL_NAME,
):
    """
    Format content for Tokyo Techies website
    """
    chunks = content.split(determine_delimiter(content))
    # TODO: add to config
    if "URL:" and "Language:" in chunks[0]:
        url = (
            "<url>"
            + (content.split("URL:"))[1].split("Language")[0].strip()
            + "</url>"
        )  # get url
    else:
        url = "<url>No URL</url>"

    for chunk in chunks[1:]:
        chunk = (
            chunk.strip()
        )  # remove leading and trailing whitespace and newline
        if not chunk:
            continue

        # get section title (first row) and content (from 2nd row)
        section_title = chunk.split("\n")[0]
        titles = [url, section_title]
        section_content = chunk.split("\n")[1:]
        section_content = "\n".join(section_content)

        if num_tokens(section_content) > max_tokens:
            print(
                f"{titles} ({num_tokens(section_content)}) has more than {max_tokens} tokens",  # noqa: E501
            )
            section_content = truncated_string(
                section_content,
                model=model,
                max_tokens=max_tokens,
            )

        string = "\n\n".join(titles + [section_content])
        strings.extend([string])
        print(string)
    return strings


def format_content_Teamhub(
    strings: str,
    content: str,
    max_tokens: int = 1000,
    model: str = MODEL_NAME,
):
    """
    Format content for Teamhub
    """

    # Add images tag to image link
    content = content.replace("![](", "![image](")

    delimiter = determine_delimiter(content)
    if delimiter:
        chunks = content.split(delimiter)
    else:
        chunks = [content]

    # TODO: add to config
    url = ""
    if "Title:" and "URL:" in chunks[0]:
        title = "Title: " + (
            (content.split("Title:"))[1].split("URL")[0].strip()
        )
        if "Language:" in chunks[0]:
            url = (
                "<url>"
                + (content.split("URL:"))[1].split("Language:")[0].strip()
                + "</url>"
            )
    else:
        title = ""

    # Extract content between title and the first sub-section
    section_content = (chunks[0].split("-----"))[1].strip()

    if section_content != "":
        titles = [title, url]
        string = "\n\n".join(titles + [section_content])
        # print(f"----------\n{string}\n")
        strings.extend([string])

    # Extract contentin every sub-section
    for chunk in chunks[1:]:
        chunk = (
            chunk.strip()
        )  # remove leading and trailing whitespace and newline
        if not chunk:
            continue

        # get section title (first row) and content (from 2nd row)
        section_title = chunk.split("\n")[0]
        titles = [title + " > " + section_title, url]
        section_content = chunk.split("\n")[1:]
        section_content = "\n".join(section_content)

        if num_tokens(section_content) > max_tokens:
            print(
                f"{titles} ({num_tokens(section_content)}) has more than {max_tokens} tokens",  # noqa: E501
            )
            section_content = truncated_string(
                section_content,
                model=model,
                max_tokens=max_tokens,
            )

        string = "\n\n".join(titles + [section_content])
        # print(f"----------\n{string}\n")
        strings.extend([string])
    return strings


def format_content(
    directory: str,
    max_tokens: int = 1000,
    model: str = MODEL_NAME,
) -> list[str]:
    strings = []

    # read files
    files = list_files(directory)
    for file in files:
        print(f"File: {file}")
        file_content = read_file(
            os.path.join(
                FOLDERPATH_DOCUMENTS,
                file,
            ),
        )
        
        if SERVICE == "TokyoTechLab":
            
            strings = format_content_Tokyo_Tech_Lab(
                strings,
                file_content,
                max_tokens,
                model,
            )
        elif SERVICE == "Teamhub":
            strings = format_content_Teamhub(
                strings,
                file_content,
                max_tokens,
                model,
            )

    return strings


def embed_data():
    """
    Main function to create embbeding data from raw data
    Embedding file will be saved into FILEPATH_EMBEDDINGS
    Main flow:
    1. Read crawl data from a folder
    2. Format raw data into standard data
    3. Embed data in 3 into embedding data via OpenAI API
    4. Save embedding data into file
    """

    formatted_strings = format_content(FOLDERPATH_DOCUMENTS, MAX_TOKENS)
   
    embeddings = []
    for batch_start in range(0, len(formatted_strings), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = formatted_strings[batch_start:batch_end]
        print(f"Batch {batch_start} to {batch_end-1}")
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
        for i, be in enumerate(response["data"]):
            assert (
                i == be["index"]
            )  # double check embeddings are in same order as input
        batch_embeddings = [e["embedding"] for e in response["data"]]
        embeddings.extend(batch_embeddings)

    df = pd.DataFrame({"text": formatted_strings, "embedding": embeddings})

    # save document chunks and embeddings
    SAVE_PATH = FILEPATH_EMBEDDINGS
    df.to_csv(SAVE_PATH, index=False)


class Embedding:
    """
    A class is a service (e.g., TT, Teamhub, etc.)
    Usage:
    - Init class (__init__)
    - Add/update/remove embeddings in the class
    - Save the modifications to file (save_embedding)

    Example:
        ttl = Embedding("TokyoTechLab", ".")
        ttl.add_embedding(0, "how are they?", "Fine!")
        ttl.add_embedding(1, "how are u?", "Good!")
        ttl.update_embedding(1, "how are you?", "Good!", "Typo")
        ttl.remove_embedding(0)
        ttl.save_embedding()
    """

    def __init__(self, service: str, path: str) -> None:
        """
        Init a class for each service (e.g., TTL, Teamhub, etc.)
            Embedding file: {path}/{service}.csv
        args:
            service(str): service's name
            path(str): path to embedding file
        """
        self.model = EMBEDDING_MODEL
        self.service = service
        self.path = path
        self.filepath = os.path.join(
            path,
            f"{service}_embedding.csv",
        )
        if os.path.isfile(self.filepath):
            self.df = pd.read_csv(self.filepath)
        else:
            self.df = pd.DataFrame(
                columns=["index", "formatted_strings", "embeddings"],
            )

        self.question = ""
        self.answer = ""
        self.category = ""
        self.formatted_qa = ""
        self.embedding = []

    def format_input(
        self,
        question: str,
        answer: str,
        category: str = "",
    ) -> None:
        """
        format question & answer before embedding
        args:
            question(str): question string
            answer(str): answer string
            category(str): category string
        """
        if category != "":
            self.category = "Category: " + category + "\n"
        self.question = "Question: " + question + "\n"
        self.answer = "Answer: " + answer + "\n"
        self.formatted_qa = self.category + self.question + self.answer

    def get_embedding(self) -> None:
        """
        Get embedding from input string via OpenAI API
        """
        response = openai.Embedding.create(
            model=self.model,
            input=[self.formatted_qa],
        )

        # double check embeddings are in same order as input
        for i, be in enumerate(response["data"]):
            assert i == be["index"]

        # batch_embeddings = [e["embedding"] for e in response["data"]]
        # embeddings.extend(batch_embeddings)
        print(response["data"][0]["embedding"])
        self.embedding = str(response["data"][0]["embedding"])

    def add_embedding(
        self,
        index: int,
        question: str,
        answer: str,
        category: str = "",
    ) -> None:
        if index in list(self.df["index"]):
            msg = (
                f"'index: {index}' is avalable. "
                "Switch to 'update_embedding()' automatically."
            )
            warnings.warn(msg)
            self.update_embedding(index, question, answer, category)
            return

        self.format_input(question, answer, category)
        self.get_embedding()

        if len(self.embedding) > 0:
            dict = {
                "index": index,
                "formatted_strings": self.formatted_qa,
                "embeddings": self.embedding,
            }
            self.df = pd.concat(
                [
                    self.df,
                    pd.DataFrame([dict]),
                ],
                ignore_index=True,
            )
            self.reset_values()
        else:
            msg = f"'Couldn't receive any embedding for 'index: {index}'."
            warnings.warn(msg)

    def remove_embedding(
        self,
        index: int,
        # question:str="",
        # answer:str = "",
        # category:str="",
    ) -> None:
        """
        Update an embedding based on index
        args:
            index(int): index of the QA
        """
        if index not in list(self.df["index"]):
            msg = (
                f"'index: {index}' is not avalable. "
                "Nothing will be removed."
            )
            warnings.warn(msg)
            return

        self.df = self.df[self.df["index"] != index]

    def update_embedding(
        self,
        index: int,
        question: str,
        answer: str,
        category: str = "",
    ) -> None:
        """
        Update an embedding based on index
        args:
            index(int): index of the QA
            question(str): question string
            answer(str): answer string
            category(str): category string
        """
        if index not in list(self.df["index"]):
            msg = (
                f"'index: {index}' is not avalable. "
                "Switch to 'add_embedding()' automatically"
            )
            warnings.warn(msg)
            self.add_embedding(index, question, answer, category)
            return

        self.format_input(question, answer, category)
        self.get_embedding()

        if len(self.embedding) > 0:
            self.df.loc[
                self.df["index"] == index,
                ["formatted_strings", "embeddings"],
            ] = [
                self.formatted_qa,
                self.embedding,
            ]
            self.reset_values()
        else:
            msg = f"'Couldn't receive any embedding for 'index: {index}'."
            warnings.warn(msg)

    def save_embedding(self) -> None:
        """
        Save document chunks and embeddings
        """
        self.df.to_csv(self.filepath, index=False)

    def reset_values(self) -> None:
        """
        reset all strings to ""
        """
        self.question = self.answer = self.category = ""
        self.formatted_qa = ""
        self.embedding = []


if __name__ == "__main__":
    embed_data()
