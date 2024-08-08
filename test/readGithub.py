from llama_index.readers.github import GithubRepositoryReader, GithubClient
import os 

github_token = os.environ['GITHUBTOKEN']
client = github_client = GithubClient(github_token=github_token, verbose=False)

reader = GithubRepositoryReader(
    github_client=github_client,
    owner="run-llama",
    repo="llama_index",
    use_parser=False,
    verbose=True,
    filter_directories=(
        ["docs"],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
    filter_file_extensions=(
        [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".ico",
            "json",
            ".ipynb",
        ],
        GithubRepositoryReader.FilterType.EXCLUDE,
    ),
)

documents = reader.load_data(branch="main")

print(documents)