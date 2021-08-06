
import requests
import pathlib as pl
import pandas as pd
import io


class GitHubData:
    def __init__(self, user, repo, branch, path):
        self.user = user
        self.repo = repo
        self.branch = branch
        self.path = pl.Path(path)

        self._content = None

    @property
    def content(self):
        if self._content is None:
            print(f"Fetching content from {self.url(raw=False)}")
            self.fetch()

        return self._content

    def url(self, raw=True):
        if raw:
            return "https://raw.githubusercontent.com/" + self.user + "/" + self.repo \
                   + "/" + self.branch + "/" + str(self.path.as_posix())
        return "https://github.com/" + self.user + "/" + self.repo + "/tree/" + self.branch + "/" + str(self.path.as_posix())

    def fetch(self, save=True, path=None):
        url = self.url()
        response = requests.get(url)
        response.raise_for_status()

        self._content = response.content

        if save:
            self.save(path)

        return self._content

    def unfetch(self):
        self._content = None

    def save(self, path=None):
        if path is None:
            path = self.path

        if not path.exists():
            path.mkdir(parents=True)
            path.rmdir()
        path.write_bytes(self.content)

    def load(self, path=None):
        if path is None:
            path = self.path

        self._content = path.read_bytes()

    def to_df(self, *args, **kwargs):
        df = pd.read_csv(io.BytesIO(self.content), *args, **kwargs)
        self.unfetch()
        return df

    def reencode(self, old="cp1252", new="utf8"):
        content = self.content
        bytes_io = io.BytesIO(content)
        decoded = bytes_io.read().decode(old)
        str_io = io.StringIO(decoded)
        encoded = str_io.read().encode(new)
        bytes_io = io.BytesIO(encoded)

        self._content = bytes_io.read()


def fetch():
    responses = GitHubData(user="OxCGRT", repo="covid-policy-tracker", branch="master", path="data/OxCGRT_latest.csv")
    responses2 = GitHubData(user="amel-github", repo="covid19-interventionmeasures", branch="master", path="COVID19_non-pharmaceutical-interventions_version2_utf8.csv")
    testing = GitHubData(user="owid", repo="covid-19-data", branch="master", path="public/data/owid-covid-data.csv")

    return responses, responses2, testing
