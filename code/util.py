"""
Owen Wang (1831955), Flora Tang (1863826)
CSE 163 AA

Provides the functionality to cache datasets and college documentation. Also
contains some helper functions for main.py.
"""


import geopandas as gpd
import numpy as np
import os
import pandas as pd
import pickle
import re
import requests
from tempfile import TemporaryDirectory, TemporaryFile
from zipfile import ZipFile
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")


COLLEGE_DOCUMENT_PATH = ("https://collegescorecard.ed.gov/assets/" +
                         "CollegeScorecardDataDictionary.xlsx")
COLLEGE_PATH = ("https://ed-public-download.app.cloud.gov/downloads/" +
                "CollegeScorecard_Raw_Data.zip")
GEO_PATH = ("https://www.naturalearthdata.com/" +
            "http//www.naturalearthdata.com/download/110m/" +
            "cultural/ne_110m_admin_1_states_provinces.zip")


def ensure_dir_exists(path):
    """
    Takes a path.

    Creates the directories of the path if they don't exist.
    """
    os.makedirs(path, exist_ok=True)


def download_file(target, dest):
    """
    Takes a path of the target file and a destination.

    Downloads the file from the target and saves to destination.
    """
    r = requests.get(target)
    if type(dest) is str:
        ensure_dir_exists(os.path.dirname(dest))
        with open(dest, mode="w+b") as f:
            f.write(r.content)
    else:
        dest.write(r.content)


def join_curdir(*paths):
    """
    Appends the given paths to the current working directory.
    """
    return os.path.join(os.getcwd(), *paths)


class CacheLoader:
    """
    An abstract class that provides common methods for loading cache.
    """

    def __init__(self, log_type, remote_path, use_cache=True):
        """
        Takes and stores a log type, a remote path, and an optional bool
        indicating if the loader uses cache.
        """
        self._log_type = log_type
        self._remote_path = remote_path
        self._use_cache = use_cache

    def load_cache(self, path):
        """
        Abstract method. Takes a path.

        The implementation should load cache from the given path.
        """
        raise NotImplementedError()

    def build_cache(self, path):
        """
        Abstract method. Takes a path.

        The implementation should create cache at the given path.
        """
        raise NotImplementedError()

    def load(self, path):
        """
        Takes a path.

        Loads the cache at the given path. If the cache doesn't
        exist, builds the cache first.
        """
        self.log_cache_load(path)
        if self.use_cache and self.has_cache_file(path):
            self.log_cache_found()
            return self.load_cache(path)
        self.log_cache_found(False)
        cache = self.build_cache(path)
        self.log_cache_built(path)
        return cache

    @property
    def log_type(self):
        """
        Returns the log type.
        """
        return self._log_type

    @property
    def remote_path(self):
        """
        Returns the remote file path.
        """
        return self._remote_path

    @property
    def use_cache(self):
        """
        Returns whether the loader uses cache.
        """
        return self._use_cache

    def log_cache_load(self, path):
        """
        Takes a path.

        Logs that the loader is loading cache from the given path.
        """
        print(f"Loading {self.log_type} from '{path}'")

    def log_cache_found(self, found=True):
        """
        Takes a path and an optional bool indicating whether the cache is
        found.

        Logs whether the cache is found.
        """
        if found:
            print(f"Cache found! Loading {self.log_type} from cache")
        else:
            print(f"Cache not found. Downloading {self.log_type} " +
                  f"from '{self.remote_path}'")

    def log_cache_built(self, path):
        """
        Takes a path.

        Logs that the cache is built.
        """
        print(f"Done caching {self.log_type} from '{path}'")

    def has_cache_file(self, path):
        """
        Takes a path.

        Returns whether the file exists at the given path.
        """
        return os.path.isfile(path if type(path) is str else path.name)


class CollegeDocumentColumn:
    """
    A data class representing a column of the college dataset.
    """

    def __init__(self, name, description, type, values_mapping=None):
        """
        Takes a name, a description, a type, and a values mapping that maps
        values to their corresponding labels.
        """
        self._name = name
        self._type = type
        self._values_mapping = values_mapping
        self._description = description

    @property
    def column_name(self):
        """
        Returns the column name.
        """
        return self._name

    @property
    def column_type(self):
        """
        Returns the column data type.
        """
        return self._type

    @property
    def description(self):
        """
        Returns the column description.
        """
        return self._description

    @property
    def values_mapping(self):
        """
        Returns the values mapping of this column.
        """
        return self._values_mapping


class CollegeDocument:
    """
    A data class representing the college dataset.
    """

    def __init__(self, columns):
        """
        Takes and stores a list of columns. Adds the additional "Academic Year"
        column.
        """
        self._columns = {m.column_name: m for m in columns}
        self._columns["Academic Year"] = \
            CollegeDocumentColumn("Academic Year", "Academic Year", str)

    @property
    def columns(self):
        """
        Returns the list of columns.
        """
        return self._columns

    def get_column(self, id):
        """
        Takes an id.

        Returns the column with the given id.
        """
        return self._columns[id]

    def to_dataframe_args(self):
        """
        Generates and returns a dictionary of arguments for pandas.DataFrame.
        """
        includings = CollegeDocument.get_including_columns()
        return {
            "dtype": {c: d.column_type for c, d in self._columns.items()},
            "usecols": lambda col: col in includings,
            "na_values": ["NULL", "PrivacySuppressed"]
        }

    @staticmethod
    def get_including_columns():
        """
        Returns the columns to include when loading the dataset.
        """
        return (
            "Academic Year",
            "MD_EARN_WNE_P10",
            "CONTROL",
            "ST_FIPS",
            "SATVRMID",
            "SATMTMID",
            "SATWRMID",
            "ACTCMMID"
        )


class CollegeDocumentLoader(CacheLoader):
    """
    Loader for the college documentation.
    """

    def __init__(self, remote_path=COLLEGE_DOCUMENT_PATH, use_cache=True):
        """
        Takes an optional remote path and an optional bool indicating if the
        loader uses cache.

        Initializes the CollegeDocumentLoader.
        """
        super().__init__("CollegeDocument", remote_path, use_cache)

    def load_cache(self, path):
        """
        Takes a path.

        Implements the abstract method in CacheLoader.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def build_cache(self, path):
        """
        Takes a path.

        Implements the abstract method in CacheLoader.
        """
        with TemporaryFile() as raw_college_document:
            download_file(self.remote_path, raw_college_document)
            college_document = self._parse_raw_college_document(
                raw_college_document)
            self._save_college_document(college_document, path)
            return college_document

    def _parse_raw_college_document(self, raw_path):
        """
        Takes a raw path.

        Parses the raw documentation at the given path to a CollegeDocument.
        """
        df = pd.read_excel(raw_path, sheet_name="data_dictionary")
        df.fillna(method="ffill", inplace=True)
        groups = df.groupby("VARIABLE NAME")
        column_metas = []
        for col, col_detail in groups:
            if col not in CollegeDocument.get_including_columns():
                continue
            column_type = self._raw_type_to_numpy_type(
                col_detail["API data type"].iloc[0]
            )
            description = str(col_detail["NAME OF DATA ELEMENT"].iloc[0])
            values_mapping = col_detail.loc[:, ["VALUE", "LABEL"]].dropna() \
                .set_index("VALUE").to_dict()["LABEL"]
            column_metas.append(CollegeDocumentColumn(str(col), description,
                                                      column_type,
                                                      values_mapping))
        return CollegeDocument(column_metas)

    def _raw_type_to_numpy_type(self, dataset_type):
        """
        Takes a dataset type.

        Converts the given dataset type in the raw document to the
        corresponding numpy type.
        """
        converter = {
            "float": np.float64,
            "integer": np.float64,
            "boolean": np.bool_,
            "string": str,
            "autocomplete": str
        }
        return str if not dataset_type else converter.get(dataset_type, str)

    def _save_college_document(self, college_document, path):
        """
        Takes a college documentation and a path.

        Stores the given documentation at the given path.
        """
        ensure_dir_exists(os.path.dirname(path))
        with open(path, "wb") as f:
            pickle.dump(college_document, f)


class CollegeLoader(CacheLoader):
    """
    Loader for the college dataset.
    """

    def __init__(self, college_document, remote_path=COLLEGE_PATH,
                 use_cache=True):
        """
        Takes a college documentation, an optional remote path, and an optional
        bool indicating if the loader uses cache.

        Initializes the CollegeLoader.
        """
        super().__init__("dataset", remote_path, use_cache)
        self._college_document = college_document

    def load_cache(self, path):
        """
        Takes a path.

        Implements the abstract method in CacheLoader.
        """
        return pd.read_csv(path, **self._college_document.to_dataframe_args())

    def build_cache(self, path):
        """
        Takes a path.

        Implements the abstract method in CacheLoader.
        """
        with TemporaryDirectory() as raw_dataset_path:
            self._download_raw_dataset(raw_dataset_path)
            df = self._combine_raw_dataset(raw_dataset_path)
            ensure_dir_exists(os.path.dirname(path))
            df.to_csv(path, index=False)
            return df

    def _download_raw_dataset(self, raw_dataset_path):
        """
        Takes a raw dataset path.

        Downloads the raw dataset from the given path.
        """
        zip_dataset = TemporaryFile()
        download_file(self.remote_path, zip_dataset)
        ensure_dir_exists(raw_dataset_path)
        with ZipFile(zip_dataset, "r") as zip_file:
            # Extracts all .csv dataset file and renames them in the form
            # of "<begin year>-<end year>.csv"
            for f in zip_file.infolist():
                file_name = os.path.basename(f.filename)
                if file_name and file_name.endswith(".csv"):
                    year_index = re.search(r"\d{4}", file_name)
                    year = file_name[year_index.start():year_index.end()]
                    f.filename = f"{year}-{int(year) + 1}.csv"
                    zip_file.extract(f, raw_dataset_path)

    def _process_raw_dataset(self, raw_dataset_path, dataframe_args):
        """
        Takes a raw dataset path and a dictionary of dataframe arguments.

        Loads, processes, and returns the dataset at the given path with the
        given arguments.
        """
        print(f"Processing raw dataset at '{raw_dataset_path}'")
        df = pd.read_csv(raw_dataset_path, **dataframe_args)
        academic_year = os.path.basename(raw_dataset_path).split(".")[0]
        begin_year, end_year = academic_year.split("-")
        df["Academic Year"] = academic_year
        return df

    def _combine_raw_dataset(self, raw_path):
        """
        Takes a raw path.

        Combines the datasets in the given path.
        """
        print("Combining raw datasets")
        dataframe_properties = self._college_document.to_dataframe_args()
        df = pd.concat(self._process_raw_dataset(os.path.join(raw_path, f),
                                                 dataframe_properties)
                       for f in os.listdir(raw_path) if f.endswith(".csv"))
        return df


class GeoLoader(CacheLoader):
    """
    Loader for the geo dataset.
    """

    def __init__(self, remote_path=GEO_PATH, use_cache=True):
        """
        Takes an optional remote path and an optional bool indicating if the
        loader uses cache.

        Initializes the GeoLoader.
        """
        super().__init__("geodata", remote_path, use_cache=use_cache)

    def load_cache(self, path):
        """
        Takes a path.

        Implements the abstract method in CacheLoader.
        """
        return gpd.read_file(path)

    def build_cache(self, path):
        """
        Takes a path.

        Implements the abstract method in CacheLoader.
        """
        with TemporaryDirectory() as raw_geodata_path:
            self._download_geodata(raw_geodata_path, os.path.dirname(path))
        return gpd.read_file(path)

    def _download_geodata(self, from_path, dest):
        """
        Takes a path and a destination.

        Downloads the geospatial dataset from the given path to the
        destination.
        """
        zip_dataset = TemporaryFile()
        download_file(self.remote_path, zip_dataset)
        ensure_dir_exists(from_path)
        with ZipFile(zip_dataset, "r") as zip_file:
            for f in zip_file.infolist():
                if not f.is_dir():
                    f.filename = os.path.basename(f.filename)
                    zip_file.extract(f, dest)
