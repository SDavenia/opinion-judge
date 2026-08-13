import json, csv
import os
from typing import Union
from pathlib import Path
from datetime import datetime

def write_csv(list_of_dics, name_file):

    if (len(list_of_dics)>0):
        with open(name_file, 'w', newline='',  encoding="utf-8") as output_file:
            dict_writer = csv.DictWriter(output_file, list_of_dics[0].keys(), extrasaction="ignore")
            dict_writer.writeheader()
            dict_writer.writerows(list_of_dics )
            

def read_csv_delim(file: str, delimiter:str) -> None:
    list_dicts = []
    with open(file , encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        list_dicts = list(reader)
    return list_dicts

def dump_json(data, file_name, cls=None):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4,  ensure_ascii=False, cls=cls)

def read_json(file_name):
    data = {}
    with open(file_name, encoding="utf-8") as f:
        data = json.load(f)
    return data

def read_txt(file_name, return_list=True, strip=False):
    with open(file_name, "r", encoding= "utf-8") as f:
        if return_list:
            list_lines = f.readlines()
            list_lines = [l.strip().strip("\n") for l in list_lines]
        else:
            file_string = f.read()
            if strip:
                file_string.strip().strip("\n")
            return file_string
    return list_lines

def dump_jsonl(data, file_name, encoder=None):
    with open(file_name, "w", encoding="utf-8") as f:
        for ex in data:
            json.dump(ex, f, ensure_ascii=False, cls=encoder)
            f.write("\n")

def read_jsonl(file_name):
        data = []
        with open(file_name, encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

def append_jsonl(new_data: Union[list, dict], file_name: str):
    with open(file_name, "a", encoding="utf-8") as f:
        if isinstance(new_data, list):
            for record in new_data:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        else:
            json.dump(new_data, f, ensure_ascii=False)
            f.write("\n")

def get_files_in_dir(dir_path: Union[Path, str]) -> list[str]:
    #return [(f.name, f.path) for f in os.scandir(dir_path) if f.is_file()]
    return [f.path for f in os.scandir(dir_path) if f.is_file()]


def get_name_file(path: Union[Path, str]) -> str:
    return os.path.basename(path)

def get_extention_file(path: Union[Path, str]) -> str:
    filename, file_extension = os.path.splitext(path)
    return file_extension

def get_no_extention_file(path: Union[Path, str]) -> str:
    filename, file_extension = os.path.splitext(path)
    return filename

def get_dir_of_path(path:Union[Path, str]):
    if not (isinstance(path, Path)):
        path = Path(path)
    return path.parent


def get_dirs_in_dir(dir_path) -> list[str]:
    return [d.path for d in os.scandir(dir_path) if d.is_dir()]



def join_paths(*to_concatenate, create_dir = False) -> Path:
    assert len(to_concatenate) >= 2, "To concatenate a path it must have at least 2 args"
    starting_path = Path(to_concatenate[0]) if not (isinstance(to_concatenate[0], Path)) else to_concatenate[0]
    rest_of_paths = to_concatenate[1:]
    resulting_path = starting_path.joinpath(*rest_of_paths)
    if create_dir: 
        resulting_path.mkdir(parents=True, exist_ok=True)

    return resulting_path

def create_dir(path) -> None:
    if not (isinstance(path, Path)):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    
def get_str_date() -> str:
    return f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"