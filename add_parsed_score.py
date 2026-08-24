from utils.scoring_utils import parse_generation_scoring
from utils.files_io import read_csv_delim, create_dir, write_csv
import argparse
from pathlib import Path


def read_scoring_outputs(dir_to_parse: Path) -> list[dict]:
    outputs = []
    for file in dir_to_parse.iterdir():
        if file.is_file():
            file_scores = read_csv_delim(file, ",")
            dict_file = {"path": file,
                         "scores": file_scores}
            outputs.append(dict_file)
    return outputs

def parse_scores(scoring_outputs: list[dict]) -> None:

    for file_dict in scoring_outputs:
        scores = file_dict["scores"]
        for row_score in scores:
            if "parsed_score_1to2" not in row_score:
                row_score["parsed_score_1to2"] = parse_generation_scoring(row_score["generated_score_1to2"])
            if "parsed_score_2to1" not in row_score:
                            row_score["parsed_score_2to1"] = parse_generation_scoring(row_score["generated_score_2to1"])
                       
def write_scores(scoring_outputs: list[dict], out_dir:Path)-> None:

    create_dir(out_dir)

    for file_dict in scoring_outputs:
        file_path = file_dict["path"]
        scores = file_dict["scores"]
        out_file_path = out_dir / file_path.name
        write_csv(scores, out_file_path)

def main(args: argparse.Namespace) -> None:

    dir_to_parse = args.dir_to_parse 
    out_dir = args.out_dir

    print(f"Reading the scoring outputs from: {dir_to_parse}")
    scoring_outputs = read_scoring_outputs(dir_to_parse)
    print(f"Now adding the scores...")
    parse_scores(scoring_outputs)
    print(f"Now writing the parsed scores to: {out_dir}")
    write_scores(scoring_outputs, out_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_to_parse",
                        type=Path,
                        help="Directory where there are .csv which are result of scoring to parse.",
                        required=True)
    parser.add_argument("--out_dir",
                        type=Path,
                        required=True)
    args = parser.parse_args()
    
    main(args)

