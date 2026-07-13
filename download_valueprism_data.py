import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

def main():
    load_dotenv()
    ds = load_dataset("allenai/ValuePrism", "full")
    

    ds = pd.DataFrame(ds['train'])
    ds.to_csv("data/valueprism_data.csv", index=False)


if __name__ == "__main__":
    main()