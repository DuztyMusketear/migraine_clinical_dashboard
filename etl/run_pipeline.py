from etl.extract.extract import extract
from etl.transform.transform import transform
from etl.load.load import load

def run_pipeline():
    df = extract()
    df = transform(df)
    load(df)
    return df

if __name__ == "__main__":
    run_pipeline()