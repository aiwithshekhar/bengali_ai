import pandas as pd
import joblib
import glob
from tqdm import tqdm

if __name__ == "__main__":
    files = glob.glob("input/*.parquet")
    for f in files:
        df = pd.read_parquet(f)
        img_ids = df.image_id.values
        df = df.drop("image_id", axis=1)
        imag_array = df.values
        for j, im_id in tqdm(enumerate(img_ids), total=len(img_ids)):
            joblib.dump(imag_array[j, :], f"input/image_pickles/{im_id}.pkl")
    # print(files)
