import pandas as pd
import matplotlib.pyplot as plt 
import loguru as logging
import os


class Label_Processing(): # base class
''' Base class for label processing'''    

    def __init__(self, df: pd.DataFrame,save_path: str,) -> None:
        self.df = df
        self.save_path = save_path

    def save_csv(self, df: pd.DataFrame, name: str) -> None:
        path = os.join(self.save_path, f"{name}.csv")
        df.to_csv(path, index=False)
        logging.info(f"Saved {name} to {path}")
    
    def save_parequet(self, df: pd.DataFrame, name: str) -> None:
        path = os.join(self.save_path, f"{name}.parquet")
        df.to_parquet(path, index=False)
        logging.info(f"Saved {name} to {path}")