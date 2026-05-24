import pandas as pd
import matplotlib.pyplot as plt 
import loguru as logging
import os


class Label_Processing(): # base class
    ''' Base class for label processing'''    

    def __init__(self, df: pd.DataFrame,save_path: str) -> None:
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


class Label_Processing_Classic(Label_Processing):
    ''' Label processing for galaxy using the orignal paper by (Zhu et al., 2019)'''    

    def __init__(self, df: pd.DataFrame, save_path: str,        
        f_smooth:float = 0.469,
        f_completely_round: float = 0.500,
        f_in_between: float = 0.500,
        f_cigar_shaped: float = 0.500,
        f_faturesdisk: float = 0.430,
        f_edge_onyes: float = 0.602,
        f_edge_onno: float = 0.715,
        f_spiral_yes: float = 0.619) -> None:
     
        super().__init__(df, save_path)

        self.f_smooth = f_smooth
        self.f_completely_round = f_completely_round
        self.f_in_between = f_in_between
        self.f_cigar_shaped = f_cigar_shaped
        self.f_faturesdisk = f_faturesdisk
        self.f_edge_onyes = f_edge_onyes
        self.f_edge_onno = f_edge_onno
        self.f_spiral_yes = f_spiral_yes

    def process_labels(self) -> pd.DataFrame:
        pass