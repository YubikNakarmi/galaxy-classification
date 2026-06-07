import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger as logging
import os
from typing import Optional, Dict


class Label_Processing(): # base class
    ''' Base class for label processing'''    

    def __init__(self, df: pd.DataFrame,save_path: str) -> None:
        self.df = df
        self.save_path = save_path

    def save_csv(self, df: pd.DataFrame, name: str) -> None:
        path = os.path.join(self.save_path, f"{name}.csv")
        df.to_csv(path, index=False)
        logging.info(f"Saved {name} to {path}")
    
    def save_parequet(self, df: pd.DataFrame, name: str) -> None:
        path = os.path.join(self.save_path, f"{name}.parquet")
        df.to_parquet(path, index=False)
        logging.info(f"Saved {name} to {path}")


class Label_Processing_Classic(Label_Processing):
    ''' Label processing for galaxy using the orignal paper by (Zhu et al., 2019)
    
    Args:
    df (pd.DataFrame): The input dataframe containing the raw data.
    f_smooth (float): Threshold for smoothness.
    f_completely_round (float): Threshold for completely round class.
    f_in_between (float): Threshold for in-between class.
    f_cigar_shaped (float): Threshold for cigar-shaped class.
    f_faturesdisk (float): Threshold for features/disk class.
    
    
    '''    


    CLASS_NAMES = {
        0: "completely_round_smooth",
        1: "in_between_smooth",
        2: "cigar_shaped_smooth",
        3: "edge_on",
        4: "spiral"}

    def __init__(self, df: pd.DataFrame, save_path: str,        
        f_smooth:float = 0.469,
        f_completely_round: float = 0.500,
        f_in_between: float = 0.500,
        f_cigar_shaped: float = 0.500,
        f_faturesdisk: float = 0.430,
        f_edge_onyes: float = 0.602,
        f_edge_onno: float = 0.715,
        f_spiral_yes: float = 0.619,
        voting_columns:Dict[str, str]=None, min_votes: int = 50) -> None:

        ''' 
         Args:
            df (pd.DataFrame): The input dataframe containing the raw data.
            f_smooth (float): Threshold for smoothness.
            f_completely_round (float): Threshold for completely round class.
            f_in_between (float): Threshold for in-between class.
            f_cigar_shaped (float): Threshold for cigar-shaped class.
            f_faturesdisk (float): Threshold for features/disk class.
            f_edge_onyes (float): Threshold for edge-on yes class.
            f_edge_onno (float): Threshold for edge-on no class.
            f_spiral_yes (float): Threshold for spiral yes class.
        
        '''


        self.f_smooth = f_smooth
        self.f_completely_round = f_completely_round
        self.f_in_between = f_in_between
        self.f_cigar_shaped = f_cigar_shaped
        self.f_faturesdisk = f_faturesdisk
        self.f_edge_onyes = f_edge_onyes
        self.f_edge_onno = f_edge_onno
        self.f_spiral_yes = f_spiral_yes
        self.voting_columns: Dict[str, str] = voting_columns or {}
        self.min_votes = min_votes

        self.final_df = None
        super().__init__(df,save_path)
    
    def find_voting_columns(self) -> None:
        ''' Finding relevant columens if not provided'''

        self.voting_columns['smooth_col'] = self.df.columns[self.df.columns.str.startswith('t01') & self.df.columns.str.endswith('smooth_debiased')][0]
        self.voting_columns['features_col'] = self.df.columns[self.df.columns.str.startswith('t01') & self.df.columns.str.endswith('features_or_disk_debiased')][0]
        self.voting_columns['edge_yes_col'] = self.df.columns[self.df.columns.str.startswith('t02') & self.df.columns.str.endswith('yes_debiased')][0]
        self.voting_columns['edge_no_col'] = self.df.columns[self.df.columns.str.startswith('t02') & self.df.columns.str.endswith('no_debiased')][0]
        self.voting_columns['spiral_col'] = self.df.columns[self.df.columns.str.startswith('t04') & self.df.columns.str.endswith('spiral_debiased')][0]
        self.voting_columns['round_col'] = self.df.columns[self.df.columns.str.startswith('t07') & self.df.columns.str.endswith('completely_round_debiased')][0]
        self.voting_columns['inbet_col'] = self.df.columns[self.df.columns.str.startswith('t07') & self.df.columns.str.endswith('in_between_debiased')][0]
        self.voting_columns['cigar_col'] = self.df.columns[self.df.columns.str.startswith('t07') & self.df.columns.str.endswith('cigar_shaped_debiased')][0]


    def _assign_label(self, row) -> Optional[int]:
        """Assign labels using standard method from Zhu et al. (2019)"""




        vc = self.voting_columns
        required = [
            'smooth_col','features_col','round_col','inbet_col','cigar_col',
            'edge_yes_col','edge_no_col','spiral_col'
        ]
        for k in required:
            if k not in vc:
                raise KeyError(f"voting_columns must include '{k}' mapping")

        smooth_col = vc['smooth_col']
        features_col = vc['features_col']
        round_col = vc['round_col']
        inbet_col = vc['inbet_col']
        cigar_col = vc['cigar_col']
        edge_yes_col = vc['edge_yes_col']
        edge_no_col = vc['edge_no_col']
        spiral_col = vc['spiral_col']

        s = row[smooth_col]
        fd = row[features_col]

        if s >= self.f_smooth and fd < self.f_faturesdisk:
            t07 = {0: row[round_col], 1: row[inbet_col], 2: row[cigar_col]}
            best_cls, best_val = max(t07.items(), key=lambda x: x[1])
            if best_val >= self.f_completely_round:
                return best_cls
            return None

        elif fd >= self.f_faturesdisk and s < self.f_smooth:
            if row[edge_yes_col] >= self.f_edge_onyes:
                return 3
            if row[edge_no_col] >= self.f_edge_onno and row[spiral_col] >= self.f_spiral_yes:
                return 4
            return None

        return None

    def assign_label(self, df: pd.DataFrame,series=False) -> pd.Series|pd.DataFrame:
        """Apply labels"""
        df["label"] = df.apply(lambda r: self._assign_label(r), axis=1)
        super().df = df #pass to base for saving options
        if series:
        
            return df["label"]
        else:
            return df


    def extract_debiased_df(self) -> pd.DataFrame:
        ''' extract debiased columns from raw data'''
        debiased_df = self.df[self.df.columns[self.df.columns.str.endswith('_debiased')]]
        self.df = debiased_df.copy()
        return debiased_df

    def stats(self,original_df:pd.DataFrame, labeled_df: pd.DataFrame) -> None:
        ''' Print stats about the labeling process

        Args:
            original_df (pd.DataFrame): The original dataframe before labeling.
            labeled_df (pd.DataFrame): The dataframe after labeling.

        Returns:
            None: This function prints the statistics to the console.
        '''


        counts = labeled_df["label"].value_counts().sort_index()
        total   = len(original_df)
        labeled = len(labeled_df)
        dropped = total - labeled
        percentage_dropped = (dropped / total) * 100
        logging.info(f"Total samples: {total}")
        logging.info(f"Labeled samples: {labeled}")
        logging.info(f"Dropped samples: {dropped}")
        logging.info(f"Percentage dropped: {percentage_dropped:.2f}%")

        for cls, count in counts.items():
            logging.info(f"{self.CLASS_NAMES.get(cls, 'Unknown')}: {count}")

    def plot(self, labeled_df: pd.DataFrame) -> None:
        counts = labeled_df["label"].value_counts().sort_index()
        plt.figure(figsize=(8, 4))
        plt.bar([self.CLASS_NAMES[i] for i in counts.index], counts.values, color='steelblue')
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("No overlap distribution")
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.show()
                

class Label_Processing_Hubble(Label_Processing):
    ''' Label processing for galaxy using the Hubble Tuning Fork classification'''    
    pass

class Label_Processing_Voting(Label_Processing):
    pass