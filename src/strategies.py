from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
from pandera.typing import DataFrame

from src.schema import (
    CoefSchema,
    XRDataSheetSchema,
    XRsDataSheetSchema,
    npDataSheetSchema,
    pDataSheetSchema,
    cDataSheetSchema,
    uDataSheetSchema,
)

class Strategy(ABC):
    @abstractmethod
    def _validate_coef(self):
        pass

    @abstractmethod
    def set_data(self, data_path: str, *args, **kwargs):
        pass

    @abstractmethod
    def process_data(self):
        pass

    @abstractmethod
    def plot(self):
        pass

class XRChartStrategy(Strategy):
    data_df: DataFrame[XRDataSheetSchema]
    coef_df: DataFrame[CoefSchema]
    required_coef = {"A2", "D3", "D4"}

    def _validate_coef(self):
        missing_coef = self.required_coef - set(self.coef_df.columns)
        if missing_coef:
            raise ValueError(f"Missing coefficients in the coefficient DataFrame: {missing_coef}")

    def set_data(self, data_path: str, coef_path: str):
        self.data_df = pd.read_csv(data_path)
        self.coef_df = pd.read_csv(coef_path)
        self._validate_coef()

    def process_data(self):
        self.bar_X = self.data_df.mean(axis=1)
        self.bar_bar_X = self.bar_X.mean()
        self.R = self.data_df.max(axis=1) - self.data_df.min(axis=1)
        self.bar_R = self.R.mean()
        self.sample_size = self.data_df.shape[1]
        self.A2 = self.coef_df.loc[self.coef_df["n"] == self.sample_size, "A2"].values[0]
        self.D3 = self.coef_df.loc[self.coef_df["n"] == self.sample_size, "D3"].values[0]
        self.D4 = self.coef_df.loc[self.coef_df["n"] == self.sample_size, "D4"].values[0]
        self.UCL_X = self.bar_bar_X + self.A2 * self.bar_R
        self.LCL_X = self.bar_bar_X - self.A2 * self.bar_R
        self.UCL_R = self.D4 * self.bar_R
        self.LCL_R = self.D3 * self.bar_R

    def plot(self):
        # Plot X-Control Chart
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.bar_X, marker="o", linestyle="-", color="blue", label="X-bar")
        plt.axhline(self.bar_bar_X, color="green", linestyle="--", label="X-bar Average")
        plt.axhline(self.UCL_X, color="red", linestyle="--", label="UCL")
        plt.axhline(self.LCL_X, color="red", linestyle="--", label="LCL")
        plt.title("X-Control Chart")
        plt.xlabel("Sample Number")
        plt.ylabel("X-bar")
        plt.legend()
        plt.grid()
        # Plot R-Control Chart
        plt.subplot(2, 1, 2)
        plt.plot(self.R, marker="o", linestyle="-", color="blue", label="R")
        plt.axhline(self.bar_R, color="green", linestyle="--", label="R Average")
        plt.axhline(self.UCL_R, color="red", linestyle="--", label="UCL")
        if self.LCL_R > 0:
            plt.axhline(self.LCL_R, color="red", linestyle="--", label="LCL")
        plt.title("R-Control Chart")
        plt.xlabel("Sample Number")
        plt.ylabel("R")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        # Show the plot
        plt.show()

class XRsChartStrategy(Strategy):
    data_df: DataFrame[XRsDataSheetSchema]
    coef_df: DataFrame[CoefSchema]
    required_coef = {"D3", "D4", "E2"}

    def _validate_coef(self):
        missing_coef = self.required_coef - set(self.coef_df.columns)
        if missing_coef:
            raise ValueError(f"Missing coefficients in the coefficient DataFrame: {missing_coef}")
    
    def set_data(self, data_path: str, coef_path: str):
        self.data_df = pd.read_csv(data_path)
        self.coef_df = pd.read_csv(coef_path)
        self._validate_coef()
    
    def process_data(self):
        self.bar_X = self.data_df["X"].mean()
        self.Rs = self.data_df["X"].diff().abs().dropna()
        self.bar_Rs = self.Rs.mean()
        self.sample_size = 2
        self.D3 = self.coef_df.loc[self.coef_df["n"] == self.sample_size, "D3"].values[0]
        self.D4 = self.coef_df.loc[self.coef_df["n"] == self.sample_size, "D4"].values[0]
        self.E2 = self.coef_df.loc[self.coef_df["n"] == self.sample_size, "E2"].values[0]
        self.UCL_X = self.bar_X + self.E2 * self.bar_Rs
        self.LCL_X = self.bar_X - self.E2 * self.bar_Rs
        self.UCL_R = self.D4 * self.bar_Rs
        self.LCL_R = self.D3 * self.bar_Rs
    
    def plot(self):
        # Plot X-Control Chart
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.data_df["X"], marker="o", linestyle="-", color="blue", label="X")
        plt.axhline(self.bar_X, color="green", linestyle="--", label="X Average")
        plt.axhline(self.UCL_X, color="red", linestyle="--", label="UCL")
        plt.axhline(self.LCL_X, color="red", linestyle="--", label="LCL")
        plt.title("X-Control Chart")
        plt.xlabel("Sample Number")
        plt.ylabel("X")
        plt.legend()
        plt.grid()
        # Plot Rs-Control Chart
        plt.subplot(2, 1, 2)
        plt.plot(self.Rs, marker="o", linestyle="-", color="blue", label="R")
        plt.axhline(self.bar_Rs, color="green", linestyle="--", label="R Average")
        plt.axhline(self.UCL_R, color="red", linestyle="--", label="UCL")
        if self.LCL_R > 0:
            plt.axhline(self.LCL_R, color="red", linestyle="--", label="LCL")
        plt.title("Rs-Control Chart")
        plt.xlabel("Sample Number")
        plt.ylabel("Rs")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        # Show the plot
        plt.show()

class MedianChartStrategy(Strategy):
    data_df: DataFrame[XRDataSheetSchema]
    coef_df: DataFrame[CoefSchema]
    required_coef = {"A4", "D3", "D4"}

    def _validate_coef(self):
        missing_coef = self.required_coef - set(self.coef_df.columns)
        if missing_coef:
            raise ValueError(f"Missing coefficients in the coefficient DataFrame: {missing_coef}")

    def set_data(self, data_path: str, coef_path: str):
        self.data_df = pd.read_csv(data_path)
        self.coef_df = pd.read_csv(coef_path)
        self._validate_coef()

    def process_data(self):
        self.median = self.data_df.median(axis=1)
        self.bar_median = self.median.mean()
        self.R = self.data_df.max(axis=1) - self.data_df.min(axis=1)
        self.bar_R = self.R.mean()
        self.sample_size = self.data_df.shape[1]
        self.A4 = self.coef_df.loc[self.coef_df["n"] == self.sample_size, "A4"].values[0]
        self.D3 = self.coef_df.loc[self.coef_df["n"] == self.sample_size, "D3"].values[0]
        self.D4 = self.coef_df.loc[self.coef_df["n"] == self.sample_size, "D4"].values[0]
        self.UCL_median = self.bar_median + self.A4 * self.bar_R
        self.LCL_median = self.bar_median - self.A4 * self.bar_R
        self.UCL_R = self.D4 * self.bar_R
        self.LCL_R = self.D3 * self.bar_R
    
    def plot(self):
        # Plot Median-Control Chart
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.median, marker="o", linestyle="-", color="blue", label="Median")
        plt.axhline(self.bar_median, color="green", linestyle="--", label="Median Average")
        plt.axhline(self.UCL_median, color="red", linestyle="--", label="UCL")
        plt.axhline(self.LCL_median, color="red", linestyle="--", label="LCL")
        plt.title("Median-Control Chart")
        plt.xlabel("Sample Number")
        plt.ylabel("Median")
        plt.legend()
        plt.grid()
        # Plot R-Control Chart
        plt.subplot(2, 1, 2)
        plt.plot(self.R, marker="o", linestyle="-", color="blue", label="R")
        plt.axhline(self.bar_R, color="green", linestyle="--", label="R Average")
        plt.axhline(self.UCL_R, color="red", linestyle="--", label="UCL")
        if self.LCL_R > 0:
            plt.axhline(self.LCL_R, color="red", linestyle="--", label="LCL")
        plt.title("R-Control Chart")
        plt.xlabel("Sample Number")
        plt.ylabel("R")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        # Show the plot
        plt.show()

class npChartStrategy(Strategy):
    data_df: DataFrame[npDataSheetSchema]

    def _validate_coef(self):
        pass

    def set_data(self, data_path: str, sample_size: int):
        self.data_df = pd.read_csv(data_path)
        self.sample_size = sample_size

    def process_data(self):
        self.n_bar_p = self.data_df["np"].mean()
        self.bar_p = self.n_bar_p / self.sample_size
        self.UCL = self.n_bar_p + 3 * ((self.n_bar_p * (1 - self.bar_p))) ** 0.5
        self.LCL = self.n_bar_p - 3 * ((self.n_bar_p * (1 - self.bar_p))) ** 0.5
        if self.LCL < 0:
            self.LCL = 0
    
    def plot(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data_df["np"], marker="o", linestyle="-", color="blue", label="np Chart")
        plt.axhline(self.n_bar_p, color="green", linestyle="--", label="np Average")
        plt.axhline(self.UCL, color="red", linestyle="--", label="UCL")
        if self.LCL > 0:
            plt.axhline(self.LCL, color="red", linestyle="--", label="LCL")
        plt.title("np-Control Chart")
        plt.xlabel("Sample Number")
        plt.ylabel("Number of Defects")
        plt.legend()
        plt.grid()
        # Show the plot
        plt.show()

class pChartStrategy(Strategy):
    data_df: DataFrame[pDataSheetSchema]

    def _validate_coef(self):
        pass

    def set_data(self, data_path: str):
        self.data_df = pd.read_csv(data_path)

    def process_data(self):
        self.p = self.data_df["np"] / self.data_df["n"]
        self.bar_p = self.data_df["np"].sum() / self.data_df["n"].sum()
        self.UCL = self.bar_p + 3 * ((self.bar_p * (1 - self.bar_p)) / self.data_df["n"]) ** 0.5
        self.LCL = self.bar_p - 3 * ((self.bar_p * (1 - self.bar_p)) / self.data_df["n"]) ** 0.5
        self.LCL[self.LCL < 0] = 0  # Ensure LCL is not negative
    
    def plot(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.p, marker="o", linestyle="-", color="blue", label="p Chart")
        plt.axhline(self.bar_p, color="green", linestyle="--", label="p Average")
        plt.plot(self.UCL, color="red", linestyle="--", label="UCL")
        plt.plot(self.LCL, color="red", linestyle="--", label="LCL")
        plt.title("p-Control Chart")
        plt.xlabel("Sample Number")
        plt.ylabel("Proportion of Defects")
        plt.legend()
        plt.grid()
        # Show the plot
        plt.show()

class cChartStrategy(Strategy):
    data_df: DataFrame[cDataSheetSchema]

    def _validate_coef(self):
        pass

    def set_data(self, data_path: str):
        self.data_df = pd.read_csv(data_path)

    def process_data(self):
        self.c = self.data_df["c"].mean()
        self.UCL = self.c + 3 * (self.c) ** 0.5
        self.LCL = self.c - 3 * (self.c) ** 0.5
        if self.LCL < 0:
            self.LCL = 0
    
    def plot(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data_df["c"], marker="o", linestyle="-", color="blue", label="c Chart")
        plt.axhline(self.c, color="green", linestyle="--", label="c Average")
        plt.axhline(self.UCL, color="red", linestyle="--", label="UCL")
        if self.LCL > 0:
            plt.axhline(self.LCL, color="red", linestyle="--", label="LCL")
        plt.title("c-Control Chart")
        plt.xlabel("Sample Number")
        plt.ylabel("Count of Defects")
        plt.legend()
        plt.grid()
        # Show the plot
        plt.show()

class uChartStrategy(Strategy):
    data_df: DataFrame[uDataSheetSchema]

    def _validate_coef(self):
        pass

    def set_data(self, data_path: str):
        self.data_df = pd.read_csv(data_path)

    def process_data(self):
        self.u = self.data_df["c"] / self.data_df["n"]
        self.bar_u = self.data_df["c"].sum() / self.data_df["n"].sum()
        self.UCL = self.bar_u + 3 * ((self.bar_u) / self.data_df["n"]) ** 0.5
        self.LCL = self.bar_u - 3 * ((self.bar_u) / self.data_df["n"]) ** 0.5
        self.LCL[self.LCL < 0] = 0
    
    def plot(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.u, marker="o", linestyle="-", color="blue", label="u Chart")
        plt.axhline(self.bar_u, color="green", linestyle="--", label="u Average")
        plt.plot(self.UCL, color="red", linestyle="--", label="UCL")
        plt.plot(self.LCL, color="red", linestyle="--", label="LCL")
        plt.title("u-Control Chart")
        plt.xlabel("Sample Number")
        plt.ylabel("Defects per Unit")
        plt.legend()
        plt.grid()
        # Show the plot
        plt.show()
