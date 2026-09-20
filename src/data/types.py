from dataclasses import dataclass
import pandas as pd


@dataclass
class TimeSeriesDataset:
    name: str
    domain: str

    timestamps: pd.DatetimeIndex
    values: pd.DataFrame

    frequency: str | None

    target_columns: list[str]
    primary_target: str | None

    seasonal_period: int | None = None

    @property
    def n_observations(self) -> int:
        return len(self.values)

    @property
    def n_variates(self) -> int:
        return self.values.shape[1]

    @property
    def start_time(self):
        return self.timestamps.min()

    @property
    def end_time(self):
        return self.timestamps.max()
