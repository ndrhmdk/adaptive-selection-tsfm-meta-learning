import numpy as np
import torch

from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from uni2ts.transform.imputation import CausalMeanImputation
from src.models.base import BaseForecaster, ForecastResult


class Moirai2Forecaster(BaseForecaster):
    """Forecast each variate independently using a batch of univariate series."""

    def __init__(
        self, checkpoint: str = "Salesforce/moirai-2.0-R-small", device: str | None = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        print(f"Loading Moirai-2 from {checkpoint} on {device}")

        self.module = Moirai2Module.from_pretrained(checkpoint)
        self.module = self.module.to(self.device)
        self.module.eval()
        self.quantile_levels = list(self.module.quantile_levels)
        print("Moirai-2 loaded")
        print(f"Quantiles: {self.quantile_levels}")

    def _prepare_history(self, history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 2:
            raise ValueError("history must have shape (context_length, n_variates)")
        if len(history) == 0:
            raise ValueError("history cannot be empty")
        if np.isinf(history).any():
            raise ValueError("history contains infinite values")

        # construct the mask before imputation
        observed_mask = np.isfinite(history)
        clean_history = history.copy()

        # moirai's own prediction helper uses CausalMeanImputation for missing values
        if np.isnan(clean_history).any():
            imputer = CausalMeanImputation()
            # Uni2TS imputation expects time on the last axis.
            clean_history = imputer(clean_history.T).T

        return clean_history.astype(np.float32), observed_mask

    def predict(self, history: np.ndarray, prediction_length: int) -> ForecastResult:
        if prediction_length <= 0:
            raise ValueError("prediction_length must be > 0")
        clean_history, observed_mask = self._prepare_history(history)
        context_length = clean_history.shape[0]
        n_variates = clean_history.shape[1]
        if n_variates == 0:
            raise ValueError("history must contain at least one variate")

        # forecasting head
        model = Moirai2Forecast(
            module=self.module,
            prediction_length=prediction_length,
            context_length=context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        model = model.to(self.device)
        model.eval()

        # Uni2TS 2.0's recursive forecast path requires a univariate target.
        # Treat columns as batch items: (variables, time, 1).
        past_target = torch.tensor(
            clean_history.T, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)
        past_observed_target = torch.tensor(
            observed_mask.T, dtype=torch.bool, device=self.device
        ).unsqueeze(-1)
        past_is_pad = torch.zeros(
            (n_variates, context_length), dtype=torch.bool, device=self.device
        )

        # forecast
        with torch.no_grad():
            quantile_predictions = model(
                past_target=past_target,
                past_observed_target=(past_observed_target),
                past_is_pad=past_is_pad,
            )
        quantile_predictions = quantile_predictions.detach().cpu().numpy()

        # expected moirai-2 output: (variables, quantiles, future_time, 1)
        # for a one-dimensional target some versions may omit the final dimension
        if quantile_predictions.ndim == 3:
            quantile_predictions = quantile_predictions[..., np.newaxis]

        if quantile_predictions.ndim != 4:
            raise ValueError(f"Unexpected Moirai output shape: {quantile_predictions.shape}")

        expected_shape = (n_variates, len(self.quantile_levels), prediction_length, 1)
        if quantile_predictions.shape != expected_shape:
            raise ValueError(
                f"Unexpected Moirai forecast shape:\n- Expected: {expected_shape}\n- Received: {quantile_predictions.shape}"
            )

        # extract quantiles
        levels = np.asarray(self.quantile_levels)
        median_index = int(np.argmin(np.abs(levels - 0.5)))
        lower_index = int(np.argmin(np.abs(levels - 0.1)))
        upper_index = int(np.argmin(np.abs(levels - 0.9)))

        # Restore the project's (horizon, variables) convention and column order.
        predictions = quantile_predictions[:, median_index, :, 0].T
        lower = quantile_predictions[:, lower_index, :, 0].T
        upper = quantile_predictions[:, upper_index, :, 0].T

        return ForecastResult(
            predictions=predictions, lower=lower, upper=upper, model_name="moirai2"
        )
