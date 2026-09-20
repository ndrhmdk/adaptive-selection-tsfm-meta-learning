from src.models.moirai import Moirai2Forecaster
from src.evaluation.experiment import run_single_forecast

def main():
    model = Moirai2Forecaster()
    result = run_single_forecast(
        forecaster=model,
        dataset_name="ETTh1",
        context_length=512,
        prediction_length=96)
    
    print("\nResults:")
    
    for key, value in (result["metrics"].items()):
        if (key != "mase_per_variable"):
            print(f"{key:25}: {value}")

if __name__ == "__main__":
    main()