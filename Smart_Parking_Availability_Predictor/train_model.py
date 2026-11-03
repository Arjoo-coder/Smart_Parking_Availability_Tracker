"""Train, tune, compare and save the models using the saved processed dataset.

    python train_model.py              # with the small hyper-parameter search (about 35 s)
    python train_model.py --no-tune    # faster

Needs data/processed/parking_data_processed.csv (create it with python generate_dataset.py).
Saves models/best_model.joblib and the evaluation files in outputs/results/ and outputs/charts/.
"""

import argparse

from src import train


def main():
    parser = argparse.ArgumentParser(description="Train the parking availability models")
    parser.add_argument("--no-tune", action="store_true", help="skip the hyper-parameter search")
    args = parser.parse_args()
    train.run_training(tune=False if args.no_tune else None)
    print("\nDone. Next: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
