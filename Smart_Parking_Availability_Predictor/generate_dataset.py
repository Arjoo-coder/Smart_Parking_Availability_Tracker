"""Create the SIMULATED parking dataset and prepare it for training.

    python generate_dataset.py

This is only needed if you want to (re)create the data without training. It writes
  data/raw/parking_data_simulated.csv         (raw simulated readings, with a few deliberate problems)
  data/processed/parking_data_processed.csv   (cleaned + engineered features)
The data is SIMULATED (rule-generated), not real parking data. The same seed always
gives the same dataset. Next step: python train_model.py
"""

from src import config, data_generator, preprocessing


def main():
    config.ensure_directories()
    data_generator.generate_and_save()
    preprocessing.run_preprocessing(is_simulated=True)
    print("\nDone. Next: python train_model.py")


if __name__ == "__main__":
    main()
