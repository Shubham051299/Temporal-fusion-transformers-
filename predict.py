import pandas as pd
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import os

# Load preprocessed data
total_df = pd.read_csv("preprocessed_data_with_turbulence.csv")

# Ensure consistent date format with timezone
total_df['date'] = pd.to_datetime(total_df['date'])

# Create Time_idx column for individual series
total_df['time_idx'] = total_df.groupby('ticker').cumcount()

# Define parameters
context_length = 30
prediction_length = 5
TEST_START_DATE = pd.Timestamp('2023-01-03', tz='UTC-05:00')

# Get the prediction start index
boundary = total_df[total_df['date'] == TEST_START_DATE]
if not boundary.empty:
    prediction_start_idx = boundary['time_idx'].iloc[0]
    print(f"Prediction start index: {prediction_start_idx}")
else:
    raise ValueError(f"No data found for the specified TEST_START_DATE: {TEST_START_DATE}")

drop_cols = ['date', 'close', 'ticker']
time_varying_known_reals = [s for s in total_df.columns if s not in drop_cols]

# Prepare datasets
context_length = 30
prediction_length = 5
training = TimeSeriesDataSet(
    total_df,
    time_idx="time_idx",
    target="close",
    group_ids=["ticker"],
    static_categoricals=["ticker"],
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=["close"],
    max_encoder_length=context_length,
    max_prediction_length=prediction_length,
    add_relative_time_idx=True,
    target_normalizer=GroupNormalizer(groups=['ticker'], transformation="softplus")
)

validation = TimeSeriesDataSet.from_dataset(training, total_df, min_prediction_idx=prediction_start_idx)

batch_size = 64
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)

# Load the best model
best_model_path = "lightning_logs/lightning_logs/version_9/checkpoints/epoch=49-step=117300.ckpt"  # Update with the correct path
best_model_path = best_model_path.replace("\\", "/")  # Ensure path is correctly formatted for the operating system
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# Make predictions
predictions = best_tft.predict(val_dataloader, return_index=True)

# Unpack the returned values correctly
predictions_output, predictions_index = predictions.output, predictions.index

# Save predictions
predictions_df = pd.DataFrame(predictions_output.cpu().numpy())
predictions_df['ticker'] = predictions_index['ticker']
predictions_df['time_idx'] = predictions_index['time_idx']

# Map time_idx back to dates
time_idx_to_date = total_df[['time_idx', 'date']].drop_duplicates()
predictions_df = predictions_df.merge(time_idx_to_date, on='time_idx', how='left')

# Save the predictions to a CSV file
predictions_df.to_csv("predictions.csv", index=False)

print("Predictions saved to predictions.csv")
