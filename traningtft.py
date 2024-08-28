import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss, MAE, MAPE
from pytorch_forecasting.data import GroupNormalizer

# Load preprocessed data
total_df = pd.read_csv("preprocessed_data_with_turbulence.csv")

# Ensure consistent date format with timezone
total_df['date'] = pd.to_datetime(total_df['date'])

# Print unique dates to find a suitable TEST_START_DATE
unique_dates = total_df['date'].unique()
print("Unique dates in the dataset:")
print(unique_dates)

# Choose the test start date and ensure it has the correct timezone
TEST_START_DATE = pd.Timestamp('2023-01-03', tz='UTC-05:00')

# Check if TEST_START_DATE is present in the dataframe
if TEST_START_DATE in total_df['date'].values:
    print(f"{TEST_START_DATE} is present in the dataframe.")
else:
    print(f"{TEST_START_DATE} is NOT present in the dataframe. Please choose a date from the unique dates printed above.")
    raise ValueError(f"No data found for the specified TEST_START_DATE: {TEST_START_DATE}")

# Create Time_idx column for individual series
total_df['time_idx'] = total_df.groupby('ticker').cumcount()

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
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)

# Custom callback to retrieve training and validation loss
class MetricTracker(pl.Callback):
    def __init__(self):
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_mae_list = []
        self.val_mae_list = []
        self.train_mape_list = []
        self.val_mape_list = []

    def on_train_epoch_end(self, trainer, module):
        train_loss = trainer.logged_metrics.get('train_loss_epoch', None)
        if train_loss is not None:
            self.train_loss_list.append(train_loss.item())
        train_mae = trainer.logged_metrics.get('train_mae_epoch', None)
        if train_mae is not None:
            self.train_mae_list.append(train_mae.item())
        train_mape = trainer.logged_metrics.get('train_mape_epoch', None)
        if train_mape is not None:
            self.train_mape_list.append(train_mape.item())

    def on_validation_epoch_end(self, trainer, module):
        val_loss = trainer.logged_metrics.get('val_loss', None)
        if val_loss is not None:
            self.val_loss_list.append(val_loss.item())
        val_mae = trainer.logged_metrics.get('val_mae', None)
        if val_mae is not None:
            self.val_mae_list.append(val_mae.item())
        val_mape = trainer.logged_metrics.get('val_mape', None)
        if val_mape is not None:
            self.val_mape_list.append(val_mape.item())

cb = MetricTracker()

# Set random seed
pl.seed_everything(42)

# Hyper parameters
gradient_clip_val = 0.1
hidden_size = 200
dropout = 0.1
hidden_continuous_size = 100
attention_head_size = 4
learning_rate = 0.0001

# Configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=20, verbose=True, mode="min")
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("lightning_logs")

trainer = pl.Trainer(
    max_epochs=50,
    accelerator="cuda",
    enable_model_summary=True,
    gradient_clip_val=gradient_clip_val,
    callbacks=[lr_logger, early_stop_callback, cb],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=learning_rate,
    hidden_size=hidden_size,
    attention_head_size=attention_head_size,
    dropout=dropout,
    hidden_continuous_size=hidden_continuous_size,
    loss=QuantileLoss(),
    logging_metrics=[MAE(), MAPE()],
    log_interval=10,
    log_val_interval=1,
    optimizer="Adam",
    reduce_on_plateau_patience=4,
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# Fit the network
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# Save the best model path
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"Best model path: {best_model_path}")

# Save MetricTracker results
import pickle
with open("metric_tracker.pkl", "wb") as f:
    pickle.dump({
        "train_loss_list": cb.train_loss_list,
        "val_loss_list": cb.val_loss_list,
        "train_mae_list": cb.train_mae_list,
        "val_mae_list": cb.val_mae_list,
        "train_mape_list": cb.train_mape_list,
        "val_mape_list": cb.val_mape_list
    }, f)
