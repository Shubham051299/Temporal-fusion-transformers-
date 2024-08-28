Temporal Fusion Transformer for Stock Prediction
Project Overview
This repository contains a machine learning project aimed at forecasting stock prices using a Temporal Fusion Transformer (TFT). The project leverages historical stock price data, volume information, and static covariates such as sector classification to make quantile forecasts.


Repository Contents
- `analysis.py`: Script for performing exploratory data analysis and feature inspection.
- `preprocessed.py`: Contains the data preprocessing steps to prepare the dataset for training.
- `predict.py`: Script used to make predictions using the trained TFT model.
- `traningtft.py`: Core script for training the Temporal Fusion Transformer model.
- `preprocessed_data.csv`: The dataset after initial preprocessing.
- `preprocessed_data_with_turbulence.csv`: The dataset with added market turbulence indicators.
- `predictions.csv`: Model predictions output.
- `sorted_stock_predictions_summary.csv`: Summarized predictions sorted based on performance metrics.
Setup and Installation
To get this project up and running on your local machine, follow these steps:
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd <repository-directory>
   ```
Usage
To train the model:
```
python traningtft.py
```
To generate predictions:
```
python predict.py
```
How to Contribute
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
License
Distributed under the MIT License. See `LICENSE` for more information.
Contact
shubham wagh - shubhamwagh0512@gmail.com
Project Link: gitthub.com/Shubham051299/Temporal-fusion-transformers
Acknowledgements
- This project utilizes data and libraries that are freely available and were essential in the successful implementation of the Temporal Fusion Transformer.
- Special thanks to everyone who has contributed to the development of the libraries and tools used in this project.
