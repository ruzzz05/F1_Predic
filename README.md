

##🏎️ F1 Position Predictor

This project uses historical Formula 1 data to train a machine learning model that predicts a driver's final race position based on their starting grid position, qualifying results, and other relevant factors.

##📊 Description
A regression model based on RandomForestRegressor from scikit-learn is trained using merged data from:

Race results (results.csv)

Races (races.csv)

Drivers (drivers.csv)

Constructors (constructors.csv)

Qualifying results (qualifying.csv)

The model input features include:

Starting grid position (grid)

Qualifying position (position_qualifying)

Encoded driver ID

Encoded team (constructor) ID

The model predicts the final race position (position) of each driver.

##📌 Data Source:
All CSV files were obtained from this public Kaggle dataset:
Formula 1 World Championship (1950 - 2020) by Rohan Rao

##🧠 How It Works
Data Preprocessing:

Merge multiple CSV files into a single dataset.

Filter useful columns and clean invalid entries.

Focus on the last two seasons of available data.

Encode categorical values like drivers and constructors.

Model Training:

Train a RandomForestRegressor using historical race data.

Prediction:

Simulate predictions for all current drivers at a specific circuit (e.g., circuitId = 4).

If a driver has no past data on that circuit, their most recent general data is used as fallback.

##📁 File Structure
bash
Copiar
Editar
archive/
├── results.csv
├── races.csv
├── drivers.csv
├── constructors.csv
└── qualifying.csv
main.py  # This script
##📦 Requirements
Python 3.8+

pandas

scikit-learn

Install the dependencies with:

bash
Copiar
Editar
pip install pandas scikit-learn
##🚀 Running the Project
Make sure the archive folder contains the necessary CSV files. Then, run:

bash
Copiar
Editar
python main.py
You'll see a simulated classification output for current F1 drivers at the selected circuit.

##📄 License
This project is open-source and available under the MIT License.
