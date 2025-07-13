import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

fastf1.Cache.enable_cache("f1_cache")



# Cargamos la sesion de Mónaco de 2024
session_2024 = fastf1.get_session(2024, 8, "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)


# Pasa los tiempos a segundos
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()


# Calcula el tiempo promedio por sector
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()


# Calcula su tiempo medio, con su media por sector
sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)


# Tiempo estimado en aire limpio en carrera (un poco invent)
clean_air_race_pace = {
    "VER": 93.191067, "HAM": 94.020622, "LEC": 93.418667, "NOR": 93.428600, "ALO": 94.784333,
    "PIA": 93.232111, "RUS": 93.833378, "SAI": 94.497444, "STR": 95.318250, "HUL": 95.345455,
    "OCO": 95.682128
}


# Tiempos de clasificacion 2025
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO",
               "HAM", "STR", "GAS", "ALO", "HUL"],
    "QualifyingTime (s)": [  
        70.669, 69.954, 70.129, None, 71.362, 71.213, 70.063,
        70.942, 70.382, 72.563, 71.994, 70.924, 71.596
    ]
})

# Une cada tiempo de carrera en Aire Limpio a cada piloto
qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)
qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]


# Puntos de cada equipo ( a mayor puntos mejor rendimiento )
team_points = {
    "McLaren": 279, "Mercedes": 147, "Red Bull": 131, "Williams": 51, "Ferrari": 114,
    "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 10, "Alpine": 7
}

# Saca el mayor valor de los puntos del equipo
max_points = max(team_points.values())
# Crea un valor del 0 al 1, recorre cada equipo con sus puntos, y lo divide entre su maximo
team_performance_score = {team: points / max_points for team, points in team_points.items()}


# Asigna a cada piloto su equipo
driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin"
}

# Los une
qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

# Cuantas posiciones baja o sube de media cada piloto en Monaco
average_position_change_monaco = {
    "VER": -1.0, "NOR": 1.0, "PIA": 0.2, "RUS": 0.5, "SAI": -0.3, "ALB": 0.8,
    "LEC": -1.5, "OCO": -0.2, "HAM": 0.3, "STR": 1.1, "GAS": -0.4, "ALO": -0.6, "HUL": 0.0
}
qualifying_2025["AveragePositionChange"] = qualifying_2025["Driver"].map(average_position_change_monaco)

merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["QualifyingTime"] = merged_data["QualifyingTime"]

# Coge los pilotos válidos que sirven y que tienen datos
valid_drivers = merged_data["Driver"].isin(laps_2024["Driver"].unique())
merged_data = merged_data[valid_drivers]

# Le decimos que esos valores son los útiles
X = merged_data[[
    "QualifyingTime", "TeamPerformanceScore", "CleanAirRacePace (s)", "AveragePositionChange"
]]

# Esto es lo que se quiere predecir
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])


# Si faltan datos los añade con la mediana ???????
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Decide que el 30% son de entrenamiento el otro 70% para la prediccion
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=37)


# Crea 100 arboles, cuanto aprende o modifica los datos, profundiad maxima del arbol de 3, y una semilla aleatoria de 37 ---------------------------------------------------------------------
gbr_model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=5, random_state=37)
# Entranando MODELO
gbr_model.fit(X_train, y_train)
# Predice el resultado del tiempo de carreras
merged_data["PredictedRaceTime_GBR (s)"] = gbr_model.predict(X_imputed)

# Modelo adicional: Random Forest ----------------------------------------------------------------------------------------------------------------------------------------------------
rf_model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=37)
rf_model.fit(X_train, y_train)
merged_data["PredictedRaceTime_RF (s)"] = rf_model.predict(X_imputed)

# Ordena los datos de menor a mayor, cuanto menor mejor posicion
final_results_gbr = merged_data.sort_values("PredictedRaceTime_GBR (s)").reset_index(drop=True)
final_results_rf = merged_data.sort_values("PredictedRaceTime_RF (s)").reset_index(drop=True)



print("\n🏁 Predicted 2025 Monaco GP Winner (Gradient Boosting) 🏁\n")
print(final_results_gbr[["Driver", "PredictedRaceTime_GBR (s)"]])

print("\n🏁 Predicted 2025 Monaco GP Winner (Random Forest) 🏁\n")
print(final_results_rf[["Driver", "PredictedRaceTime_RF (s)"]])

y_pred_gbr = gbr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

print(f"\nModel Error (MAE) - Gradient Boosting: {mean_absolute_error(y_test, y_pred_gbr):.2f} seconds")
print(f"Model Error (MAE) - Random Forest:     {mean_absolute_error(y_test, y_pred_rf):.2f} seconds")



# Resultados reales del GP de Mónaco 2025 (ordenados por posición final real)
# Asegúrate de usar los códigos de piloto correctos
real_race_results = [
    "LEC", "PIA", "NOR", "RUS", "VER", "HAM", "ALO", "GAS", "SAI", "OCO", "STR", "ALB", "HUL"
]

# Crea un DataFrame con la posición real
real_results_df = pd.DataFrame({
    "Driver": real_race_results,
    "RealPosition": list(range(1, len(real_race_results) + 1))
})

# Mezcla con las predicciones
comparison_df = final_results_gbr[["Driver", "PredictedRaceTime_GBR (s)"]].copy()
comparison_df["PredictedPosition"] = comparison_df.index + 1  # Posición según el modelo

# Une los datos reales con los predichos
comparison_df = comparison_df.merge(real_results_df, on="Driver", how="left")

# Calcula el error de posición
comparison_df["PositionError"] = (comparison_df["PredictedPosition"] - comparison_df["RealPosition"]).abs()

# Muestra el resumen
print("\n📊 Comparación entre predicción y resultado real del GP de Mónaco 2025:\n")
print(comparison_df[["Driver", "PredictedPosition", "RealPosition", "PositionError"]])

