import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Cargar datasets
results = pd.read_csv("archive/results.csv")
races = pd.read_csv("archive/races.csv")
drivers = pd.read_csv("archive/drivers.csv")
constructors = pd.read_csv("archive/constructors.csv")
qualifying = pd.read_csv("archive/qualifying.csv")

# 🔒 1. Filtrar solo pilotos actuales desde drivers.csv
pilotos_actuales_ids = set(drivers['driverId'].unique())

# 🧹 2. Filtrar results y qualifying desde el principio
results = results[results['driverId'].isin(pilotos_actuales_ids)]
qualifying = qualifying[qualifying['driverId'].isin(pilotos_actuales_ids)]

# 🔗 3. Merge
df = results.merge(races, on='raceId', how='left')
df = df.merge(drivers, on='driverId', how='left')
df = df.merge(constructors, on='constructorId', how='left')
df = df.merge(qualifying[['raceId', 'driverId', 'position']], 
              on=['raceId', 'driverId'], 
              how='left', 
              suffixes=('', '_qualifying'))

# ✂️ 4. Columnas útiles
df = df[['raceId', 'year', 'round', 'driverId', 'surname', 'constructorRef', 'grid', 'position', 'position_qualifying']]

# ❌ 5. Limpieza de datos inválidos
df = df.dropna(subset=['position', 'position_qualifying'])
df = df[df['position'] != '\\N']
df['position'] = df['position'].astype(int)
df['position_qualifying'] = df['position_qualifying'].astype(int)

# 🔤 6. Codificar variables categóricas
le_driver = LabelEncoder()
le_team = LabelEncoder()

df['driver_encoded'] = le_driver.fit_transform(df['surname'])
df['team_encoded'] = le_team.fit_transform(df['constructorRef'])

# 🧠 7. Features y target
X = df[['grid', 'position_qualifying', 'driver_encoded', 'team_encoded']]
y = df['position']

# 🧪 8. Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔮 9. Predicción
X_test = X_test.copy()
X_test['predicted_position'] = model.predict(X_test)

# 🏁 10. Eliminar duplicados por piloto
X_test_unique = X_test.sort_values('predicted_position').drop_duplicates('driver_encoded')

# 📢 11. Mostrar resultados
print("\nClasificación predicha (simulada):\n")
top_n = min(20, len(X_test_unique))

for i, row in enumerate(X_test_unique.head(top_n).itertuples(), 1):
    driver_name = le_driver.inverse_transform([row.driver_encoded])[0]
    print(f"{i}. {driver_name} (posición estimada: {round(row.predicted_position, 2)})")
