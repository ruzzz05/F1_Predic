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

# Fusionar datos
df = results.merge(races, on='raceId', how='left')
df = df.merge(drivers, on='driverId', how='left')
df = df.merge(constructors, on='constructorId', how='left')
df = df.merge(qualifying[['raceId', 'driverId', 'position']], on=['raceId', 'driverId'], how='left', suffixes=('', '_qualifying'))

# Filtrar columnas útiles
df = df[['raceId', 'year', 'round', 'circuitId', 'surname', 'constructorRef', 'grid', 'position', 'position_qualifying']]

# Filtrar datos válidos
df = df.dropna(subset=['position', 'position_qualifying'])
df = df[df['position'] != '\\N']
df['position'] = df['position'].astype(int)
df['position_qualifying'] = df['position_qualifying'].astype(int)

# Últimas 2 temporadas
temporadas_recientes = sorted(df['year'].unique())[-2:]
df = df[df['year'].isin(temporadas_recientes)]

# Pilotos actuales
pilotos_actuales = drivers['surname'].unique()

# Codificar
le_driver = LabelEncoder()
le_team = LabelEncoder()

df['driver_encoded'] = le_driver.fit_transform(df['surname'])
df['team_encoded'] = le_team.fit_transform(df['constructorRef'])

# Entrenar modelo
X = df[['grid', 'position_qualifying', 'driver_encoded', 'team_encoded']]
y = df['position']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Obtener equipo principal por piloto (último registrado)
piloto_equipo = df.drop_duplicates('driver_encoded')[['driver_encoded', 'team_encoded']].set_index('driver_encoded')['team_encoded'].to_dict()

# Predicción para todos los pilotos actuales en circuitoId=4
print("\nClasificación simulada en Circuito ID 4:\n")
predicciones = []

for piloto in pilotos_actuales:
    if piloto not in le_driver.classes_:
        print(f"- {piloto}: sin datos suficientes en el histórico, no se incluye.")
        continue

    driver_id = le_driver.transform([piloto])[0]

    # Verificamos si ha corrido en ese circuito
    datos_circuito = df[(df['driver_encoded'] == driver_id) & (df['circuitId'] == 4)]
    
    if not datos_circuito.empty:
        # Usar el promedio de sus datos en el circuito
        row = datos_circuito.iloc[-1]
        x_new = pd.DataFrame([{
            'grid': row['grid'],
            'position_qualifying': row['position_qualifying'],
            'driver_encoded': driver_id,
            'team_encoded': row['team_encoded']
        }])
        pred = int(round(model.predict(x_new)[0]))
        predicciones.append((piloto, pred))
    else:
        # Si no ha corrido en ese circuito, usar sus datos generales (opcional)
        datos_generales = df[df['driver_encoded'] == driver_id]
        if datos_generales.empty:
            print(f"- {piloto}: sin datos suficientes en ningún circuito.")
            continue

        row = datos_generales.iloc[-1]
        x_new = pd.DataFrame([{
            'grid': row['grid'],
            'position_qualifying': row['position_qualifying'],
            'driver_encoded': driver_id,
            'team_encoded': row['team_encoded']
        }])
        pred = int(round(model.predict(x_new)[0]))
        predicciones.append((piloto + " (sin datos en circuito)", pred))

# Ordenar e imprimir
predicciones.sort(key=lambda x: x[1])
for i, (piloto, _) in enumerate(predicciones, 1):
    print(f"{i}. {piloto}")

