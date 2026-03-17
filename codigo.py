import gymnasium as gym
import numpy as np
from river import stream
from river import compose
from river import preprocessing
from river import linear_model
from river import naive_bayes
from river.tree import HoeffdingTreeClassifier
from river import metrics

# ==========================================
# Configuración previa
# ==========================================

datafile = 'CartPoleInstances.csv'

# Devuelve los nombres de los atributos de entrada
def AttributeNames():
    return ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']

# Devuelve el nombre del atributo objetivo
def ClassName():
    return 'action'

# Conversores para transformar las filas del CSV a los tipos de datos correctos
converters = {i: float for i in AttributeNames()}
converters[ClassName()] = lambda x: int(float(x))

# ==========================================
# Definición de modelos
# ==========================================
# Naive-Bayes Gaussiano (baseline)
model_nb = naive_bayes.GaussianNB()

# Hoeffding Tree
model_ht = HoeffdingTreeClassifier()

# Regresión Logística
model_lr = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression()
)

models = {
    "Naive Bayes (Baseline)": model_nb,
    "Hoeffding Tree": model_ht,
    "Logistic Regression": model_lr
}

# Métricas de Accuracy para cada modelo
accuracies = {name: metrics.Accuracy() for name in models}

# ==========================================
# Entrenamiento
# ==========================================

# Carga del flujo de datos
flujo = stream.iter_csv(datafile, converters=converters, target=ClassName())

# Bucle prequential (Test-then-Train)
for x, y in flujo:
    for name, model in models.items():
        # 1. Predecir (Test)
        y_pred = model.predict_one(x)
        
        # Actualizamos la métrica si el modelo ya puede predecir
        if y_pred is not None:
            accuracies[name].update(y, y_pred)
            
        # 2. Aprender (Train)
        model.learn_one(x, y)

print("\n--- RESULTADOS DEL ENTRENAMIENTO (ACCURACY) ---")
for name, acc in accuracies.items():
    print(f"{name}: {acc.get() * 100:.2f}%")

# ==========================================
# Evaluación (Test)
# ==========================================
print("\n--- INICIANDO EVALUACIÓN EN GYMNASIUM ---")

env = gym.make('CartPole-v1', render_mode='human') # render_mode='human' para ver la simulación
episodes = 10
attr_names = AttributeNames()

test_results = {name: [] for name in models}

for name, model in models.items():
    print(f"\nEvaluando modelo: {name}")
    for ep in range(episodes):
        obs, _ = env.reset()
        env.render()
        total_reward = 0
        truncated, done = False, False
        
        while not (truncated or done):
            # Transformamos la observación (numpy array) en diccionario para River
            x = {attr_names[i]: obs[i] for i in range(len(attr_names))}
            
            # El modelo predice la mejor acción
            action = model.predict_one(x)
            
            """""
            # Fallback de seguridad en caso de que un modelo falle la predicción
            if action is None:
                action = env.action_space.sample()
            """""
                
            # Ejecutamos la acción en el entorno
            obs, r, done, truncated, _ = env.step(action)
            env.render()
            
            # Acumulamos la recompensa
            total_reward += r
            
        test_results[name].append(total_reward)
        print(f"  Episodio {ep + 1}: Recompensa = {total_reward}")

env.close()

print("\n--- RESULTADOS DE EVALUACIÓN (TEST) ---")
for name, rewards in test_results.items():
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"{name}: Recompensa Media = {mean_reward:.2f} ± {std_reward:.2f}")