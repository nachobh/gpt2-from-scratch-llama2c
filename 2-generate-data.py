#!/usr/bin/env python3
"""
PASO 2: Generar dataset de entrenamiento para llama2.c

El modelo necesita texto natural para aprender patrones.
Generamos texto sobre combinaciones de Euromillón con contexto estadístico.
"""

import random
import os

# Frecuencias históricas más sorteadas
NUMEROS_FRECUENTES = [7, 19, 23, 32, 42, 17, 38, 14, 27, 45, 5, 11, 28, 34, 49]
ESTRELLAS_FRECUENTES = [2, 3, 8, 11, 5]

# Textos descriptivos para dar contexto al modelo
TEMPLATES_TEXTO = [
    "En el sorteo de Euromillón salieron los siguientes números: ",
    "Último resultado de Euromillón: ",
    "Combinación ganadora: ",
    "Números del día: ",
    "Sorteo reciente: ",
    "Resultados de Euromillón: ",
    "Números de hoy: ",
    "Combinación de esta noche: ",
]

# Textos de contexto estadístico
CONTEXTO_ESTADÍSTICO = [
    "\n\nLos números más frecuentes son: ",
    "\n\nSegún estadísticas históricas, los números calientes son: ",
    "\n\nLos números que más salen son: ",
]

def generar_combinacion():
    pesos_n = [2.0 if n in NUMEROS_FRECUENTES else 1.0 for n in range(1, 51)]
    pesos_e = [2.5 if e in ESTRELLAS_FRECUENTES else 1.0 for e in range(1, 13)]
    
    numeros = []
    disp_n = list(range(1, 51))
    w_n = pesos_n[:]
    for _ in range(5):
        n = random.choices(disp_n, weights=w_n, k=1)[0]
        numeros.append(n)
        idx = disp_n.index(n)
        disp_n.pop(idx)
        w_n.pop(idx)
    
    estrellas = []
    disp_e = list(range(1, 13))
    w_e = pesos_e[:]
    for _ in range(2):
        e = random.choices(disp_e, weights=w_e, k=1)[0]
        estrellas.append(e)
        idx = disp_e.index(e)
        disp_e.pop(idx)
        w_e.pop(idx)
    
    return sorted(numeros), sorted(estrellas)

def generar_texto():
    """Genera un fragmento de texto natural sobre Euromillón."""
    # 30% probabilidad de incluir contexto estadístico
    texto = ""
    if random.random() < 0.3:
        texto += random.choice(CONTEXTO_ESTADÍSTICO)
        nums = ", ".join(str(n) for n in NUMEROS_FRECUENTES[:random.randint(3, 8)])
        texto += nums + ". "
    
    # Agregar combinación
    texto += random.choice(TEMPLATES_TEXTO)
    
    nums, stars = generar_combinacion()
    n_str = ", ".join(str(n) for n in nums)
    s_str = ", ".join(str(s) for s in stars)
    texto += f"{n_str} con las estrellas {s_str}"
    
    # Agregar información adicional aleatoria
    if random.random() < 0.4:
        extras = [
            f" (aproximadamente {random.randint(1000000, 1000000000):,]}€ en premio mayor)",
            f" en {random.randint(2020, 2026)}",
            f" en la semana {random.randint(1, 52)}",
            f" (acumulación creciente)",
            f" (premio estimado: {random.randint(50, 200)}M €)",
        ]
        texto += random.choice(extras)
    
    return texto

def main():
    os.makedirs("data", exist_ok=True)
    
    NUM_FRAGMENTS = 50000  # Frases de contexto
    
    print(f"Generando {NUM_FRAGMENTS} fragmentos de texto...")
    
    texto_completo = ""
    for i in range(NUM_FRAGMENTS):
        fragmento = generar_texto()
        # Cada fragmento termina con un salto de línea
        texto_completo += fragmento + "\n"
        
        if (i + 1) % 10000 == 0:
            print(f"  Progreso: {i + 1}/{NUM_FRAGMENTS}")
    
    # Guardar dataset
    output_path = "data/euromillon_train.txt"
    with open(output_path, "w") as f:
        f.write(texto_completo)
    
    print(f"\n✓ Dataset generado: {output_path}")
    print(f"  Tamaño: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    print(f"  Líneas: {texto_completo.count(chr(10))}")
    
    # Generar un texto de ejemplo
    print("\n--- Ejemplo de texto generado ---")
    print(texto_completo[:500] + "...")
    print("-------------------------------")

if __name__ == "__main__":
    main()