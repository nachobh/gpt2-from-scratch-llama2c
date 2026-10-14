#!/usr/bin/env python3
"""
Genera combinaciones de Euromillón usando el modelo entrenado.
"""

import subprocess
import sys
import os
import glob

# Configuración
MODEL_SIZE = "tiny"
OUTPUT_DIR = "models"
GUF_NAME = f"euromillon-gpt2-{MODEL_SIZE}.gguf"
NUM_COMBINACIONES = int(sys.argv[1]) if len(sys.argv) > 1 else 5

# Verificar modelo
model_path = os.path.join(OUTPUT_DIR, GUF_NAME)
if not os.path.exists(model_path):
    print(f"❌ Modelo no encontrado: {model_path}")
    print("   Ejecuta primero: ./run.sh convert")
    sys.exit(1)

# Prompt para generar combinaciones
prompt = f"""Genera {NUM_COMBINACIONES} combinaciones de Euromillón.
Cada combinación debe tener 5 números (1-50) y 2 estrellas (1-12).
Formato: NUMEROS + ESTRELLAS
Ejemplo: 7, 19, 23, 32, 42 + 2, 8

Combinaciones:
"""

# Ejecutar llama-cli
cmd = [
    "/usr/local/bin/llama-cli",
    "-m", model_path,
    "-p", prompt,
    "-n", "256",
    "-t", "16",
    "--temp", "0.7",
    "--top-p", "0.95",
    "--no-display-prompt",
    "-c", "512"
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    output = result.stdout
except Exception as e:
    print(f"❌ Error ejecutando llama-cli: {e}")
    sys.exit(1)

# Extraer combinaciones del output
import re
pattern = r'(\d+(?:\s*,\s*\d+){4})\s*\+\s*(\d+(?:\s*,\s*\d+)?)'
matches = re.findall(pattern, output)

if not matches:
    print("⚠️ No se encontraron combinaciones en el output")
    print("   Intentando con patrón alternativo...")
    # Patrón alternativo
    pattern2 = r'(\d{1,2})\s*[,]\s*(\d{1,2})\s*[,]\s*(\d{1,2})\s*[,]\s*(\d{1,2})\s*[,]\s*(\d{1,2})\s*[+]\s*(\d{1,2})\s*[,]\s*(\d{1,2})'
    matches2 = re.findall(pattern2, output)
    if matches2:
        for m in matches2:
            nums = [int(x) for x in m[:5]]
            stars = [int(x) for x in m[5:]]
            print(f"  {', '.join(map(str, nums))} + {', '.join(map(str, stars))}")
    else:
        print("   Output generado:")
        print(output[:500])
else:
    for i, (nums_str, stars_str) in enumerate(matches, 1):
        nums = [int(x.strip()) for x in nums_str.split(',')]
        stars = [int(x.strip()) for x in stars_str.split(',')]
        print(f"  {i}. {', '.join(map(str, nums))} + {', '.join(map(str, stars))}")
