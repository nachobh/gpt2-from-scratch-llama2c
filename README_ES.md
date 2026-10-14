# Euromillón llama2c - Generador de Combinaciones con IA

Genera combinaciones de Euromillón entrenando un modelo GPT-2 desde cero con llama2.c.

## Requisitos

- gcc (compilador C)
- git
- llama-cli (incluido en tu sistema)
- Python 3.13+

## Pipeline Completo

### Paso 1: Clonar y compilar llama2.c
```bash
./run.sh setup
```

### Paso 2: Generar dataset de entrenamiento
```bash
./run.sh data 50000
```
Genera 50,000 frases sobre Euromillón con combinaciones ponderadas por frecuencia histórica.

### Paso 3: Entrenar modelo GPT-2
```bash
./run.sh train tiny
```

Opciones de modelo:
- **tiny**: 8 layers, 8 heads, 512 embedding (~117M parámetros) - Rápido (~30 min)
- **medium**: 12 layers, 12 heads, 768 embedding (~354M parámetros) - Medio (~2 horas)
- **large**: 24 layers, 16 heads, 1024 embedding (~774M parámetros) - Lento (~6 horas)

### Paso 4: Convertir a GGUF
```bash
./run.sh convert
```

### Paso 5: Generar combinaciones
```bash
./run.sh generate 10
```

## Pipeline Completo (todo junto)
```bash
./run.sh all tiny
```

## Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `data/euromillon_train.txt` | Dataset de entrenamiento |
| `models/euromillon-gpt2-tiny.bin` | Modelo entrenado (formato llama2.c) |
| `models/euromillon-gpt2-tiny.gguf` | Modelo convertido a GGUF |

## Integración con proyecto existente

Para usar el modelo entrenado con tu proyecto `~/Euromillón/`:

```bash
# Copiar modelo GGUF al proyecto existente
cp ~/euromillon-llama2c/models/euromillon-gpt2-tiny.gguf ~/Euromillón/

# Actualizar predict_llama.py para usar tu modelo
sed -i 's|gemma4-v2-Q8_0.gguf|euromillon-gpt2-tiny.gguf|' ~/Euromillon/scripts/predict_llama.py
```

## ¿Por qué llama2.c?

1. **Sin dependencias de Python/PyTorch** - Solo gcc
2. **Entrenamiento en CPU** - Aprovecha tus 32 núcleos Ryzen
3. **Formato GGUF nativo** - Compatible con llama-cli
4. **Modelo ligero** - Entrenable en minutos, no horas
5. **Fácil de entender** - Código C simple y legible