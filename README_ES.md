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
./run.sh data
```
Ejecuta `2a-extract-real-data.py`: extrae datos reales de Euromillón y, si la
fuente remota falla, genera un fallback de 2,000 combinaciones ponderadas por
frecuencia histórica. **Nota:** `run.sh data` ignora cualquier argumento
(numérico o no); el generador de 50,000 frases (`2-generate-data.py`) existe
pero **no** es invocado por `run.sh` — se ejecuta aparte con
`python3 2-generate-data.py`.

### Paso 3: Entrenar modelo GPT-2
```bash
./run.sh phase1        # FASE 1: modelo base (3a-phase1-train.sh -> train.py)
./run.sh phase2        # FASE 2: finetune con SWE-Zero (3b-phase2-finetune.sh)
```
No existe un modo `train` ni presets `tiny/medium/large`. La config entrenada
y medida (ver `output/phase1-v2.log`) es: 8 layers, 8 heads, 512 embedding,
vocabulario de 45 caracteres = **33,586,176 parámetros** (~33.6M, no 117M).
Tiempo real medido: 500 pasos ≈ 13.8 horas (batch 32, seq 128, GPU).

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
./run.sh all           # setup + data + phase1 + phase2 + convert + generate
```

## Archivos Generados (nombres reales)

| Archivo | Descripción |
|---------|-------------|
| `data/euromillon_real.txt` | Dataset extraído de datos reales (fase 1) |
| `data/euromillon_merged.txt` | Dataset fusionado (real + 8,000 JSONL) |
| `models/euromillon-base-gpt2.pt` (+ `.meta.json`) | Modelo fase 1 (train.py guarda `.pt`, no `.bin`) |
| `models/euromillon-v2-gpt2.pt` / `.safetensors` | Checkpoints entrenados |
| `models/euromillon-v2-gpt2.gguf` | Modelo convertido a GGUF (~128 MB) |

No se generan `euromillon-gpt2-tiny.bin/.gguf`. **Conocido:** `scripts/generate.py`
buscaba ese nombre obsoleto; su default se actualizó a `euromillon-v2-gpt2.gguf`.

## GGUF gotchas (por qué hay tantos fix-*.py)

`build-gguf.py` escribía la longitud de cada clave KV como **uint64** (8 bytes),
pero `llama-cli` (formato GGUF v1) espera **uint32** (4 bytes); `build-gguf-v2.py`
lo corrigió escribiendo `key_len` como uint32. Ese desfase de 4 bytes por par KV
desalineaba todo el header, y los `fix-*.py` / `add-kv-padding.py` parchean el
hueco de padding resultante en ficheros ya escritos. Lección de portfolio: el
formato GGUF exige exactitud byte a byte — n_tensors/n_kv como uint64 pero
key_len/val_len como uint32, y cualquier desalineación produce un modelo que
`llama-cli` carga con pesos corruptos o rechaza.

## Integración con proyecto existente

Para usar el modelo entrenado con tu proyecto `~/Euromillón/`:

```bash
# Copiar modelo GGUF al proyecto existente
cp ~/euromillon-llama2c/models/euromillon-v2-gpt2.gguf ~/Euromillón/

# Actualizar predict_llama.py para usar tu modelo
sed -i 's|gemma4-v2-Q8_0.gguf|euromillon-v2-gpt2.gguf|' ~/Euromillon/scripts/predict_llama.py
```

## ¿Por qué llama2.c?

1. **Sin dependencias de Python/PyTorch** - Solo gcc
2. **Entrenamiento en CPU** - Aprovecha tus 32 núcleos Ryzen
3. **Formato GGUF nativo** - Compatible con llama-cli
4. **Modelo ligero** - Entrenable en minutos, no horas
5. **Fácil de entender** - Código C simple y legible