# Roadmap: Niveles 2 y 3 del Modelo Euromillón

## Estado Actual (Nivel 1)

- **Modelo:** GPT-2 tiny (33.6M parámetros)
- **Datos:** 11,159 líneas (3,159 + 8,000 JSONL)
- **Salida:** `7, 14, 19, 28, 45 + 3, 8`
- **Herramientas:** llama.c/runq, llama.cpp/llama-cli, llama-server

---

## Nivel 2: Lenguaje Natural

### Objetivo
Que el modelo genere combinaciones con explicaciones en lenguaje natural.

### Dataset Necesario
```json
{"input": "Genera una combinacion con contexto", "output": "La combinacion del 2026-04-17 fue: 13, 20, 25, 46, 50 + 2, 11. Los numeros calientes fueron 13 y 46"}
```

### Pasos

1. **Preparar dataset descriptivo**
   - Añadir texto descriptivo a cada combinación
   - Incluir contexto de fecha, frecuencia, patrones
   - Objetivo: 10,000-50,000 líneas

2. **Entrenar nuevo modelo**
   ```bash
   python3 train.py data/euromillon-descriptivo.txt models/euromillon-l2-gpt2 \
     --n-layer 8 --n-head 8 --n-embd 512 --n-step 2000 \
     --batch-size 32 --seq-len 256 --lr 0.002 --warmup 100
   ```

3. **Probar con llama-server**
   ```bash
   bash run-llama-server.sh models/euromillon-l2-gpt2.gguf
   curl http://127.0.0.1:8084/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Genera una combinacion con contexto"}]}
   ```

---

## Nivel 3: Interpretación/Analítica

### Objetivo
Que el modelo analice patrones y proporcione estadísticas.

### Dataset Necesario
```json
{"input": "Analiza las estadisticas de Euromillon", "output": "Segun los datos, el numero 7 ha salido 45 veces, seguido de 19 con 42 veces. La frecuencia promedio es de 28.3 veces por numero."}
```

### Pasos

1. **Calcular estadísticas reales**
   - Frecuencia de cada número (1-50)
   - Frecuencia de cada estrella (1-12)
   - Patrones temporales (año, mes, semana)
   - Correlaciones entre números

2. **Generar dataset de análisis**
   - Preguntas frecuentes sobre Euromillón
   - Respuestas con datos reales
   - Análisis de tendencias

3. **Entrenar modelo analítico**
   - Mismo arquitectura que Nivel 2
   - Más pasos de entrenamiento (5000+)
   - Mayor secuencia (512 tokens)

---

## Herramientas Disponibles

| Herramienta | Uso | Puerto |
|-------------|-----|--------|
| `run-llama-c.sh` | Generación rápida | CLI |
| `run-llama-cli.sh` | Generación controlada | CLI |
| `run-llama-server.sh` | API REST | 8084 |

### Comandos de prueba

```bash
# API REST
curl http://127.0.0.1:8084/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Genera una combinacion"}]}
```

---

## Estimación de Tiempo

| Nivel | Datos | Entrenamiento | Resultado |
|-------|-------|---------------|-----------|
| 1 | 11,159 líneas | 500 pasos (~13.8h medido) | Combinaciones |
| 2 | 10,000+ líneas | 2,000 pasos (~55h a ritmo medido) | Combinaciones + explicaciones |
| 3 | 5,000+ líneas | 5,000 pasos (~140h a ritmo medido) | Análisis + estadísticas |

> Corrección: la estimación original de 1.5h para 500 pasos era optimista.
> Medido en `output/phase1-v2.log`: 500 pasos = 49,596s ≈ 13.8h (~99s/paso).
