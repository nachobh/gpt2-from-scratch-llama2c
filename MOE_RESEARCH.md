# MoE (Mixture of Experts) — Investigación y Plan

## ¿Qué es un Modelo MoE?

Un modelo **Mixture of Experts** (MoE) es una arquitectura donde cada capa contiene múltiples redes neuronales independientes ("expertos"), y un **router/gate** decide qué expertos activar para cada token.

### Ventajas clave:
- **Más capacidad** con menos computación por token
- **Especialización**: cada experto puede aprender patrones distintos
- **Escalabilidad**: se añaden expertos sin aumentar la latencia

---

## Arquitecturas MoE Principales

### 1. Mixtral 8x7B (Mistral AI)
- **Total:** 46.7B parámetros
- **Activo por token:** 12.9B parámetros
- **Estructura:** 8 expertos por capa, **2 activos por token**
- **Cada experto:** ~7B parámetros
- **Ventaja:** 3.5x más capacidad que un modelo denso de 12.9B

### 2. Gemma 3 27B MoE
- **Total:** 27B parámetros (variante MoE)
- **Contexto:** 128K tokens
- **Multimodal:** texto + imagen
- **Optimizado para dispositivos locales**

### 3. Gemma 4 MoE
- **Variantes:** Dense y MoE
- **Escalable para on-device**
- **Llama.cpp compatible** (soporte nativo)

---

## ¿Cómo funciona un experto en cada capa?

```
[Input] → [Router/Gate] → [Selecciona Top-K expertos] → [Suma resultados] → [Output]
```

### Componentes clave:

1. **Expertos (FFN)**: Cada experto es una capa Feed-Forward independiente
2. **Router/Gate**: Decide cuáles K expertos activar (normalmente K=1 ó 2)
3. **Load balancing**: Distribuye tokens entre expertos equitativamente
4. **Noisy gating**: Añade ruido aleatorio para evitar que un experto domine

---

## Plan para crear un MoE desde cero

### Opción A: GPT-2 tiny MoE (para Euromillón)

```python
# Arquitectura propuesta:
# - 8 capas de transformer
# - Cada capa con 4 expertos (en lugar de 1 FFN)
# - Router selecciona 1 experto por token
# - 8 expertos × 4.2M parámetros = 33.6M total
# - Pero solo 4.2M activos por token (como GPT-2 tiny normal)
```

**Ventaja:** Mismo tamaño que GPT-2 tiny normal, pero con especialización
**Entrenamiento:** 500-2000 pasos (como el actual)
**Tamaño GGUF:** ~134 MB (igual que el actual)

### Opción B: Modelo mediano MoE

```python
# Basado en Mixtral pero más pequeño:
# - 16-32 capas de transformer
# - Cada capa con 4-8 expertos
# - Router selecciona 2 expertos por token
# - Cada experto: 1-2M parámetros
# - Total: 20-50M parámetros activos
```

**Ventaja:** Más capacidad, especialización por dominio
**Entrenamiento:** 5000-20000 pasos
**Tamaño GGUF:** 1-4 GB

---

## Datasets para cada tipo de experto

### Estrategia de entrenamiento multi-dominio:

1. **Experto 1-2:** Combinaciones de Euromillón
2. **Experto 3-4:** Código/programación
3. **Experto 5-6:** Análisis estadístico
4. **Experto 7-8:** Lenguaje natural/conversación

### Dataset por experto:
- Cada experto se entrena con datos de su dominio específico
- El router aprende a enviar cada token al experto correcto
- Se usa **loss balancing** para que todos los expertos mejoren

---

## Implementación en PyTorch (esqueleto)

```python
class MoEFeedForward(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Cada experto es una FFN independiente
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_experts)
        ])
        
        # Router: decide qué expertos activar
        self.gate = nn.Linear(dim, num_experts)
    
    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        gate_logits = self.gate(x)  # (batch, seq_len, num_experts)
        
        # Seleccionar Top-K expertos por token
        top_k_weights, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # Sumar resultados de cada experto
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]
            weight = top_k_weights[:, :, i]
            
            # Aplicar experto seleccionado
            expert_output = self.experts[expert_idx](x)
            output = output + weight.unsqueeze(-1) * expert_output
        
        return output
```

---

## Comparativa: MoE vs Modelo Denso

| Característica | MoE | Denso (GPT-2 tiny) |
|----------------|-----|-------------------|
| Parámetros totales | 33.6M (4×8 expertos) | 33.6M (1 FFN) |
| Activos por token | 4.2M | 33.6M |
| Latencia inferencia | ~igual | ~igual |
| Capacidad total | 33.6M | 33.6M |
| Especialización | Sí (por experto) | No |
| Entrenamiento | Más complejo (routing) | Más simple |
| Tamaño GGUF | ~134 MB | ~134 MB |

**Nota:** En inferencia, el MoE es **más rápido** porque solo activa 1/8 de los parámetros por token.

---

## Herramientas disponibles

### Infraestructura actual:
- ✅ PyTorch 2.12.1+rocm7.2 (ROCm/AMD)
- ✅ llama.cpp con soporte MoE (`--cpu-moe`, `--n-cpu-moe`)
- ✅ huggingface_hub para descargar modelos
- ✅ GPU AMD Radeon 8060S (103GB VRAM unificada)

### Modelos MoE descargables:
- `mistralai/Mixtral-8x7B-Instruct-v0.1` (47B, GGUF)
- `google/gemma-3-27b-it` (27B, GGUF)
- `google/gemma-4-moe` (variante MoE)

---

## Próximos pasos

1. **Crear esqueleto MoE** en `train_moe.py`
2. **Entrenar con datos de Euromillón** (primera fase)
3. **Añadir datasets de código** (segunda fase)
4. **Añadir datasets de lenguaje natural** (tercera fase)
5. **Evaluar con workbench** (como con Gemma4)
6. **Convertir a GGUF** con soporte MoE

---

## Estimación de recursos

| Componente | VRAM | CPU | Tiempo |
|------------|------|-----|--------|
| MoE pequeño (8 expertos × 4.2M) | ~256 MB | 8 threads | 2-4h (500 pasos) |
| MoE mediano (8 expertos × 20M) | ~1 GB | 16 threads | 8-16h (2000 pasos) |
| MoE grande (8 expertos × 100M) | ~4 GB | 16 threads | 16-32h (5000 pasos) |
