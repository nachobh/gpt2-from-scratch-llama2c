#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# EUROMILLÓN LAMA2C - Pipeline de dos fases
# ═══════════════════════════════════════════════════════════════
#
# FASE 1: Entrenar modelo BASE con datos reales de Euromillón
# FASE 2: FINETUNEAR con SWE-Zero (318,115 trayectorias)
#
# USOS:
#   ./run.sh setup              - Clonar y compilar llama2.c
#   ./run.sh data               - Extraer datos reales + generar dataset
#   ./run.sh phase1 [STEPS]     - Entrenar modelo BASE (fase 1)
#   ./run.sh phase2 [STEPS]     - Finetune con SWE-Zero (fase 2)
#   ./run.sh all [STEPS]        - Pipeline completo
#   ./run.sh generate [N]       - Generar N combinaciones
#
# ═══════════════════════════════════════════════════════════════

set -e

MODO="${1:-help}"
N_COMBINACIONES="${2:-5}"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║    EUROMILLÓN LAMA2C - Generador de Combinaciones con IA  ║"
echo "║           FASE 1 + FASE 2 (FINETUNEO)                     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

case "$MODO" in
    setup)
        bash 1-clone.sh
        ;;
    
    data)
        echo "→ Extraendo datos reales de Euromillón..."
        python3 2a-extract-real-data.py
        ;;
    
    phase1)
        echo "→ FASE 1: Entrenando modelo BASE..."
        bash 3a-phase1-train.sh
        ;;
    
    phase2)
        echo "→ FASE 2: Finetune con SWE-Zero..."
        bash 3b-phase2-finetune.sh
        ;;
    
    convert)
        bash 4-convert.sh
        ;;
    
    generate)
        echo "→ Generando $N_COMBINACIONES combinaciones..."
        python3 scripts/generate.py "$N_COMBINACIONES"
        ;;
    
    all)
        echo "→ EJECUTANDO PIPELINE COMPLETO (FASE 1 + FASE 2)..."
        echo ""
        bash 1-clone.sh
        echo ""
        bash 2a-extract-real-data.py
        echo ""
        bash 3a-phase1-train.sh
        echo ""
        bash 3b-phase2-finetune.sh
        echo ""
        bash 4-convert.sh
        echo ""
        echo "→ Generando combinaciones con modelo finetuneado..."
        python3 scripts/generate.py "$N_COMBINACIONES"
        ;;
    
    help|*)
        echo "Modos disponibles:"
        echo "  setup        - Clonar y compilar llama2.c"
        echo "  data         - Extraer datos reales + generar dataset"
        echo "  phase1 [N]   - Entrenar modelo BASE (fase 1)"
        echo "  phase2 [N]   - Finetune con SWE-Zero (fase 2)"
        echo "  convert      - Convertir modelo a GGUF"
        echo "  generate [N] - Generar N combinaciones"
        echo "  all [N]      - Pipeline completo"
        ;;
esac