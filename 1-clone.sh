#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# PASO 1: Clonar llama2.c
# ═══════════════════════════════════════════════════════════════

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  PASO 1: Clonar llama2.c                                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

REPO="https://github.com/karpathy/llama2.c.git"

if [ -d "llama2.c" ]; then
    echo "→ llm2.c ya existe, actualizando..."
    cd llama2.c
    git pull
else
    echo "→ Clonando $REPO..."
    git clone "$REPO" llama2.c
fi

# Compilar llama2.c
echo ""
echo "→ Compilando llama2.c..."
cd llama2.c
make clean
make

# Verificar que se compiló
if [ -f "bin/train" ] || [ -f "bin/tokenizer" ]; then
    echo "✓ Compilación exitosa"
    echo "  bin/train"
    echo "  bin/tokenizer"
    ls -la bin/
else
    # Algunos builds generan binarios diferentes
    ls -la bin/ 2>/dev/null || echo "  (binarios en otra ubicación)"
fi

echo ""
echo "✓ Paso 1 completado"
