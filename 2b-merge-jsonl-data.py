#!/usr/bin/env python3
"""
PASO 2b: Fusionar datos JSONL (8000 samples) con el dataset existente.

Convierte los 8000 samples de euromillon-first-approach al formato
de texto plano y los mezcla con los 3537 líneas existentes.

NOTA sobre la cifra: data/euromillon_real.txt tiene 3,537 líneas FÍSICAS,
pero solo 3,159 no vacías; este script filtra las vacías, así que el merge
real aportó 3,159 muestras (3,159 + 8,000 = 11,159, la cifra que citan
compare-models.sh / run-comparison.sh / LEVELS_2_3_ROADMAP.md).
"""

import json
import random

def main():
    jsonl_path = "/home/ignacia/euromillon-first-approach/data/train.jsonl"
    existing_path = "/home/ignacia/euromillon-llama2c/data/euromillon_real.txt"
    merged_path = "/home/ignacia/euromillon-llama2c/data/euromillon_merged.txt"
    
    # Leer JSONL
    print("→ Leyendo JSONL (8000 samples)...")
    jsonl_lines = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # Extraer el output limpio
                    output = data.get("output", "").strip()
                    if output:
                        jsonl_lines.append(output)
                except json.JSONDecodeError:
                    continue
    print(f"  Samples JSONL válidos: {len(jsonl_lines)}")
    
    # Leer dataset existente
    print("→ Leyendo dataset existente...")
    with open(existing_path, 'r', encoding='utf-8') as f:
        existing_text = f.read()
    existing_lines = [l for l in existing_text.split('\n') if l.strip()]
    print(f"  Líneas existentes: {len(existing_lines)}")
    
    # Mezclar todo
    all_lines = existing_lines + jsonl_lines
    random.shuffle(all_lines)
    
    # Escribir
    merged_text = '\n'.join(all_lines)
    with open(merged_path, 'w', encoding='utf-8') as f:
        f.write(merged_text)
    
    print(f"\n✓ Dataset fusionado: {merged_path}")
    print(f"  Total líneas: {len(all_lines)}")
    print(f"  Total caracteres: {len(merged_text)}")
    print(f"  Tamaño: {len(merged_text) / (1024*1024):.1f} MB")
    print(f"  Breakdown: {len(existing_lines)} existentes + {len(jsonl_lines)} JSONL = {len(all_lines)}")

if __name__ == "__main__":
    main()
