#!/usr/bin/env python3
"""Corrige el header del archivo GGUF del modelo SWE-Zero."""
import os
import shutil

# Leer el header correcto del archivo GGUF del modelo base
base_gguf = "~/euromillon-llama2c/models/euromillon-base-gpt2.gguf"
swe_gguf = "~/euromillon-llama2c/models/euromillon-swe-zero.gguf"

print(f"Modelo base GGUF: {base_gguf}")
print(f"  Tamaño: {os.path.getsize(base_gguf) / (1024*1024):.1f} MB")

print(f"Modelo SWE-Zero GGUF: {swe_gguf}")
print(f"  Tamaño: {os.path.getsize(swe_gguf) / (1024*1024):.1f} MB")

# Leer los primeros 4 bytes del modelo base (header GGUF)
with open(base_gguf, 'rb') as f:
    base_header = f.read(4)
    print(f"  Header base: {base_header.hex()} ({base_header})")

# Leer los primeros 4 bytes del modelo SWE-Zero
with open(swe_gguf, 'rb') as f:
    swe_header = f.read(4)
    print(f"  Header SWE-Zero: {swe_header.hex()} ({swe_header})")

# Copiar el header del modelo base al modelo SWE-Zero
print("\nCorrigiendo header del modelo SWE-Zero...")
with open(swe_gguf, 'r+b') as f:
    f.seek(0)
    f.write(base_header)

# Verificar la corrección
with open(swe_gguf, 'rb') as f:
    new_header = f.read(4)
    print(f"  Nuevo header SWE-Zero: {new_header.hex()} ({new_header})")
    print(f"  Header correcto: {new_header == base_header}")

# Copiar el archivo completo del modelo base como referencia
print("\nCopiando archivo GGUF del modelo base como referencia...")
shutil.copy2(base_gguf, swe_gguf + ".backup")
print(f"  Backup creado: {swe_gguf}.backup")