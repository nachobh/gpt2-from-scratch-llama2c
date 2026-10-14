#!/usr/bin/env python3
import os, sys, struct, torch

PT = "models/euromillon-swe-zero.pt"
BIN = "models/euromillon-swe-zero.bin"
BASE = "models/euromillon-base-gpt2"

# Cargar modelo SWE-Zero
print("Cargando modelo SWE-Zero...")
ck = torch.load(PT, map_location='cpu', weights_only=False)
sd = ck.get('model_state', ck.get('model', {}))

# Leer binario original completo
print("Leyendo estructura original...")
with open(BASE, 'rb') as f:
    raw = f.read()

# Header (5 bytes): G2M2\x00
header = raw[:5]

# La estructura del archivo binario es:
# 1. Header (5 bytes): G2M2\x00
# 2. Config: cada valor son 2 bytes little-endian
#    - El primer valor (124) es la longitud del nombre del primer tensor
#    - Los siguientes valores son: n_layer, n_head, n_kv_heads, hidden_dim, max_seq_len, vocab_size, ...
#    - La config termina cuando encontramos un valor > 100 (nombre de tensor)
# 3. Tensores: cada tensor tiene:
#    - name_len (2 bytes): longitud del nombre
#    - name (name_len bytes): nombre del tensor
#    - dims (8 bytes): 4 valores de 2 bytes cada uno (d0, d1,d2, d3)
#    - data (d0*d1*d2*d3*4 bytes o d0*d1*4 bytes): datos del tensor

# Leer todos los valores de config (cada uno 2 bytes)
# Necesitamos saber cuántos valores hay antes del primer tensor
# Buscamos el primer nombre de tensor válido (ej: "tok_embeddings.weight")
pos = 5
values = []
while pos + 2 <= len(raw):
    val = struct.unpack('<H', raw[pos:pos+2])[0]
    values.append(val)
    pos += 2
    # Parar cuando encontramos un valor que parece un nombre de tensor
    # Los nombres de tensor tienen valores como 124, 45, etc.
    # Los valores de config suelen ser potencias de 2 o 0
    if val > 100 and len(values) > 5:
        break

print(f"  Header: {header}")
print(f"  Valores de config: {values[:15]}")
print(f"  Posición después de config: {pos}")

# Leer todos los tensores del original
print("Leyendo tensores originales...")
tensors = []
while pos + 2 <= len(raw):
    name_len = struct.unpack('<H', raw[pos:pos+2])[0]
    pos += 2
    if name_len > 2000 or name_len == 0:
        break
    if name_len < 500:
        name = raw[pos:pos+name_len].decode('utf-8', errors='replace').rstrip('\x00')
        pos += name_len
    else:
        break
    d0 = struct.unpack('<H', raw[pos:pos+2])[0]; pos += 2
    d1 = struct.unpack('<H', raw[pos:pos+2])[0]; pos += 2
    d2 = struct.unpack('<H', raw[pos:pos+2])[0]; pos += 2
    d3 = struct.unpack('<H', raw[pos:pos+2])[0]; pos += 2
    if d0 and d1 and d2 and d3:
        dlen = d0*d1*d2*d3*4
    elif d0 and d1:
        dlen = d0*d1*4
    else:
        dlen = 4
    tdata = raw[pos:pos+dlen]; pos += dlen
    if name:
        print(f"  {name} dims=({d0},{d1},{d2},{d3})")
        tensors.append((name, (d0,d1,d2,d3), tdata))

print(f"  Total tensores: {len(tensors)}")

# Escribir nuevo binario
print(f"Escribiendo {BIN}...")
with open(BIN, 'wb') as f:
    f.write(header)
    for v in values:
        f.write(struct.pack('<H', v))
    for name, dims, tdata in tensors:
        n_bytes = name.encode('utf-8')
        f.write(struct.pack('<H', len(n_bytes)))
        f.write(n_bytes)
        for d in dims:
            f.write(struct.pack('<H', d))
        if name in sd:
            tf = sd[name].float()
            f.write(tf.cpu().numpy().tobytes('C'))
        else:
            f.write(tdata)

print(f"  Tamaño: {os.path.getsize(BIN) / (1024*1024):.1f} MB")