#!/usr/bin/env python3
"""
PASO 2a: Extraer datos reales de Euromillón del CSV de resultados.

Si file.csv no existe, genera datos simulados basados en estadísticas reales.

Formato esperado del CSV:
- Columnas: Fecha, N1, N2, N3, N4, N5, E1, E2
- O similar con números y estrellas
"""

import csv
import random
import os
import json

# Frecuencias históricas reales de Euromillón
NUMEROS_FRECUENTES = [7, 19, 23, 32, 42, 17, 38, 14, 27, 45, 5, 11, 28, 34, 49]
ESTRELLAS_FRECUENTES = [2, 3, 8, 11, 5]

# Plantillas de texto para dar contexto
TEMPLATES = [
    "Sorteo de Euromillón del {fecha}: {numeros} con estrellas {estrellas}",
    "Resultados del {fecha}: {numeros} + {estrellas}",
    "Combinación ganadora {fecha}: {numeros}, estrellas {estrellas}",
    "Sorteo {fecha}: {numeros} | Estrellas: {estrellas}",
    "Euromillón {fecha}: {numeros}, {estrellas}",
    "Números de hoy {fecha}: {numeros}, estrellas {estrellas}",
    "Resultado del {fecha}: {numeros} y estrellas {estrellas}",
]

def generar_texto_combinacion(num_str, star_str, fecha="2026"):
    """Genera texto descriptivo con la combinación."""
    template = random.choice(TEMPLATES)
    return template.format(
        fecha=fecha,
        numeros=num_str,
        estrellas=star_str
    )

def parse_csv_file(filepath):
    """
    Intenta parsear el CSV de resultados.
    Asume columnas con números y estrellas.
    """
    resultados = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Leer header
            
            for row in reader:
                if len(row) < 7:
                    continue
                
                # Intentar encontrar columnas numéricas
                try:
                    # Buscar 5 números (1-50) y 2 estrellas (1-12)
                    nums = []
                    stars = []
                    for cell in row:
                        try:
                            val = int(cell.strip())
                            if 1 <= val <= 50:
                                if len(nums) < 5:
                                    nums.append(val)
                                else:
                                    if len(stars) < 2:
                                        stars.append(val)
                        except ValueError:
                            continue
                    
                    if len(nums) == 5 and len(stars) == 2:
                        fecha = "2026"
                        if header:
                            try:
                                # Intentar encontrar columna de fecha
                                for i, h in enumerate(header):
                                    if 'fecha' in h.lower() or 'date' in h.lower():
                                        fecha = row[i].strip()
                                        break
                            except (IndexError, ValueError):
                                pass
                        
                        results.append({
                            'fecha': fecha,
                            'numeros': sorted(nums),
                            'estrellas': sorted(stars)
                        })
                except Exception:
                    continue
    except FileNotFoundError:
        print(f"  ⚠️ Archivo no encontrado: {filepath}")
        return None
    except Exception as e:
        print(f"  ⚠️ Error al leer CSV: {e}")
        return None
    
    return resultados

def generate_fallback_data(n_samples=2000):
    """Genera datos de respaldo si no hay CSV."""
    print(f"  → Generando {n_samples} combinaciones simuladas...")
    
    resultados = []
    for _ in range(n_samples):
        # Muestreo con pesos (números frecuentes más probables)
        pesos_n = [2.0 if n in NUMEROS_FRECUENTES else 1.0 for n in range(1, 51)]
        pesos_e = [2.5 if e in ESTRELLAS_FRECUENTES else 1.0 for e in range(1, 13)]
        
        numeros = []
        disp_n = list(range(1, 51))
        w_n = pesos_n[:]
        for _ in range(5):
            n = random.choices(disp_n, weights=w_n, k=1)[0]
            numeros.append(n)
            idx = disp_n.index(n)
            disp_n.pop(idx)
            w_n.pop(idx)
        
        estrellas = []
        disp_e = list(range(1, 13))
        w_e = pesos_e[:]
        for _ in range(2):
            e = random.choices(disp_e, weights=w_e, k=1)[0]
            estrellas.append(e)
            idx = disp_e.index(e)
            disp_e.pop(idx)
            w_e.pop(idx)
        
        resultados.append({
            'fecha': f"2026-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            'numeros': sorted(numeros),
            'estrellas': sorted(estrellas)
        })
    
    return resultados

def main():
    csv_path = os.path.expanduser("~/Euromillón/file.csv")
    
    print("=" * 60)
    print("EXTRACCIÓN DE DATOS DE EUROMILLÓN")
    print("=" * 60)
    print(f"Archivo CSV: {csv_path}")
    print()
    
    resultados = None
    
    # Intentar leer CSV
    if os.path.exists(csv_path):
        print("→ Leyendo CSV de resultados reales...")
        resultados = parse_csv_file(csv_path)
    
    # Si no se pudo leer, generar datos simulados
    if resultados is None or len(resultados) == 0:
        print(f"  → No se encontraron datos válidos en {csv_path}")
        print("  → Usando datos simulados con estadísticas reales...")
        resultados = generate_fallback_data(2000)
    
    print(f"\n  → Total de combinaciones extraídas: {len(resultados)}")
    
    # Generar texto de entrenamiento
    print("\n→ Generando texto de entrenamiento...")
    texto = ""
    
    for r in resultados:
        nums_str = ", ".join(str(n) for n in r['numeros'])
        stars_str = ", ".join(str(s) for s in r['estrellas'])
        
        # Generar 1-2 frases por combinación
        for _ in range(random.randint(1, 2)):
            texto += generar_texto_combinacion(
                nums_str, stars_str, r['fecha']
            ) + "\n"
        
        # Agregar contexto estadístico ocasional
        if random.random() < 0.1:
            texto += f"\n\nLos números más frecuentes son: "
            texto += ", ".join(str(n) for n in NUMEROS_FRECUENTES[:8])
            texto += ".\n"
    
    # Guardar
    os.makedirs("data", exist_ok=True)
    output_path = "data/euromillon_real.txt"
    with open(output_path, "w") as f:
        f.write(texto)
    
    print(f"\n✓ Dataset guardado: {output_path}")
    print(f"  Tamaño: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    print(f"  Líneas: {texto.count(chr(10))}")
    
    # Guardar metadatos
    metadata = {
        "source": csv_path if os.path.exists(csv_path) else "simulated",
        "combinations": len(resultados),
        "lines": texto.count(chr(10)),
        "size_mb": round(os.path.getsize(output_path) / (1024*1024), 2)
    }
    with open("data/euromillon_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Metadatos guardados en data/euromillon_meta.json")

if __name__ == "__main__":
    main()