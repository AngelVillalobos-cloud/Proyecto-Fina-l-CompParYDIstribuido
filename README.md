# Proyecto Final: Cómputo Paralelo y Distribuido (2ª Evaluación)
## Análisis de Tráfico Aéreo y Condiciones Ambientales

Sistema concurrente, distribuido y acelerado numéricamente para analizar la correlación entre sensores ambientales y la operación del tráfico aéreo.

---

## 🛠️ Arquitectura y Tecnologías
1. **Asyncio**: Descarga concurrente no bloqueante de APIs en la nube (**ThingSpeak** y esquema **AirLabs**).
2. **Pandas**: Limpieza, alineación temporal y cruce de datos ambientales con vuelos.
3. **MPI4Py**: Computación distribuida con patrón Maestro-Trabajador (`scatter` y `gather`).
4. **Numba (JIT)**: Aceleración de cálculos matemáticos para detección de anomalías estadísticas (Z-Scores) y coeficientes de correlación de Pearson.
5. **Matplotlib**: Generación del dashboard analítico de 4 cuadrantes.

---

## 📦 Instalación

En una terminal o entorno virtual de Python:

```bash
pip install pandas numpy matplotlib numba
```

*(Opcional para distribución con múltiples procesos en red/nodos):*
```bash
pip install mpi4py
```

---

## 🚀 Ejecución

### Modo Estándar / Monoproceso
```bash
python proyecto_final.py
```

### Modo Distribuido con MPI (ejemplo con 4 procesos)
```bash
mpiexec -n 4 python proyecto_final.py
```

---

## 📊 Archivos del Repositorio
* `proyecto_final.py`: Código principal del proyecto optimizado (~195 líneas).
* `guion_explicacion_proyecto.txt`: Guion de exposición paso a paso, respuestas a preguntas del profesor y explicación de cada bloque.
* `reporte_analisis.png`: Dashboard gráfico generado automáticamente al ejecutar el script.
