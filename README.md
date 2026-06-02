# Análisis Multimodal de Células Sanguíneas

Repositorio asociado al Trabajo Fin de Máster centrado en la clasificación celular y la generación automática de descripciones morfológicas mediante modelos de visión y lenguaje.

El proyecto integra técnicas de visión por computador, modelos visión-lenguaje y evaluación semántica automática para el análisis morfológico de células sanguíneas.

---

## Objetivos

- Clasificar células sanguíneas a partir de imágenes microscópicas.
- Generar descripciones morfológicas automáticas.
- Adaptar modelos de captioning al dominio hematológico.
- Evaluar la calidad descriptiva mediante auditoría semántica automática.
- Analizar el rendimiento global y por atributos morfológicos.

---

## Arquitectura general

```text
Imágenes microscópicas
        │
        ▼
Clasificación celular (ViT)
        │
        ▼
Generación y refinamiento de descripciones
(BLIP, BLIP + LoRA, Gemma 3 4B, Qwen3-VL)
        │
        ▼
Evaluación semántica y análisis de concordancia
(Gemini-3-flash-preview, Qwen2.5-VL)
        │
        ▼
Análisis final
```

---

## Modelos utilizados

### Clasificación celular

- Vision Transformer (ViT) fine-tuned para clasificación de subtipos celulares

### Generación y refinamiento de descripciones

- BLIP
- BLIP + LoRA
- Gemma 3 4B VLM condicionado
- Qwen3-VL fine-tuned

### Evaluación y validación

- Gemini-3-flash-preview
- Qwen2.5-VL

---

## Resultados principales

### Clasificación celular

| Métrica | Valor |
|----------|----------:|
| Accuracy | 98.50 % |
| F1 weighted | 0.985 |

### Comparativa global

| Modelo | Correcto ampliado (%) | Utilidad alta (%) | Mejora (%) | Accuracy atributos (%) |
|---|---:|---:|---:|---:|
| BLIP+LoRA libre base | 35.71 | 31.11 | 31.17 | 69.34 |
| BLIP+LoRA híbrido | 83.80 | 80.95 | 47.71 | 86.11 |
| Gemma 3 4B VLM condicionado | 94.04 | 69.28 | 70.32 | 86.01 |
| Qwen3-VL fine-tuned | 99.74 | 74.72 | 99.74 | 48.86 |


Aunque Qwen3-VL obtuvo una elevada tasa de generación de descripciones consideradas correctas a nivel global, mostró limitaciones en la identificación explícita de atributos morfológicos específicos, lo que redujo significativamente su accuracy por atributo.

### Rendimiento por atributo morfológico

| Atributo | BLIP+LoRA libre base | BLIP+LoRA híbrido | Gemma 3 4B VLM | Qwen3-VL |
|-----------|----------:|----------:|----------:|----------:|
| Clase celular | 99.35 | 97.34 | 99.48 | 99.61 |
| Tamaño celular | 94.82 | 96.04 | 92.05 | 6.04 |
| Forma nuclear | 90.25 | 91.39 | 92.49 | 57.19 |
| Cromatina | 92.65 | 97.56 | 92.99 | 50.62 |
| Citoplasma | 84.38 | 82.65 | 87.13 | 47.27 |
| Granulación | 43.94 | 51.66 | 87.34 | 32.40 |


Los resultados muestran que el rendimiento no fue homogéneo entre atributos morfológicos, observándose mayores dificultades en la identificación de la granulación y, en menor medida, del citoplasma.

---

## Resultados adicionales

El repositorio incluye figuras y análisis complementarios desarrollados durante el trabajo:

- Matriz de confusión multicategoría para clasificación celular.
- Heatmap de accuracy por atributo morfológico.
- Diagramas de concordancia entre evaluadores automáticos.
- Comparativas globales entre modelos multimodales.

---

## Estructura del repositorio

```text
blood-cell-vllm/
│
├── codigo/
│   └── codigo_pulido.ipynb
│
├── modelos/
│   ├── vit_subclases_final/
│   └── blip_lora_hibrido_final/
│
├── figuras/
│   ├── matriz_confusion_vit.png
│   ├── heatmap_accuracy_atributos.png
│   ├── diagrama_aluvial_concordancia.png
│   └── pipeline_multimodal_hematologia.png
│
└── README.md
```

---

## Datos

El dataset utilizado en este trabajo procede de un entorno hospitalario y está sujeto a restricciones de acceso. Por este motivo, las imágenes originales y los datasets empleados durante los experimentos no forman parte de los materiales incluidos en este repositorio.

---

## Autor

**Diana Gutiérrez Martínez**

Trabajo Fin de Máster

Máster en Bioinformática y Bioestadística

Universitat Oberta de Catalunya (UOC)

2025

---

## Director

**Edwin Santiago Alférez Baquero**
