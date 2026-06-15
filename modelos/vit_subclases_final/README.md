# Vision Transformer (ViT) para Clasificación Celular

Repositorio asociado al modelo de clasificación celular utilizado en el Trabajo Fin de Máster sobre análisis multimodal de células sanguíneas mediante modelos de visión y lenguaje.

---

## Descripción

Este modelo está basado en la arquitectura **Vision Transformer (ViT)** y fue entrenado para la clasificación automática de subtipos celulares a partir de imágenes microscópicas de sangre periférica.

Dentro del pipeline multimodal desarrollado, el modelo constituye la primera etapa del sistema, proporcionando la predicción de la clase celular que posteriormente es utilizada para condicionar la generación de descripciones morfológicas mediante modelos visión-lenguaje.

---

## Función dentro del pipeline

```text
Imagen microscópica
        │
        ▼
Clasificación celular (ViT)
        │
        ▼
Predicción de la clase celular
        │
        ▼
Generación de descripción morfológica
(BLIP+LoRA, Gemma 3 4B VLM, Qwen3-VL)
```

---

## Arquitectura

- Modelo base: Vision Transformer (ViT)
- Tarea: Clasificación multiclase
- Dominio: Hematología digital
- Entrada: Imágenes microscópicas de células sanguíneas
- Salida: Subtipo celular predicho

---

## Rendimiento

| Métrica | Valor |
|----------|----------:|
| Accuracy | 98.50 % |
| F1-score ponderado | 0.985 |

---

## Resultados

El modelo obtuvo un rendimiento elevado en la clasificación de subtipos celulares, permitiendo su utilización como componente de entrada para los experimentos de generación automática de descripciones morfológicas desarrollados durante el trabajo.

---

## Disponibilidad

Los datos utilizados durante el entrenamiento no se distribuyen públicamente debido a restricciones de acceso y confidencialidad.

---

## Autor

**Diana Gutiérrez Martínez**

Trabajo Fin de Máster

Máster en Bioinformática y Bioestadística

Universitat Oberta de Catalunya (UOC)

## Director

**Edwin Santiago Alférez Baquero**

18 Junio 2026
