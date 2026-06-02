# BLIP + LoRA Híbrido para Descripción Morfológica

Modelo desarrollado a partir de un experimento basado en **BLIP adaptado mediante LoRA**, diseñado para la generación automática de descripciones morfológicas de células sanguíneas.

---

## Descripción

Este modelo corresponde a la variante híbrida desarrollada durante el Trabajo Fin de Máster. A diferencia de la generación libre, el enfoque híbrido incorpora información morfológica y de clasificación celular para guiar la generación de descripciones más precisas y útiles desde el punto de vista hematológico.

El objetivo fue mejorar la calidad descriptiva y la concordancia con los atributos morfológicos presentes en las imágenes microscópicas.

---

## Función dentro del pipeline

```text
Imagen microscópica
        │
        ▼
Clasificación celular (ViT)
        │
        ▼
Generación con BLIP + LoRA
        │
        ▼
Condicionamiento híbrido
        │
        ▼
Descripción morfológica final
```

---

## Arquitectura

- Modelo base: BLIP
- Adaptación: LoRA
- Tipo de experimento: Generación híbrida condicionada
- Dominio: Hematología digital
- Entrada: Imagen microscópica y contexto morfológico
- Salida: Descripción morfológica automática

---

## Rendimiento

| Métrica | Valor |
|----------|----------:|
| Correcto ampliado | 83.80 % |
| Utilidad alta | 80.95 % |
| Mejora respecto al input | 80.43 % |
| Accuracy por atributos | 86.11 % |

---

## Resultados

El modelo mostró una mejora sustancial respecto a las variantes de generación libre, alcanzando elevados niveles de concordancia en la identificación de atributos morfológicos y una mayor utilidad descriptiva global.

---

## Disponibilidad

Los pesos entrenados, imágenes originales y conjuntos de datos utilizados durante el entrenamiento no se distribuyen públicamente debido a restricciones de acceso asociadas al origen hospitalario de los datos.

---

## Autor

**Diana Gutiérrez Martínez**

Trabajo Fin de Máster

Máster en Bioinformática y Bioestadística

Universitat Oberta de Catalunya (UOC)

18 de junio de 2026

---

## Director

**Edwin Santiago Alférez Baquero**
