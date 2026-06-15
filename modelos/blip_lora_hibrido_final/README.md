# BLIP + LoRA Híbrido para Descripción Morfológica

Modelo desarrollado a partir de un experimento basado en **BLIP adaptado mediante LoRA**, diseñado para la generación automática de descripciones morfológicas de células sanguíneas.

---

## Descripción

Modelo multimodal basado en BLIP adaptado mediante LoRA para la generación automática de descripciones morfológicas de células sanguíneas a partir de imágenes microscópicas.

La aproximación híbrida incorpora información visual y contexto celular para producir descripciones más precisas y consistentes desde el punto de vista morfológico.

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
Integración de contexto adicional
        │
        ▼
Descripción morfológica final
```

---

## Arquitectura

- Modelo base: BLIP
- Adaptación: LoRA
- Tipo de experimento: Generación de descripciones morfológicas
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

El modelo mostró una mejora significativa respecto a enfoques de generación más simples, alcanzando elevados niveles de concordancia en la identificación de atributos morfológicos y una mayor utilidad descriptiva global.

---

## Reproducibilidad

El repositorio incluye la arquitectura general, el flujo experimental y los componentes principales utilizados durante el desarrollo.

Determinados recursos, modelos entrenados y conjuntos de datos no se distribuyen públicamente debido a restricciones de acceso y confidencialidad..

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
