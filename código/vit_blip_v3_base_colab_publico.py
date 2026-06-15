# %% [markdown]
# # ViT + BLIP híbrido — generación de V3 base
#
# Este notebook contiene la fase ViT + BLIP del flujo experimental. Su objetivo es generar la V3 base,
# que posteriormente se utiliza como archivo de entrada para los modelos BLIP + LoRA, Gemma 3 y Qwen-VL.
#
# La versión presentada mantiene una descripción metodológica general del proceso, evitando detallar reglas
# internas o listas operativas más allá de lo necesario para ejecutar el flujo.
#
# ## Evolución de versiones hasta V3 base
#
# **V1 — Generación directa.** Se generaban descripciones de forma directa con un modelo base. El resultado era demasiado general.
#
# **V2 — Normalización textual.** Se añadieron reglas de limpieza y normalización para obtener salidas más estables.
#
# **V3 base — Integración visual y textual.** Se integró un modelo visual de apoyo y un generador de descripciones.

# %%
# -----------------------------------------------------------
# 1. Instalación e imports
# -----------------------------------------------------------
!pip install -q transformers accelerate datasets evaluate torchvision pillow scikit-learn openpyxl pyyaml

import os
import re
import time
import json
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import evaluate
import yaml

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.model_selection import train_test_split
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    BlipProcessor,
    BlipForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# %%
# -----------------------------------------------------------
# 2. Directorios del proyecto
# -----------------------------------------------------------
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    print("No se está ejecutando en Colab o Drive no está disponible.")

if Path("/content/drive/MyDrive/TFM/blood-cell-vllm").exists():
    BASE_DIR = Path("/content/drive/MyDrive/TFM/blood-cell-vllm")
else:
    BASE_DIR = Path("..").resolve()

ruta_zip = BASE_DIR / "imagenes.zip"
imagen_celulas_inicial = BASE_DIR / "imagenes"

RUTA_ANALISIS = BASE_DIR / "datos" / "analisis"
RUTA_ANALISIS.mkdir(parents=True, exist_ok=True)

RUTA_CSV_RECONSTRUIDO = RUTA_ANALISIS / "dataset_final_reconstruido.csv"
RUTA_TRAIN_CSV = RUTA_ANALISIS / "train_reconstruido.csv"
RUTA_VAL_CSV   = RUTA_ANALISIS / "val_reconstruido.csv"
RUTA_TEST_CSV  = RUTA_ANALISIS / "test_reconstruido.csv"

print("Directorio base:", BASE_DIR)
print("Ruta imágenes:", imagen_celulas_inicial)
print("Ruta análisis:", RUTA_ANALISIS)

# %% [markdown]
# ## Nota de versión
# En esta versión se mantiene la lógica necesaria para reproducir el flujo, pero la explicación textual se deja en un nivel general.
# Las categorías y reglas internas se tratan como elementos operativos del pipeline.

# %%
# -----------------------------------------------------------
# 3. Preparación del dataset y configuración privada
# -----------------------------------------------------------
if not imagen_celulas_inicial.exists() and ruta_zip.exists():
    with zipfile.ZipFile(ruta_zip, "r") as zip_ref:
        zip_ref.extractall(BASE_DIR)
    print("Dataset descomprimido.")
else:
    print("No hace falta descomprimir.")

checkpoint_dir = imagen_celulas_inicial / ".ipynb_checkpoints"
if checkpoint_dir.exists():
    shutil.rmtree(checkpoint_dir)

# -----------------------------------------------------------
# Configuración específica del dominio
# -----------------------------------------------------------
# En la versión pública se omiten:
# - categorías operativas concretas
# - reglas de reconstrucción desde nombre de archivo
# - mapas de normalización
# - vocabulario y reglas específicas de extracción morfológica
#
# Para ejecutar el notebook completo, estas funciones deben
# estar disponibles en un módulo privado no incluido en GitHub.

try:
    from config_privada import (
        cargar_categorias_operativas,
        cargar_mapa_clases,
        cargar_normalizacion_etiquetas,
        obtener_categoria_desde_nombre,
        cargar_pistas_visuales,
        extraer_cues_visuales,
    )
except ImportError as exc:
    raise ImportError(
        "No se ha encontrado 'config_privada.py'. "
        "La versión pública omite la configuración específica del dominio."
    ) from exc

categorias_operativas = cargar_categorias_operativas()
CLASS_MAP = cargar_mapa_clases()
PRED_LABEL_NORMALIZATION = cargar_normalizacion_etiquetas()
PISTAS_VISUALES = cargar_pistas_visuales()

print("Configuración de categorías cargada correctamente.")

# -----------------------------------------------------------
# 4. Reconstrucción de etiquetas operativas desde nombre de archivo
# -----------------------------------------------------------

col_imagen = "imagen"

def obtener_subclase_desde_archivo(ruta_o_nombre):
    """
    Reconstrucción de categorías operativas a partir
    de convenciones de nomenclatura del dataset.

    Las reglas concretas se omiten en la versión pública.
    """
    nombre = os.path.basename(str(ruta_o_nombre)).upper().strip()
    return obtener_categoria_desde_nombre(nombre)


rutas_imagenes = []

for ext in ("*.jpg", "*.jpeg", "*.png"):
    rutas_imagenes.extend(
        list(imagen_celulas_inicial.rglob(ext))
    )

rutas_imagenes = sorted(rutas_imagenes)

df = pd.DataFrame({
    col_imagen: [str(p) for p in rutas_imagenes]
})

df["subclase"] = df[col_imagen].apply(
    obtener_subclase_desde_archivo
)

df = df[
    df["subclase"] != "unknown"
].reset_index(drop=True)

df.to_csv(
    RUTA_CSV_RECONSTRUIDO,
    index=False,
    encoding="utf-8"
)

print("Dataset reconstruido correctamente.")
print(f"Imágenes válidas procesadas: {len(df)}")

     

# -----------------------------------------------------------
# 5. Split train / validation / test
# -----------------------------------------------------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["subclase"],
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=SEED,
    stratify=temp_df["subclase"],
)

train_df.to_csv(RUTA_TRAIN_CSV, index=False, encoding="utf-8")
val_df.to_csv(RUTA_VAL_CSV, index=False, encoding="utf-8")
test_df.to_csv(RUTA_TEST_CSV, index=False, encoding="utf-8")

print("TRAIN:", len(train_df))
print("VAL:", len(val_df))
print("TEST:", len(test_df))

     

# ----------------------------------------------------------
# 6. Dataset personalizado para ViT
# --------------------------------------------------------
modelo_vit_base = "google/vit-base-patch16-224"
processor_vit = ViTImageProcessor.from_pretrained(modelo_vit_base)

class_to_idx = {c: i for i, c in enumerate(categorias_operativas)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

class CellDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor, image_col="imagen", label_col="subclase"):
        self.df = df.reset_index(drop=True).copy()
        self.processor = processor
        self.image_col = image_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row[self.image_col]).convert("RGB")
        label = class_to_idx[str(row[self.label_col]).strip().lower()]
        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

train_dataset = CellDataset(train_df, processor_vit)
val_dataset = CellDataset(val_df, processor_vit)
test_dataset = CellDataset(test_df, processor_vit)

print("Datasets preparados correctamente.")
print("Shape de ejemplo:", train_dataset[0]["pixel_values"].shape)

     

# ----------------------------------------------------------
# 7. Entrenamiento de ViT para clasificación visual
# -----------------------------------------------------
id2label = idx_to_class
label2id = class_to_idx

model_vit = ViTForImageClassification.from_pretrained(
    modelo_vit_base,
    num_labels=len(categorias_operativas),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1_weighted": f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"],
        "precision_weighted": precision_metric.compute(predictions=preds, references=labels, average="weighted")["precision"],
        "recall_weighted": recall_metric.compute(predictions=preds, references=labels, average="weighted")["recall"],
    }

training_args = TrainingArguments(
    output_dir=str(BASE_DIR / "modelo" / "checkpoints_vit"),
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4,
    report_to="tensorboard",
    save_total_limit=2,
    remove_unused_columns=False,
    seed=SEED,
)

ruta_modelo_vit = BASE_DIR / "modelo" / "vit_categorias_final"

trainer = Trainer(
    model=model_vit,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(ruta_modelo_vit)
processor_vit.save_pretrained(ruta_modelo_vit)

print("Modelo ViT guardado en:", ruta_modelo_vit)

     
Generación V3 base con ViT + BLIP
En esta fase se recarga el modelo visual entrenado y se combina con BLIP base. El modelo visual actúa como apoyo de auditoría, mientras que BLIP aporta posibles pistas visuales. Solo se conservan señales suficientemente específicas y útiles. La descripción final se construye mediante una fusión controlada de información estructurada y evidencia visual.


# ------------------------------------------------------------
# 8. Cargar ViT entrenado y BLIP base
# ---------------------------------------------------------------
ruta_modelo_vit = BASE_DIR / "modelo" / "vit_categorias_final"

vit_processor = ViTImageProcessor.from_pretrained(ruta_modelo_vit)
vit_model = ViTForImageClassification.from_pretrained(ruta_modelo_vit).to(device)
vit_model.eval()

blip_model_name = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(blip_model_name)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name).to(device)
blip_model.config.tie_word_embeddings = False
blip_model.eval()

print("ViT cargado")
print("BLIP base cargado")

     

# --------------------------------------------------------------
# 9. Carga de recursos de normalización morfológica
# ----------------------------------------------------------------
ruta_norm = BASE_DIR / "datos" / "spec_morphology" / "vocab_normalization.yaml"
ruta_excel_desc = BASE_DIR / "datos" / "spec_morphology" / "pathologist_morphology.xlsx"

FEATURE_FIELDS = [
    "size",
    "nucleocytoplasmic_ratio",
    "nucleus_shape",
    "nucleus_segments",
    "nuclear_chromatin",
    "nucleoli",
    "cytoplasm_amount",
    "cytoplasm_color",
    "granulation",
    "inclusions",
    "villi",
]

def limpiar_texto(texto):
    if pd.isna(texto):
        return ""
    return " ".join(str(texto).strip().split())

def load_normalization(path: Path):
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Normalization spec root must be a dict")
    return data.get("value_map", {}), data.get("cell_type_map", {})

VALUE_MAP, CELL_TYPE_MAP = load_normalization(ruta_norm)

def normalize_value(value):
    text = limpiar_texto(value)
    if not text:
        return "undefined"
    text_lower = text.lower()
    if text in VALUE_MAP:
        text = limpiar_texto(VALUE_MAP[text])
    elif text_lower in VALUE_MAP:
        text = limpiar_texto(VALUE_MAP[text_lower])
    else:
        text = text_lower
    return text.lower()

def normalize_cell_type(value):
    text = limpiar_texto(value)
    if not text:
        return "Unknown"
    if text in CELL_TYPE_MAP:
        return CELL_TYPE_MAP[text]
    return text.title()

def valor(row, col):
    if col not in row or pd.isna(row[col]):
        return ""
    return limpiar_texto(row[col])

df_wbc = pd.read_excel(ruta_excel_desc, sheet_name="WBC_features")
df_abn = pd.read_excel(ruta_excel_desc, sheet_name="some_abnormal_cell_features")
df_ref = pd.concat([df_wbc, df_abn], ignore_index=True)
df_ref.columns = [str(c).strip() for c in df_ref.columns]

COL_TIPO = "TYPE OF BLOOD CELL"
df_ref = df_ref[df_ref[COL_TIPO].notna()].copy()
df_ref[COL_TIPO] = df_ref[COL_TIPO].astype(str).str.strip().str.upper()
df_ref = df_ref[df_ref[COL_TIPO] != ""]

print("Tipos de referencia:", sorted(df_ref[COL_TIPO].unique()))

     

# -----------------------------------------------------------
# 10. Funciones auxiliares para V3 base
# -----------------------------------------------------------
def extraer_clase_real(img_path):
    return obtener_subclase_desde_archivo(img_path.name if hasattr(img_path, "name") else img_path)

def normalizar_clase_referencia(clase_real):
    return CLASS_MAP.get(clase_real, clase_real.upper())

def class_name_v3_from_real(clase_real):
    return normalize_cell_type(normalizar_clase_referencia(clase_real))

def normalizar_pred_label(pred_label):
    texto = str(pred_label).strip().lower()
    return PRED_LABEL_NORMALIZATION.get(texto, texto)

def class_name_v3_from_pred(pred_label):
    return normalize_cell_type(CLASS_MAP.get(pred_label, pred_label.upper()))

def construir_morfologia_prototipo(row):
    morph = {
        "size": normalize_value(valor(row, "SIZE")),
        "nucleocytoplasmic_ratio": normalize_value(valor(row, "NULEOCYTOPLASMIC RATIO")),
        "nucleus_shape": normalize_value(valor(row, "NUCLEUS SHAPE")),
        "nucleus_segments": normalize_value(valor(row, "SEGMENTS OF THE NUCLEUS")) if "SEGMENTS OF THE NUCLEUS" in row else "undefined",
        "nuclear_chromatin": normalize_value(valor(row, "NUCLEAR CHROMATIN")),
        "nucleoli": normalize_value(valor(row, "NUCLEOLI")),
        "cytoplasm_amount": normalize_value(valor(row, "CYTOPLASM")),
        "cytoplasm_color": normalize_value(valor(row, "COLOUR CYTOPL")),
        "granulation": normalize_value(valor(row, "GRANULATION")),
        "inclusions": normalize_value(valor(row, "INCLUSIONS")) if "INCLUSIONS" in row else "undefined",
        "villi": normalize_value(valor(row, "VILLI")) if "VILLI" in row else "undefined",
    }
    for field in FEATURE_FIELDS:
        if not morph.get(field):
            morph[field] = "undefined"
    return morph

prototipos_clase = {
    valor(row, COL_TIPO).upper(): construir_morfologia_prototipo(row)
    for _, row in df_ref.iterrows()
}

def obtener_prototipo_por_clase_real(clase_real):
    return dict(prototipos_clase.get(normalizar_clase_referencia(clase_real), {}))

def obtener_prototipo_por_clase_predicha(pred_label):
    class_name = class_name_v3_from_pred(pred_label)
    for raw_name, morph in prototipos_clase.items():
        if normalize_cell_type(raw_name) == class_name:
            return dict(morph)
    return {}

def fill_missing_fields(morph):
    out = dict(morph)
    for field in FEATURE_FIELDS:
        if field not in out or out[field] in {None, ""}:
            out[field] = "undefined"
    return out

     

# ------------------------------------------------------------
# 11. Limpieza de captions y extracción de pistas BLIP
# -------------------------------------------------------------
def limpiar_caption(texto):
    texto = limpiar_texto(texto).lower()
    reemplazos = [
        "this image shows", "the image shows", "microscopic image of",
        "a microscopic image of", "microscopic view of", "an image of",
        "image of", "photo of", "picture of",
    ]
    for r in reemplazos:
        texto = texto.replace(r, " ")
    texto = re.sub(r"\s+", " ", texto).strip(" .,")
    if not texto:
        return ""
    texto = texto.replace("a a ", "a ")
    texto = texto.replace("cells in the cell", "cell")
    texto = texto.replace("showing the cells", "")
    texto = re.sub(r"\s+", " ", texto).strip(" .,")
    if not texto:
        return ""
    texto = texto[0].upper() + texto[1:]
    if not texto.endswith("."):
        texto += "."
    return texto

def caption_blip_util(texto):
    texto = limpiar_caption(texto).lower()
    if not texto:
        return 0
    basura_exacta = {
        "a cell.", "cell.", "a blood cell.", "blood cell.",
        "a white blood cell.",
        "describe the visible morphology of this blood cell image.",
    }
    if texto in basura_exacta:
        return 0
    patrones_ruido = [
        "describe the visible morphology", "focus on nucleus",
        "overall cell appearance", "white background",
        "blood vessel", "purple substance", "circle with",
    ]
    if any(p in texto for p in patrones_ruido):
        return 0
    pistas_fuertes = PISTAS_VISUALES
    return int(any(p in texto for p in pistas_fuertes))


def extraer_cues_blip(caption_blip):
    """
    Extracción de pistas visuales.
    Las reglas específicas se omiten en la versión pública.
    """
    return extraer_cues_visuales(caption_blip)

     

# ----------------------------------------------------------
# 12. Fusión morfológica y renderizado de caption V3
# ---------------------------------------------------------
def fusionar_prototipo_y_cues(prototype, blip_cues):
    morph = fill_missing_fields(prototype)
    for k, v in blip_cues.items():
        current = morph.get(k, "undefined")
        if current == "undefined":
            morph[k] = v
            continue
        if k == "nucleus_shape" and current in {"regular", "undefined"}:
            morph[k] = v
        elif k == "granulation" and current in {"undefined", "absent", "occasional"}:
            morph[k] = v
    return morph

def build_confidence_dict(morph, blip_cues_conf):
    conf = {}
    for field_name in FEATURE_FIELDS:
        value = morph.get(field_name, "undefined")
        if field_name in blip_cues_conf:
            conf[field_name] = float(blip_cues_conf[field_name])
        elif value == "undefined":
            conf[field_name] = 0.40
        else:
            conf[field_name] = 0.95
    return conf

def render_caption_v3(class_name, morph):
    parts = []
    if morph.get("size", "undefined") != "undefined":
        parts.append(f"{morph['size']} size")
    if morph.get("nucleocytoplasmic_ratio", "undefined") != "undefined":
        parts.append(f"{morph['nucleocytoplasmic_ratio']} nucleocytoplasmic ratio")
    if morph.get("nucleus_shape", "undefined") != "undefined":
        nucleus_txt = f"a {morph['nucleus_shape']} nucleus"
        if morph.get("nucleus_segments", "undefined") != "undefined":
            nucleus_txt += f" ({morph['nucleus_segments']})"
        parts.append(nucleus_txt)
    if morph.get("nuclear_chromatin", "undefined") != "undefined":
        parts.append(f"{morph['nuclear_chromatin']} chromatin")
    if morph.get("nucleoli", "undefined") != "undefined":
        parts.append(f"{morph['nucleoli']} nucleoli")

    cyto_parts = []
    if morph.get("cytoplasm_amount", "undefined") != "undefined":
        cyto_parts.append(morph["cytoplasm_amount"])
    if morph.get("cytoplasm_color", "undefined") != "undefined":
        cyto_parts.append(morph["cytoplasm_color"])
    if cyto_parts:
        parts.append(" ".join(cyto_parts) + " cytoplasm")

    if morph.get("granulation", "undefined") not in {"undefined", "absent"}:
        parts.append(f"{morph['granulation']} granulation")
    if morph.get("inclusions", "undefined") == "absent":
        parts.append("absent inclusions")
    elif morph.get("inclusions", "undefined") not in {"undefined", ""}:
        parts.append(f"inclusions {morph['inclusions']}")
    if morph.get("villi", "undefined") == "absent":
        parts.append("absent villi")
    elif morph.get("villi", "undefined") not in {"undefined", ""}:
        parts.append(f"villi {morph['villi']}")

    if not parts:
        return f"A {class_name} cell."
    if len(parts) == 1:
        return f"A {class_name} cell with {parts[0]}."
    return f"A {class_name} cell with " + ", ".join(parts[:-1]) + f", and {parts[-1]}."

     

# ------------------------------------------------------------
# 13. Procesamiento de imágenes y generación del dataset V3 base
# ------------------------------------------------------------
def abrir_imagen_segura(img_path, reintentos=3):
    for intento in range(reintentos):
        try:
            return Image.open(img_path).convert("RGB")
        except Exception as e:
            if intento < reintentos - 1:
                time.sleep(1)
            else:
                raise e

ext_validas = {".jpg", ".jpeg", ".png"}
imagenes = sorted([p for p in imagen_celulas_inicial.rglob("*") if p.suffix.lower() in ext_validas])
imagenes = [p for p in imagenes if extraer_clase_real(p) in categorias_operativas]

resultados = []

for i, img_path in enumerate(imagenes):
    try:
        img = abrir_imagen_segura(img_path)
        clase_real = extraer_clase_real(img_path)
        if clase_real not in categorias_operativas:
            continue

        # ViT como apoyo visual/auditoría
        vit_inputs = vit_processor(images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            vit_outputs = vit_model(**vit_inputs)
            probs = torch.softmax(vit_outputs.logits, dim=-1)

        pred_id = probs.argmax(dim=-1).item()
        pred_label_raw = vit_model.config.id2label[pred_id]
        pred_label = normalizar_pred_label(pred_label_raw)
        pred_prob = float(probs[0, pred_id].detach().cpu().item())

        # Categoría base y representación morfológica
        class_name = class_name_v3_from_real(clase_real)
        prototype_morph = obtener_prototipo_por_clase_real(clase_real)
        if not prototype_morph:
            class_name = class_name_v3_from_pred(pred_label)
            prototype_morph = obtener_prototipo_por_clase_predicha(pred_label)
        prototype_morph = fill_missing_fields(prototype_morph)

        # BLIP base sin prompt para evitar eco del prompt
        blip_inputs = blip_processor(images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            ids = blip_model.generate(
                **blip_inputs,
                max_new_tokens=40,
                num_beams=4,
                repetition_penalty=1.15,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )

        caption_blip_raw = blip_processor.decode(ids[0], skip_special_tokens=True)
        caption_blip = limpiar_caption(caption_blip_raw)
        caption_util = int(caption_blip_util(caption_blip))
        blip_cues, blip_cues_conf = extraer_cues_blip(caption_blip)

        morphology_v3 = fusionar_prototipo_y_cues(prototype_morph, blip_cues)
        morphology_v3["confidence"] = build_confidence_dict(morphology_v3, blip_cues_conf)

        class_prototype_caption = render_caption_v3(class_name, prototype_morph)
        caption_v3_base = render_caption_v3(class_name, morphology_v3)
        hybrid_weak_caption = (
            f"{class_prototype_caption} Visual hint: {caption_blip}"
            if caption_util and caption_blip else class_prototype_caption
        )

        resultados.append({
            "image_id": img_path.stem,
            "class_name": class_name,
            "nombre_imagen": img_path.name,
            "numero_imagen": img_path.stem,
            "carpeta_origen": img_path.parent.name.strip().lower(),
            "imagen": str(img_path),
            "clase_real": clase_real,
            "clase_real_referencia": normalizar_clase_referencia(clase_real),
            "clase_predicha_raw": str(pred_label_raw).strip(),
            "clase_predicha": pred_label,
            "probabilidad": round(pred_prob, 6),
            "image_caption_blip_raw": limpiar_caption(caption_blip_raw),
            "image_caption_blip": caption_blip,
            "caption_blip_util": caption_util,
            "blip_cues_json": json.dumps(blip_cues, ensure_ascii=True, sort_keys=True),
            "class_prototype_morphology_json": json.dumps(prototype_morph, ensure_ascii=True, sort_keys=True),
            "class_prototype_caption": class_prototype_caption,
            "morphology_v3_json": json.dumps(morphology_v3, ensure_ascii=True, sort_keys=True),
            "caption_v3_base": caption_v3_base,
            "hybrid_weak_caption": hybrid_weak_caption,
            "error_tipo": "",
            "error_msg": "",
        })

        if (i + 1) % 200 == 0:
            print(f"Procesadas {i+1}/{len(imagenes)}")

    except Exception as e:
        resultados.append({
            "image_id": img_path.stem if hasattr(img_path, "stem") else "",
            "class_name": "",
            "nombre_imagen": img_path.name if hasattr(img_path, "name") else "",
            "numero_imagen": img_path.stem if hasattr(img_path, "stem") else "",
            "carpeta_origen": img_path.parent.name.strip().lower() if hasattr(img_path, "parent") else "",
            "imagen": str(img_path),
            "clase_real": extraer_clase_real(img_path) if hasattr(img_path, "stem") else "",
            "clase_real_referencia": "",
            "clase_predicha_raw": "",
            "clase_predicha": "error",
            "probabilidad": None,
            "image_caption_blip_raw": "",
            "image_caption_blip": "",
            "caption_blip_util": 0,
            "blip_cues_json": "{}",
            "class_prototype_morphology_json": "{}",
            "class_prototype_caption": "",
            "morphology_v3_json": "{}",
            "caption_v3_base": "",
            "hybrid_weak_caption": "",
            "error_tipo": type(e).__name__,
            "error_msg": str(e),
        })
        print(f"Error en {img_path}: {e}")

print("Procesamiento finalizado:", len(resultados))

     

#-----------------------------------------------------------
# 14. Limpieza y guardado de V3 base
# -----------------------------------------------------------
df_final = pd.DataFrame(resultados)

df_final["clase_real"] = df_final["clase_real"].astype(str).str.strip().str.lower()
df_final["clase_predicha"] = df_final["clase_predicha"].astype(str).str.strip().str.lower()

df_errores_tecnicos = df_final[df_final["clase_predicha"] == "error"].copy()
df_validas = df_final[df_final["clase_predicha"] != "error"].copy()
df_errores_clasificacion = df_validas[df_validas["clase_real"] != df_validas["clase_predicha"]].copy()

df_clean = df_validas.copy()
df_clean = df_clean[df_clean["caption_v3_base"].notna()]
df_clean = df_clean[df_clean["caption_v3_base"].str.strip() != ""]
df_clean.reset_index(drop=True, inplace=True)

ruta_v3_completa = BASE_DIR / "datos" / "dataset_v3_base_vit_blip_completo.csv"
ruta_v3_limpia = BASE_DIR / "datos" / "dataset_v3_base_vit_blip_clean.csv"
ruta_v3_errores_tecnicos = BASE_DIR / "datos" / "dataset_v3_base_errores_tecnicos.csv"
ruta_v3_errores_clasificacion = BASE_DIR / "datos" / "dataset_v3_base_errores_clasificacion.csv"

df_final.to_csv(ruta_v3_completa, index=False, encoding="utf-8")
df_clean.to_csv(ruta_v3_limpia, index=False, encoding="utf-8")
df_errores_tecnicos.to_csv(ruta_v3_errores_tecnicos, index=False, encoding="utf-8")
df_errores_clasificacion.to_csv(ruta_v3_errores_clasificacion, index=False, encoding="utf-8")

print("Dataset V3 completo:", ruta_v3_completa)
print("Dataset V3 limpio/base:", ruta_v3_limpia)
print("Errores técnicos:", ruta_v3_errores_tecnicos)
print("Errores de clasificación:", ruta_v3_errores_clasificacion)
print("Tamaño dataset completo:", len(df_final))
print("Tamaño dataset limpio:", len(df_clean))
print("BLIP útil en:", int(df_validas["caption_blip_util"].sum()) if len(df_validas) else 0, "imágenes")

     
Archivo resultante
El archivo principal generado por este notebook es:

dataset_v3_base_vit_blip_clean.csv

Este archivo corresponde a la V3 base. A partir de él se pueden entrenar o comparar las siguientes aproximaciones:

BLIP + LoRA, usando las captions V3 como texto objetivo.
Gemma 3, usando la V3 base como referencia estructurada para generación/evaluación.
Qwen-VL, usando la V3 base como referencia estructurada para generación/evaluación.
La V3 base queda separada de los notebooks posteriores para que el flujo experimental sea más claro y reproducible, sin exponer en exceso los criterios internos utilizados durante la construcción del dataset.