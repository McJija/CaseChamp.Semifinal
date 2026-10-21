"""Прототип системы поддержки врача: один экран.

Запуск — двойной щелчок по start_windows.bat (macOS и Linux: bash start_mac_linux.sh) или вручную:
    python app/launch.py            # сам выберет порт и откроет браузер
    streamlit run app/app.py        # то же через Streamlit

Вся логика модели — в src/inference.py; здесь только показ результата.
Параметры адреса (для показа и скриншотов): ?example=<файл из app/examples> — выбрать пример,
?limits=1 — раскрыть блок «Ограничения системы».
Снимки-примеры лежат в app/examples/ (снимки проверочной части, на них модели не учились).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import FLAG_ATTENTION, load_predictor  # noqa: E402
from src.labels import CLASSES, CLS_RU  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent / "examples"
DISCLAIMER = "Результат работы системы не является медицинским заключением. Решение принимает врач."
MASK_COLOR = np.array([255, 60, 60], np.float32)  # RGB


@st.cache_resource(show_spinner="Загрузка моделей…")
def get_predictor():
    return load_predictor(models_dir=ROOT / "models")


def decode(data: bytes) -> np.ndarray:
    """Байты файла → серый снимок uint8 — так же, как читаются снимки при обучении."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("файл не удалось прочитать как изображение")
    return img


def with_mask(gray: np.ndarray, mask: np.ndarray | None, alpha: float) -> np.ndarray:
    """Снимок с полупрозрачной маской и ее контуром (RGB)."""
    rgb = np.repeat(gray[..., None], 3, 2).astype(np.float32)
    if mask is not None and mask.any():
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * MASK_COLOR
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rgb = cv2.drawContours(np.ascontiguousarray(rgb), contours, -1, MASK_COLOR.tolist(), 1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def with_heatmap(gray: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    """Снимок с тепловой картой Grad-CAM (RGB)."""
    rgb = np.repeat(gray[..., None], 3, 2)
    color = cv2.cvtColor(cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, color, alpha, 0)


st.set_page_config(page_title="МРТ: поддержка врача", layout="wide")
st.title("Опухоли головного мозга на МРТ — система поддержки врача")
st.warning(DISCLAIMER)

with st.sidebar:
    st.header("Снимок")
    uploaded = st.file_uploader("Загрузите снимок (JPG или PNG)", type=["jpg", "jpeg", "png"])
    examples = sorted(p.name for p in EXAMPLES.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if EXAMPLES.exists() else []
    options = ["—", *examples]
    wanted = st.query_params.get("example", "—")
    example = st.selectbox("или выберите пример", options, index=options.index(wanted) if wanted in options else 0) if examples else "—"
    st.header("Отображение")
    mask_alpha = st.slider("Прозрачность маски", 0.0, 1.0, 0.45, 0.05)
    heat_alpha = st.slider("Прозрачность тепловой карты", 0.0, 1.0, 0.45, 0.05)
    st.caption("Условия применения: МРТ головного мозга, режим T1 с контрастом. Для других режимов и для КТ система не предназначена.")

if uploaded is not None:
    name, data = uploaded.name, uploaded.getvalue()
elif example != "—":
    name, data = example, (EXAMPLES / example).read_bytes()
else:
    st.info("Загрузите снимок или выберите пример в панели слева.")
    st.stop()

try:
    image = decode(data)
except ValueError as e:
    st.error(f"{name}: {e}")
    st.stop()

with st.spinner("Обработка снимка…"):
    result = get_predictor().predict(image)

left, middle, right = st.columns([4, 4, 3])
with left:
    st.subheader("Область опухоли")
    st.image(with_mask(image, result["mask"], mask_alpha), width="stretch")
    if result["mask_area_px"]:
        st.caption(f"Выделенная область — {result['mask_area_fraction']:.1%} кадра.")
    else:
        st.caption("Область не выделена.")
with middle:
    st.subheader("На что смотрела модель")
    st.image(with_heatmap(image, result["heatmap"], heat_alpha), width="stretch")
    st.caption(f"Тепловая карта Grad-CAM для класса «{CLS_RU[result['heatmap_class']]}»: красным — участки, сильнее всего повлиявшие на ответ.")
with right:
    st.subheader("Ответ системы")
    st.metric(result["class_ru"].capitalize(), f"{result['confidence']:.0%}", help="Уверенность модели в выбранном классе")
    if FLAG_ATTENTION in result["flags"]:
        st.error(f"Требует внимания врача: {result['flag_reason']}.")
    for cls in CLASSES:
        st.progress(float(result["probs"][cls]), text=f"{CLS_RU[cls].capitalize()} — {result['probs'][cls]:.1%}")
    st.caption(f"Файл: {name}, {image.shape[1]}×{image.shape[0]} пикселей.")

with st.expander("Ограничения системы", expanded=st.query_params.get("limits") == "1"):
    st.markdown(
        "- Система обучена на открытых наборах 2D-срезов; на снимках другого томографа или режима качество не проверялось.\n"
        "- Различает три типа опухоли (глиома, менингиома, опухоль гипофиза) и их отсутствие; другие находки не распознает.\n"
        "- Сторону поражения не сообщает: часть обучающих снимков зеркальна.\n"
        "- Уверенность модели — не вероятность диагноза."
    )
