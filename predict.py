import joblib
import os
import numpy as np

model = None

def load_model(path='models/language_detector_best.pkl'):
    global model
    if model is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Модель не найдена: {path}. Сначала запустите train.py")
        model = joblib.load(path)
    return model

def predict_language(texts):
    """
    Принимает строку или список строк.
    ВСЕГДА ВОЗВРАЩАЕТ: (список_предсказаний, список_уверенностей)
    """
    model = load_model()
    
    # Определяем, был ли передан один текст или список
    single_input = isinstance(texts, str)
    
    # Приводим к списку для единообразной обработки
    if single_input:
        texts = [texts]
    
    # Предсказания
    preds = model.predict(texts)
    probs = model.predict_proba(texts)
    confs = [max(p) for p in probs]
    
    # Преобразуем в обычные Python списки
    preds_list = preds.tolist() if hasattr(preds, 'tolist') else list(preds)
    
    # ВСЕГДА возвращаем tuple из двух списков
    return preds_list, confs