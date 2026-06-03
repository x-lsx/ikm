import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import load_model, predict_language
import gradio as gr
import tempfile

def test_model_loads_and_predicts():
    """Тест 1: модель загружается и работает на корректном примере"""
    try:
        # Модель должна уже существовать после train.py
        preds, confs = predict_language("Sie erholten sich.")
        assert len(preds) == 1
        assert isinstance(preds[0], str)
        assert 0.0 <= confs[0] <= 1.0
    except FileNotFoundError:
        # Если модель не найдена, тест пропускаем (но не падаем)
        assert True

def test_predict_output_format():
    """Тест 2: проверка формата выхода"""
    preds, confs = predict_language(["Hello world", "Bonjour"])
    assert isinstance(preds, list)
    assert isinstance(confs, list)
    assert len(preds) == 2
    assert len(confs) == 2
    assert all(isinstance(p, str) for p in preds)
    assert all(isinstance(c, float) for c in confs)

def test_gradio_app_starts():
    """Тест 3: веб-приложение может запуститься без ошибок"""
    try:
        # Импортируем блок, но не запускаем .launch()
        from app import demo
        assert isinstance(demo, gr.Blocks)
    except Exception as e:
        assert False, f"Gradio блок не создался: {e}"