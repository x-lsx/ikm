import gradio as gr
from predict import predict_language

def detect(text):
    if not text.strip():
        return "Введите текст", 0.0
    
    # predict_language возвращает (список, список)
    preds, confs = predict_language(text)
    
    # Извлекаем первый элемент
    return preds[0], confs[0]

with gr.Blocks(title="Определитель языка") as demo:
    gr.Markdown("# Определение языка текста")
    gr.Markdown("Введите предложение, и модель определит язык (eng/deu/ita/por и др.)")
    
    with gr.Row():
        text_input = gr.Textbox(label="Текст", placeholder="Например: Sie erholten sich.", lines=3)
        lang_output = gr.Textbox(label="Язык", interactive=False)
        conf_output = gr.Number(label="Уверенность", interactive=False)
    
    btn = gr.Button("Определить")
    btn.click(fn=detect, inputs=text_input, outputs=[lang_output, conf_output])
    
    gr.Examples(
        examples=[
            ["Sie erholten sich nach der langen Reise."],
            ["Hello, how are you?"],
            ["Bonjour tout le monde"],
            ["Всем привет, как дела?"],
            ["Prestami attenzione, per favore."]
        ],
        inputs=text_input
    )

if __name__ == "__main__":
    demo.launch()