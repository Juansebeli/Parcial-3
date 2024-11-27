# Instalación de dependencias necesarias
#!pip install gradio transformers torch pandas

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import gradio as gr

# Configuración del dispositivo para el uso de GPU si está disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

# === SECCIÓN 1: Análisis de Títulos de Papers Económicos ===

# Configuración del Chatbot para Generación de Texto
chat_generator = pipeline(
    'text-generation',
    model='microsoft/DialoGPT-small',
    device=device
)

# Configuración del modelo de análisis de sentimientos
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_model.to(device)

def chatbot(titulo):
    if not titulo.strip():
        return "Por favor, escribe un título.", None, None

    try:
        # Generar respuesta del chatbot con el título
        respuesta = chat_generator(
            titulo,
            max_length=500,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=1
        )[0]['generated_text']
        respuesta = respuesta.replace(titulo, "").strip()

        # Análisis de sentimiento del título
        inputs = sentiment_tokenizer(titulo, return_tensors="pt", truncation=True,
                                     max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = sentiment_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        rating = torch.argmax(predictions).item() + 1
        confidence = predictions[0][rating-1].item()

        # Determinar el sentimiento
        if rating < 2:
            sentimiento = "Despropiado"
        elif rating == 2:
            sentimiento = "Poco Apropiado"
        elif rating == 3:
            sentimiento = "Neutral"
        elif rating == 4:
            sentimiento = "Apropiado"
        else:
            sentimiento = "Muy Apropiado"

        sentimiento_completo = f"{sentimiento} ({rating} estrellas)"
        confianza = round(confidence * 100, 2)

        return respuesta, sentimiento_completo, confianza

    except Exception as e:
        return f"Error: {str(e)}", "Error en el análisis", 0.0

# === SECCIÓN 2: Clasificador de Abstracts Económicos ===

# Cargar el tokenizer y el modelo para la clasificación de abstracts
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("Esmarguz/econ-classifier-multitopic")

# Crear el pipeline para la clasificación
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, function_to_apply="sigmoid", top_k=None)

# Diccionario de etiquetas originales
etiquetas_originales = {
    0: "Macroeconomía", 1: "Microeconomía", 2: "Econometría", 3: "Laboral", 4: "Internacional",
    5: "Desarrollo", 6: "Pública", 7: "Ambiental", 8: "Salud", 9: "Financiera", 10: "Comportamental", 11: "Sostenibilidad"
}

# Función para convertir las predicciones del modelo
def convertir_predicciones(resultados, umbral=0.3):
    if isinstance(resultados, list) and isinstance(resultados[0], list):
        resultados = resultados[0]

    probabilidades_con_etiquetas = {
        etiquetas_originales[int(item['label'].replace('LABEL_', ''))]: item['score']
        for item in resultados
        if item['score'] >= umbral
    }

    etiquetas_predichas = sorted(probabilidades_con_etiquetas.items(), key=lambda x: x[1], reverse=True)[:3]
    return dict(etiquetas_predichas)

# Generación de resumen para el abstract
idea_generation = pipeline("summarization", model="facebook/bart-large-cnn")

# Función para procesar el abstract
def clasificar_abstract(abstract):
    # Generar resumen del abstract
    resumen = idea_generation(abstract, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    
    # Obtener resultados de la clasificación
    resultados = pipe(abstract)
    
    # Convertir las predicciones en categorías
    etiquetas_predichas = convertir_predicciones(resultados)

    # Ordenar las categorías por relevancia
    categoria_principal = max(etiquetas_predichas, key=etiquetas_predichas.get)
    porcentaje_categoria_principal = etiquetas_predichas[categoria_principal]
    
    segunda_categoria = sorted(etiquetas_predichas, key=etiquetas_predichas.get, reverse=True)[1]
    tercera_categoria = sorted(etiquetas_predichas, key=etiquetas_predichas.get, reverse=True)[2]
    
    lista_categorias = [categoria_principal, segunda_categoria, tercera_categoria]
    lista_porcentajes = [round(porcentaje_categoria_principal * 100, 1), 
                         round(etiquetas_predichas[segunda_categoria] * 100, 1), 
                         round(etiquetas_predichas[tercera_categoria] * 100, 1)]

    # Crear un DataFrame para mostrar las categorías y porcentajes
    df = pd.DataFrame({
        "Categoría": lista_categorias,
        "Porcentaje": lista_porcentajes
    })

    return df, resumen
# === INTERFAZ DE GRADIO ===

def interfaz_completa():
    with gr.Blocks() as demo:
        with gr.Tab("Análisis de Títulos de Papers Económicos"):
            gr.Interface(
                fn=chatbot,
                inputs=[
                    gr.Textbox(
                        placeholder="Escribe el título del paper...",
                        label="Título del Paper",
                        lines=2
                    )
                ],
                outputs=[
                    gr.Textbox(label="Explicación del Título (Generada por el Chatbot)"),
                    gr.Label(label="Evaluación del Título"),
                    gr.Number(label="Confianza del Análisis (%)")
                ],
                title="Análisis de Títulos de Papers Económicos",
                description="Este modelo analiza la relevancia de un título de paper en el campo económico y genera una breve explicación.",
                examples=[
                    ["Impact of Inflation on Economic Growth in Developing Countries"],
                    ["The Role of Microfinance in Alleviating Poverty in Rural Areas"],
                    ["Global Trade Patterns and their Effect on Emerging Economies"]
                ],
                allow_flagging="never",
                cache_examples=True
            )

        with gr.Tab("Clasificador de Abstracts Económicos"):
            gr.Interface(
                fn=clasificar_abstract,
                inputs=[
                    gr.Textbox(
                        placeholder="Ingresa el abstract del paper aquí",
                        label="Abstract",
                        lines=5
                    )
                ],
                outputs=[
                    gr.DataFrame(label="Categorías y Porcentajes"),
                    gr.Textbox(label="Resumen")
                ],
                title="Clasificador de Abstracts Económicos",
                description="Inserta el abstract de un paper para obtener las tres categorías más relevantes, sus porcentajes y un resumen.",
            )

    demo.launch(share=True)


# Ejecutar la interfaz
if __name__ == "__main__":
    interfaz_completa()
