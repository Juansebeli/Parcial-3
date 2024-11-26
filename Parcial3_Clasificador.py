#!pip install gradio transformers

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
import pandas as pd  

# Cargar el tokenizer y el modelo
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Cambia si usas un tokenizer diferente
model = AutoModelForSequenceClassification.from_pretrained("Esmarguz/econ-classifier-multitopic")

# Crear el pipeline para la clasificación
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, function_to_apply="sigmoid", top_k=None)

# Diccionario de etiquetas originales
etiquetas_originales = {
    0: "Macroeconomía",
    1: "Microeconomía",
    2: "Econometría",
    3: "Laboral",
    4: "Internacional",
    5: "Desarrollo",
    6: "Pública",
    7: "Ambiental",
    8: "Salud",
    9: "Financiera",
    10: "Comportamental",
    11: "Sostenibilidad"
}

# === CONFIGURACIÓN DEL CHATBOT ===
idea_generation = pipeline("summarization", 
                           model="facebook/bart-large-cnn"
                           )

# Función para convertir las predicciones
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


# Función chatbot
def chatbot(abstract):
    # Simulando el modelo que devuelve un resumen y categorías
    resumen = idea_generation(abstract, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    
    ## Obtener resultados del modelo
    resultados = pipe(abstract)
    
    # Convertir las predicciones
    etiquetas_predichas = convertir_predicciones(resultados)

    # Obtener la categoría principal y un resumen
    categoria_principal = max(etiquetas_predichas, key=etiquetas_predichas.get)
    porcentaje_categoria_principal = etiquetas_predichas[categoria_principal]
    
    # Obtener las otras dos categorías
    segunda_categoria = sorted(etiquetas_predichas, key=etiquetas_predichas.get, reverse=True)[1]
    tercera_categoria = sorted(etiquetas_predichas, key=etiquetas_predichas.get, reverse=True)[2]
    
    lista_categorias = [categoria_principal, segunda_categoria, tercera_categoria]
    lista_porcentajes = [round(porcentaje_categoria_principal * 100, 1), 
                         round(etiquetas_predichas[segunda_categoria] * 100, 1), 
                         round(etiquetas_predichas[tercera_categoria] * 100, 1)]
    

    
    
    # Crear un DataFrame para mostrarlo en una tabla
    df = pd.DataFrame({
        "Categoría": lista_categorias,
        "Porcentaje": lista_porcentajes
    })
    
    # Retornar las categorías, los porcentajes y el resumen
    return df, resumen

# Configurar la interfaz con Gradio
interface = gr.Interface(
    fn=chatbot,  # Función que procesa el abstract
    inputs=[
        gr.Textbox(
            placeholder="Ingresa el abstract del paper aquí",
            label="Abstract",
            lines=5  # Número de líneas visibles en el textbox
        )
    ],
    outputs=[
        gr.DataFrame(label="Categorías y Porcentajes"),  # Salida de la tabla
        gr.Textbox(label="Resumen")  # Salida del resumen
    ],
    title="Clasificador de Abstracts Económicos",
    description="Inserta el abstract de un paper para obtener las tres categorías más relevantes, sus porcentajes y un resumen.",
)

# Iniciar la interfaz
if __name__ == "__main__":
    interface.launch()
