Proyecto de Clasificación y Análisis de Papers Económicos

Este repositorio contiene un proyecto que incluye un modelo de clasificación de textos económicos y una interfaz interactiva desarrollada con Gradio. El proyecto tiene dos funcionalidades principales:

Análisis de Títulos de Papers Económicos: Utiliza un modelo de lenguaje para generar explicaciones basadas en el título de un artículo, junto con una evaluación del sentimiento asociado con el título.
Clasificador de Abstracts Económicos: Clasifica abstracts de papers económicos en diversas categorías y genera un resumen de cada abstract.
Objetivos del Proyecto

El objetivo de este proyecto es proporcionar una herramienta interactiva que permita:

Analizar y generar explicaciones sobre títulos de papers económicos.
Clasificar abstracts de papers económicos en distintas categorías relacionadas con áreas clave de la economía.
Presentar los resultados de forma clara y accesible para los usuarios a través de una interfaz gráfica.
Modelos Utilizados

1. Modelo de Fine-Tuning: Esmarguz/econ-classifier-multitopic
Este modelo es un modelo de clasificación de textos fine-tuneado específicamente para identificar temas económicos en abstracts de papers. Las categorías clasificadas por el modelo incluyen temas como macroeconomía, microeconomía, econometría, y más. Este modelo fue entrenado sobre un conjunto de datos económico y tiene la capacidad de clasificar abstracts en diversas áreas de la economía.

Categorías:
Macroeconomía
Microeconomía
Econometría
Laboral
Internacional
Desarrollo
Pública
Ambiental
Salud
Financiera
Comportamental
Sostenibilidad
2. Generación de Explicaciones de Títulos y Análisis de Sentimiento
El modelo también incluye una capacidad de análisis de sentimiento para los títulos de los papers. Utiliza una red neuronal para generar explicaciones sobre el título de un paper económico, analizando si el título tiene connotaciones positivas, negativas o neutrales.

Requisitos

Para ejecutar este proyecto en tu máquina local, asegúrate de tener las siguientes librerías instaladas:

            pip install gradio transformers torch pandas


Cómo Ejecutar el Proyecto

1. Clonar el Repositorio
Primero, clona este repositorio a tu máquina local:

            git clone https://github.com/tu-usuario/nombre-del-repositorio.git
            cd nombre-del-repositorio

2. Ejecutar el Script de la Interfaz
Una vez que el repositorio esté clonado, puedes ejecutar el archivo Python para iniciar la interfaz de Gradio:

python app.py

Esto abrirá una interfaz interactiva en tu navegador, donde podrás probar las dos funcionalidades principales:

Análisis de Títulos de Papers Económicos
Clasificador de Abstracts Económicos
3. Uso de la Interfaz
Análisis de Títulos de Papers Económicos:

Ingresa el título de un paper en el cuadro de texto.
El sistema generará una explicación sobre el título y mostrará una evaluación del sentimiento asociado con el título (Positivo, Negativo, Neutral).
Clasificador de Abstracts Económicos:

Ingresa el abstract de un paper económico en el cuadro de texto.
El sistema clasificará el abstract en las categorías económicas más relevantes y proporcionará un resumen del texto.
Ejemplos

A continuación, se proporcionan algunos ejemplos de entrada para las funcionalidades del sistema:

Ejemplo 1: Análisis de Títulos
Título: "Impact of Inflation on Economic Growth in Developing Countries"

Salida: Explicación generada sobre el título + Sentimiento: Neutral (3 estrellas).
Ejemplo 2: Clasificación de Abstracts
Abstract: "This paper examines the effects of government intervention on market efficiency in emerging economies..."

Salida: Categorías: Desarrollo, Internacional, Microeconomía + Resumen: "Este artículo examina los efectos de la intervención gubernamental sobre la eficiencia del mercado..."
Contribuciones

Si deseas contribuir a este proyecto, por favor sigue estos pasos:

Fork este repositorio.
Crea una nueva rama (git checkout -b feature-nueva).
Realiza tus cambios y haz commit de ellos (git commit -am 'Agrega nueva funcionalidad').
Envía tus cambios al repositorio remoto (git push origin feature-nueva).
Abre un pull request.
Licencia

Este proyecto está bajo la Licencia MIT. Para más detalles, consulta el archivo LICENSE.
