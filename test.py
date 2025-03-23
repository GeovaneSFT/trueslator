import requests
import numpy as np
import os
import sys
from ultralytics import YOLO
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from io import BytesIO
from PIL import Image
from googletrans import Translator
import asyncio
# Load the object detection model
from utils.predict_bounding_boxes import predict_bounding_boxes
from utils.translate_manga import translate_manga
from utils.manga_ocr_utils import get_text_from_image
from utils.process_contour import process_contour
from utils.write_text_on_image import add_text


app = FastAPI()

EXTENSIONS = {'j': 'jpg', 'p': 'png', 'w': 'webp'}

IMAGE_HOST_URL = [
    "https://i1.nhentai.net/galleries/",
    "https://i2.nhentai.net/galleries/",
    "https://i3.nhentai.net/galleries/"
]

#@app.get("/")
def read_root():
    return {"Hello": "World"}

#@app.get("/g/{id}")
def get_doujin(id: int):
    url = f"https://nhentai.net/api/gallery/{id}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Erro ao acessar a API do nhentai")
    
    data = response.json()
    
    num_pages = data.get("num_pages", 0)
    media_id = data.get("media_id", 0)
    
    if not num_pages or not media_id:
        raise HTTPException(status_code=400, detail="Informações incompletas na resposta da API.")

    language_tags = [
        tag["name"] for tag in data.get("tags", [])
        if tag.get("type") == "language" and tag["name"] != "translated"
    ]

    page_list = data.get("images", {}).get("pages", [])
    if not page_list:
        raise HTTPException(status_code=400, detail="Nenhuma página encontrada para esse doujin.")

    return {
        "id": id,
        "media_id": media_id,
        "num_pages": num_pages,
        "languages": language_tags,
        "page_list": page_list
    }


#@app.get("/g")
def read_images(media_id: int, num_pages: int, pages_list: List[Dict]) -> List[Image.Image]:
    images = []

    for page in range(num_pages):  # Limita a leitura às primeiras 4 páginas
        ext = EXTENSIONS.get(pages_list[page].get('t'), 'jpg')  # Default para jpg caso não encontre
        url = f"{IMAGE_HOST_URL[0]}{media_id}/{page+1}.{ext}"
        
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"Erro ao baixar imagem {page}")
            continue
        
        img = Image.open(response.raw)
        img = img.convert("RGB")
        img.save(f"./pages/image{page}.{ext}")
        images.append(img)

    return images

def process_images_in_directory(directory_path):
    """
    Processa todas as imagens no diretório fornecido, aplicando detecção de objetos,
    extração e tradução de texto, e salva os resultados.
    """
    # Carrega o modelo de detecção de objetos
    best_model_path = "model_creation/runs/detect/train5/"
    object_detection_model = YOLO(os.path.join(best_model_path, "weights/best.pt"))

    # Lista todas as imagens no diretório
    for filename in os.listdir(directory_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):  # Verifica se o arquivo é uma imagem
            image_path = os.path.join(directory_path, filename)
            print(f"Processando a imagem: {filename}")
            
            # Carregar a imagem
            image = np.array(Image.open(image_path))

            # Prever caixas delimitadoras
            results = predict_bounding_boxes(object_detection_model, image_path)

            for result in results:
                # Descompacta as coordenadas e outras informações da detecção
                x1, y1, x2, y2, score, class_id = result
                detected_image = image[int(y1):int(y2), int(x1):int(x2)]
                
                # Converte a imagem detectada para o formato PIL
                im = Image.fromarray(np.uint8(detected_image * 255))
                
                # Extrai o texto da imagem
                text = get_text_from_image(im)
                
                # Processa os contornos da imagem
                detected_image, cont = process_contour(detected_image)
                
                # Traduz o texto extraído
                text_translated = translate_manga(text)
                
                # Adiciona o texto traduzido na imagem detectada
                image_with_text = add_text(detected_image, text_translated, cont)
                
                # Substitui a região da imagem original com a versão modificada
                image[int(y1):int(y2), int(x1):int(x2)] = image_with_text

            # Salva a imagem com as traduções e modificações
            result_image = Image.fromarray(image, 'RGB')
            result_image.save(f"translated_{filename}")

            # Exibe a imagem traduzida
            result_image = Image.open(f"translated_{filename}")
            #result_image.thumbnail((800, 800))
            #result_image.show()
            #print(f"Imagem {filename} processada e salva como 'translated_{filename}'.")

request = get_doujin(id=560273)
print(request)
images = read_images(request["media_id"],request["num_pages"],request["page_list"])
print(images)
process_images_in_directory("./pages")
