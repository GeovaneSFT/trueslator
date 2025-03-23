import requests
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from io import BytesIO
from manga_ocr import MangaOcr
from PIL import Image
from googletrans import Translator
import asyncio
import easyocr
from utils.manga_ocr_utils import get_text_from_image

app = FastAPI()
#mocr = MangaOcr()
reader = easyocr.Reader(['ja'], gpu=False)
EXTENSIONS = {'j': 'jpg', 'p': 'png', 'w': 'webp'}

IMAGE_HOST_URL = [
    "https://i1.nhentai.net/galleries/",
    "https://i2.nhentai.net/galleries/",
    "https://i3.nhentai.net/galleries/"
]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/g/{id}")
async def get_doujin(id: int):
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

    breakpoint()
    images = await read_images(media_id, num_pages, page_list)
    translations = await translate_pages(images, num_pages, page_list)

    return {
        "id": id,
        "media_id": media_id,
        "num_pages": num_pages,
        "languages": language_tags,
        "translations": translations
    }

async def read_images(media_id: int, num_pages: int, pages_list: List[Dict]) -> List[Image.Image]:
    images = []

    for page in range(3, 5):  # Limita a leitura
        ext = EXTENSIONS.get(pages_list[page].get('t'), 'jpg')  # Default para jpg caso não encontre
        url = f"{IMAGE_HOST_URL[0]}{media_id}/{page}.{ext}"
        
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"Erro ao baixar imagem {page}")
            continue
        
        img = Image.open(response.raw)
        img.save(f"./pages/image{page}.{ext}")
        images.append(img)

    return images

async def translate_pages(images: List[Image.Image], pages: int, pages_list: List[Dict]) -> List[str]:
    # Resultado final que armazenará os textos traduzidos
    translated_texts = []

    for i, image in enumerate(images):
        # Acessando os resultados da página atual da lista de páginas
        page_info = pages_list[i]

        # Convertendo a imagem para formato numpy array para a função extract_text_from_regions
        np_image = np.array(image)

        # A função extract_text_from_regions precisa ser ajustada para ser assíncrona se necessário
        image_info = await extract_text_from_regions(np_image, page_info['regions'])

        # Extraindo apenas os textos traduzidos para retornar
        translated_texts.append(image_info['translated_text'])

    return translated_texts

async def extract_text_from_regions(image: np.ndarray, results: list) -> Dict[str, Any]:
    image_info = {"detected_language": "auto", "translated_language": "en", "bounding_boxes": [], "text": [], "translated_text": []}

    for result in results:
        x1, y1, x2, y2, _, _ = result
        detected_image = image[int(y1):int(y2), int(x1):int(x2)]
        if detected_image.shape[-1] == 4:
            detected_image = detected_image[:, :, :3]
        im = Image.fromarray(np.uint8(detected_image * 255))
        text = await get_text_from_image(im)

        processed_image, cont = process_contour(detected_image)
        translated_text = await translate_manga(text, source_lang="auto", target_lang="en")
        add_text(processed_image, translated_text, cont)

        image_info["bounding_boxes"].append(result)
        image_info["text"].append(text)
        image_info["translated_text"].append(translated_text)

    return image_info

