import os
import shutil

import cv2
import numpy as np
from os import makedirs
from os.path import join
from pathlib import Path
from typing import Union
from PIL import Image
from pdf2image import convert_from_path
from pdf_features.PdfFeatures import PdfFeatures

from configuration import IMAGES_ROOT_PATH, XMLS_PATH


class PdfImages:
    def __init__(self, pdf_features: PdfFeatures, pdf_images: list[Image]):
        self.pdf_features: PdfFeatures = pdf_features
        self.pdf_images: list[Image] = pdf_images
        self.save_images()

    def show_images(self, next_image_delay: int = 2):
        for image_index, image in enumerate(self.pdf_images):
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imshow(f"Page: {image_index + 1}", image_np)
            cv2.waitKey(next_image_delay * 1000)
            cv2.destroyAllWindows()

    def save_images(self):
        makedirs(IMAGES_ROOT_PATH, exist_ok=True)
        for image_index, image in enumerate(self.pdf_images):
            image_name = f"{self.pdf_features.file_name}_{image_index}.jpg"
            image.save(join(IMAGES_ROOT_PATH, image_name))

    @staticmethod
    def remove_images():
        shutil.rmtree(IMAGES_ROOT_PATH)

    @staticmethod
    def from_pdf_path(pdf_path: Union[str, Path], pdf_name: str = "", xml_file_name: str = ""):
        xml_path = None if not xml_file_name else Path(XMLS_PATH, xml_file_name + ".xml")

        if xml_path and not xml_path.parent.exists():
            os.makedirs(xml_path.parent, exist_ok=True)

        pdf_features: PdfFeatures = PdfFeatures.from_pdf_path(pdf_path, xml_path)
        
        if pdf_features is None:
            raise ValueError(f"Failed to extract PDF features from {pdf_path}")

        if pdf_name:
            pdf_features.file_name = pdf_name
        else:
            pdf_name = Path(pdf_path).parent.name if Path(pdf_path).name == "document.pdf" else Path(pdf_path).stem
            pdf_features.file_name = pdf_name
        try:
            pdf_images = convert_from_path(pdf_path, dpi=72)
        except Exception as e:
            # Try with explicit poppler path
            poppler_path = r"C:\Users\adwai\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-24.08.0\Library\bin"
            if os.path.exists(poppler_path):
                pdf_images = convert_from_path(pdf_path, dpi=72, poppler_path=poppler_path)
            else:
                raise e
        return PdfImages(pdf_features, pdf_images)
