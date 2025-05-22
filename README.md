# PersianLicensePlateRecognition
Detection of plate location and Recognition the plate characters using 'Yolo' and CRNN for 'OCR'

A deep learning-based system for automatic recognition of Persian (Iranian) license plates, utilizing YOLO for plate detection and OCR for character recognition.
_________________
📌 Overview
This project aims to build a robust pipeline for detecting and recognizing Persian license plates in images or videos. It combines the power of YOLO (You Only Look Once) object detection and Optical Character Recognition (OCR) to extract accurate license plate text.
_________________
🔍 Components
YOLO (v5/v8): For real-time detection of Persian plates in images and video frames.

OCR (e.g. EasyOCR, Tesseract, or custom CNN): For reading the alphanumeric characters from the detected plates.

Preprocessing Pipeline: Handles image normalization, plate cropping, and character segmentation (if required).
_________________
🧠 Technologies Used
* Python 3.10
* YOLOv5 (PyTorch-based)
* CRNN
* OpenCV
* PIL
* NumPy
* Torch
* Ultralytics
_________________
💻 How to use
First run the code below on your CMD:
- python -m venv (your env name)
- (your env name)\Scripts\activate
```python
pip install -r requirements.txt

python Interface.py
```
Enjoy!
_________________
🖼️ Sample Results
#### Test Image
![test_image_2](https://github.com/user-attachments/assets/557936cb-c6a9-43e9-8f1c-3b2b74616afa)

#### Detected Plate
![image](https://github.com/user-attachments/assets/2befb8e6-fa4b-421b-ba18-b6af6ffdf5f7)

#### Cropped Plate
![image](https://github.com/user-attachments/assets/79b46ade-a9c3-45f5-9b87-d84c8063061e)

#### OCR Result
Predicted text: 11W11211 ---> 'W' in my defined dictionary is same as 'ژ' for people with disabilities
_________________
🔠 Persian OCR Challenges
Non-Latin characters and right-to-left (RTL) text direction.
Similar-looking characters and font variations.
Dirty, angled, or obscured plates.
_________________
🏆 Performance
Detection Accuracy: 98% mAP on custom Persian plate dataset.
OCR Accuracy: 90% on validation set.

Training details and evaluation scripts can be found in the model folders
_________________
📦 Dataset
We Used about 6000 generated data and 4000 read data using the repos below:
* https://github.com/RealSinaEslami/IranianLicensePlateGenerator
* https://github.com/mut-deep/IR-LPR
_________________
📜 License
MIT License — see the LICENSE file for details.
_________________
🙌 Acknowledgements
* YOLOv5 by Ultralytics
* CRNN
