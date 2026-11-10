from django.shortcuts import render
from PIL import Image
from io import BytesIO
import numpy as np
import base64

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from .forms import ImageUploadForm

# 学習済みモデルの読み込み
model = load_model('prediction/models/vgg16.h5')


def predict(request):
    results = None
    image_url = None
    form = ImageUploadForm()

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            uploaded_file = request.FILES['image']

            # アップロードされた画像を読み込む
            image_bytes = uploaded_file.read()
            image = Image.open(BytesIO(image_bytes)).convert('RGB')

            # VGG16用に画像を加工
            resized_image = image.resize((224, 224))
            image_array = np.array(resized_image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = preprocess_input(image_array)

            # 画像を分類
            predictions = model.predict(image_array)
            results = decode_predictions(predictions, top=5)[0]

            # アップロードされた画像を画面表示用に変換
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            image_url = f"data:{uploaded_file.content_type};base64,{encoded_image}"

    return render(request, 'home.html', {
        'form': form,
        'results': results,
        'image_url': image_url,
    })