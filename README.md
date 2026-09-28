# OCR TEST (PaddleOCR local)

بيئة محلية مستقلة لاختبار **PaddleOCR 3.x** على صور (جواز سفر / رخصة قيادة) باستخدام **CPU** فقط. لا يوجد تكامل مع Diamond ولا parser للحقول في هذه المرحلة.

## المتطلبات

- Windows 10+
- Python 3.12 (مثبت للمستخدم عبر winget إن لم يكن متوفراً)
- اتصال إنترنت **لأول تشغيل** لتحميل نماذج PP-OCRv6 (تُخزَّن في `%USERPROFILE%\.paddlex\official_models\`)

## الإعداد

```powershell
cd "c:\Users\Rw\OCR TEST"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r requirements.txt
```

## هيكل المجلد

```
OCR TEST/
├── .venv/
├── samples/
│   ├── passport/    ← ضع صور الجواز هنا (لا تُرفع إلى Git)
│   └── licence/     ← ضع صور الرخصة هنا
├── output/          ← نتائج JSON (مُستثناة من Git)
├── test_ocr.py
├── requirements.txt
└── README.md
```

## تشغيل الاختبار

```powershell
.\.venv\Scripts\Activate.ps1
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"   # اختياري؛ السكربت يضبطه افتراضياً
python test_ocr.py path\to\image.png
```

مثال بعد وضع صورة في `samples\passport`:

```powershell
python test_ocr.py samples\passport\your_image.png
```

## CLI الرسمي (اختياري)

```powershell
paddleocr ocr -i .\samples\passport\your_image.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False --engine paddle
```

## ملاحظات

- الصور الحساسة مستثناة في `.gitignore`.
- أول تشغيل قد يستغرق وقتاً لتحميل النماذج.
- تحذير `No ccache found` من Paddle آمن ويمكن تجاهله.
