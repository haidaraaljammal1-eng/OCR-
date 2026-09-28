# OCR TEST (PaddleOCR local)

بيئة محلية مستقلة لاختبار **PaddleOCR 3.x** على صور (جواز سفر / رخصة قيادة) باستخدام **CPU** فقط.

## المتطلبات

- Windows 10+
- Python 3.12 (مثبت للمستخدم عبر winget إن لم يكن متوفراً)
- اتصال إنترنت **لأول تشغيل** لتحميل نماذج PP-OCRv6 (تُخزَّن في `%USERPROFILE%\.paddlex\official_models\`)


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
