# Project Name: dog-cat-classifier

## Repo Structure 

dog-cat-classifier/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .env.example
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│
├── src/
│   └── dog_cat_classifier/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   └── preprocessing.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   └── cnn.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── train.py
│       │   └── evaluate.py
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   └── predict.py
│       │
│       └── utils/
│           ├── __init__.py
│           └── seed.py
│
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_api.py
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
└── scripts/
    ├── train.py
    └── evaluate.py