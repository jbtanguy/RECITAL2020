RECITAL 2020
============

Les programmes disponibles sont les suivants :
- ./process_ocr.py : ce programme permet de lancer les OCR (avec Kraken et Tesseract).
- ./evaluation/evaluate_ocr_with_language_model.py : celui-ci permet de lancer le calcul des métriques d'estimation avec les modèles de langue
- ./evaluation/language_models/launch_language_model_learnings.py : permet de lancer l'apprentissage des modèles de langue (il utilise JBTools, un package personnel aussi disponible sur ce github)
- ./evaluation/language_models/get_LM_perplexity.py : permet, pour un modèle de langue donné, de calculer la perplexité de celui-ci sur un ensemble de test
