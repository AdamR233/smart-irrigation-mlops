# Wyniki do analizy

Źródło wyników: zapisane outputs w `notebooks/02_baseline_models.ipynb`,
komórki o execution_count 47–53. Nie przeliczano modeli przy eksporcie.
Wartości zapisane w tabelach są zaokrąglone do czterech miejsc dziesiętnych;
nie są eksportem niezaokrąglonych wartości z pamięci kernela.

## 1. Aktualne 5-fold Stratified CV

Kod zapisanych komórek korzysta z `x_train`, czyli 80% danych (8000 obserwacji),
ze wspólnymi foldami: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
StandardScaler i OneHotEncoder są uczone wewnątrz foldów treningowych.
CatBoost obsługuje kategorie natywnie. Target XGBoost jest kodowany wewnątrz foldu.
Odchylenie standardowe obliczono z `ddof=0`.
Starszych wyników CV na pełnym zbiorze nie włączono do tabeli.
Outputs dokumentują zapisany przebieg; nie stanowią niezależnego odtworzenia eksperymentu.

| Model | Accuracy CV mean | Accuracy CV std | Balanced Accuracy CV mean | Balanced Accuracy CV std | Macro F1 CV mean | Macro F1 CV std | Recall Low CV mean | Recall Low CV std | Recall Medium CV mean | Recall Medium CV std | Recall High CV mean | Recall High CV std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CatBoost | 0.9968 | 0.0018 | 0.9941 | 0.0060 | 0.9953 | 0.0035 | 0.9979 | 0.0013 | 0.9957 | 0.0034 | 0.9887 | 0.0151 |
| Decision Tree | 0.9938 | 0.0019 | 0.9780 | 0.0121 | 0.9796 | 0.0044 | 0.9972 | 0.0021 | 0.9928 | 0.0022 | 0.9440 | 0.0358 |
| XGBoost | 0.9951 | 0.0034 | 0.9779 | 0.0210 | 0.9853 | 0.0119 | 0.9979 | 0.0013 | 0.9957 | 0.0022 | 0.9401 | 0.0602 |

## 2. Ablation study

Warianty używają tych samych foldów treningowego zbioru danych i parametrów
modeli. Preprocessing i lista kategorii CatBoost są tworzone ponownie dla
każdego podzbioru cech. Dodatni `drop mean` oznacza pogorszenie względem
wariantu All features; różnice wyliczono przed zaokrągleniem tabel.
Recall High std nie był wyświetlony w zapisanym wyniku tej komórki, więc
nie jest dostępny w tym eksporcie. Nie uzupełniano brakujących wartości.

| Model | Variant | Balanced Accuracy CV mean | Balanced Accuracy CV std | Macro F1 CV mean | Macro F1 CV std | Balanced Accuracy drop mean | Macro F1 drop mean | Recall Low CV mean | Recall Medium CV mean | Recall High CV mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision Tree | All features | 0.9780 | 0.0121 | 0.9796 | 0.0044 | 0.0000 | 0.0000 | 0.9972 | 0.9928 | 0.9440 |
| Decision Tree | Without Soil_Moisture | 0.6531 | 0.0295 | 0.6498 | 0.0238 | 0.3249 | 0.3298 | 0.8269 | 0.7053 | 0.4272 |
| Decision Tree | Without Temperature_C | 0.7924 | 0.0392 | 0.7915 | 0.0282 | 0.1856 | 0.1881 | 0.8932 | 0.8263 | 0.6576 |
| Decision Tree | Without Rainfall_mm | 0.7667 | 0.0131 | 0.7580 | 0.0163 | 0.2113 | 0.2217 | 0.9075 | 0.8349 | 0.5578 |
| Decision Tree | Without Crop_Growth_Stage | 0.6762 | 0.0228 | 0.6705 | 0.0178 | 0.3018 | 0.3091 | 0.7954 | 0.6461 | 0.5871 |
| Decision Tree | Without Mulching_Used | 0.7808 | 0.0150 | 0.7871 | 0.0125 | 0.1972 | 0.1925 | 0.8960 | 0.8036 | 0.6429 |
| Decision Tree | Without all five selected features | 0.3678 | 0.0165 | 0.3668 | 0.0158 | 0.6102 | 0.6128 | 0.5999 | 0.4181 | 0.0853 |
| CatBoost | All features | 0.9941 | 0.0060 | 0.9953 | 0.0035 | 0.0000 | 0.0000 | 0.9979 | 0.9957 | 0.9887 |
| CatBoost | Without Soil_Moisture | 0.6122 | 0.0271 | 0.6493 | 0.0369 | 0.3820 | 0.3460 | 0.9834 | 0.6599 | 0.1932 |
| CatBoost | Without Temperature_C | 0.7293 | 0.0253 | 0.7650 | 0.0241 | 0.2649 | 0.2303 | 0.9650 | 0.7730 | 0.4497 |
| CatBoost | Without Rainfall_mm | 0.7882 | 0.0153 | 0.8411 | 0.0141 | 0.2059 | 0.1542 | 0.9955 | 0.8707 | 0.4983 |
| CatBoost | Without Crop_Growth_Stage | 0.6332 | 0.0184 | 0.6570 | 0.0196 | 0.3609 | 0.3382 | 0.8265 | 0.6378 | 0.4352 |
| CatBoost | Without Mulching_Used | 0.7664 | 0.0296 | 0.7833 | 0.0243 | 0.2277 | 0.2120 | 0.8996 | 0.8010 | 0.5986 |
| CatBoost | Without all five selected features | 0.3752 | 0.0049 | 0.3650 | 0.0056 | 0.6190 | 0.6303 | 0.7962 | 0.3293 | 0.0000 |

## 3. Decision Tree

```yaml
Tree depth: 8
Number of leaves: 64
```

Model interpretacyjny został dopasowany do całego `x_train`, po sklonowaniu
parametrów istniejącego Decision Tree. Te wartości nie opisują każdego drzewa z CV.

## 4. Pochodzenie datasetu i etykiet

Lokalny plik: `data/raw/irrigation_prediction.csv`.
W README i notebookach nie znaleziono linku źródłowego ani kodu generującego target.

Potencjalne źródło odpowiadające nazwie datasetu z notebooka EDA:
[Irrigation Water Requirement Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/miadul/irrigation-water-requirement-prediction-dataset).
Nie potwierdzono tożsamości lokalnego CSV z plikiem z tej strony ani reguł
tworzenia `Irrigation_Need`. Treść opisu autora nie była dostępna w odczycie strony.

[Opis konkursu Kaggle Playground S6E4](https://www.kaggle.com/competitions/playground-series-s6e4/data)
potwierdza generowanie danych KONKURSOWYCH przez model deep learning na podstawie
oryginalnego Irrigation Prediction dataset. Nie jest to dowód, że lokalny
oryginalny CSV powstał w ten sam sposób.

Sposób utworzenia Irrigation_Need: nieustalony. Hipoteza regułowego etykietowania
wymaga potwierdzenia w dokumentacji autora lub kodzie generatora.
