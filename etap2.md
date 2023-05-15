# IUM

Piotr Brzeziński      - 310606
Bartłomiej Rasztabiga - 304117

## Etap 2

## Raport z budowy modeli

### Preprocessing danych

Wykorzystywane są dwie kolumny z logu zdarzeń użytkownika:
- genres
- favourite_genres

Predykowaną klasą jest kolumna skipped.

Aby ograniczyć liczbę klas, zastosowaliśmy klastrowanie metodą k-means. Liczba klastrów została dobrana eksperymentalnie i wynosi 100.

Aby zmniejszyć wpływ klasy większościowej, zastosowaliśmy balansowanie danych w proporcji 1:1.

Kolumny genres i favourite_genres zawierają listy gatunków. Zastosowaliśmy one-hot encoding, aby uzyskać macierz binarną.

### Podział danych

Dane zostały podzielone na zbiór treningowy i testowy w proporcji 80/20.

Oba modele trenowane były na danych v2 (dla 50 użytkowników).

Dane v3 nie zostały użyte z powodu ich zbyt dużego rozmiaru dla dostępnych zasobów.

### Model 1

Definicja pierwszego modelu znajduje się w pliku `notebooks/model.ipynb`

Jest to RandomForest z 100 drzewami.

```
model = RandomForestClassifier(random_state=42)
```

Poniżej wyniki dla zbiorów testowego, treningowego i wszystkich danych.

```
TEST
Accuracy: 0.6155969634230504
Confusion matrix:
 [[438 283]
 [274 454]]
Classification report:
               precision    recall  f1-score   support

           0       0.62      0.61      0.61       721
           1       0.62      0.62      0.62       728

    accuracy                           0.62      1449
   macro avg       0.62      0.62      0.62      1449
weighted avg       0.62      0.62      0.62      1449

TRAIN
Accuracy: 0.8529259451061626
Confusion matrix:
 [[2465  435]
 [ 417 2476]]
Classification report:
               precision    recall  f1-score   support

           0       0.86      0.85      0.85      2900
           1       0.85      0.86      0.85      2893

    accuracy                           0.85      5793
   macro avg       0.85      0.85      0.85      5793
weighted avg       0.85      0.85      0.85      5793

ALL
Accuracy: 0.8054404860535763
Confusion matrix:
 [[2903  718]
 [ 691 2930]]
Classification report:
               precision    recall  f1-score   support

           0       0.81      0.80      0.80      3621
           1       0.80      0.81      0.81      3621

    accuracy                           0.81      7242
   macro avg       0.81      0.81      0.81      7242
weighted avg       0.81      0.81      0.81      7242
```

### Model 2

Definicja drugiego modelu znajduje się w pliku notebooks/model2.ipynb

Jest to wielowarstwowy perceptron z 2 warstwami ukrytymi, po których występuje dropout w celu zapobiegnięcia przetrenowaniu.

Liczby neuronów w warstwach ukrytych zostały dobrane w procesie strojenia i wynoszą 111 i 41.

Funkcje aktywacji warstw ukrytych to `relu`, a warstwy wyjściowej to `sigmoid`.

Learning rate został dobrany w procesie strojenia i wynosi 0.003.

Liczba epok została dobrana w procesie strojenia i wynosi 20.

```
model = Sequential()
model.add(Input(shape=(X_train.shape[1])))
model.add(Dense(units=111, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=41, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.compile(
    optimizer=Adam(learning_rate=0.003),
    loss='binary_crossentropy', metrics=['accuracy'])
```

```
Epoch 1/20
182/182 - 3s - loss: 0.6847 - accuracy: 0.5512 - val_loss: 0.6588 - val_accuracy: 0.5990 - 3s/epoch - 16ms/step
Epoch 2/20
182/182 - 2s - loss: 0.6599 - accuracy: 0.6037 - val_loss: 0.6484 - val_accuracy: 0.6046 - 2s/epoch - 8ms/step
Epoch 3/20
182/182 - 1s - loss: 0.6489 - accuracy: 0.6116 - val_loss: 0.6402 - val_accuracy: 0.6467 - 1s/epoch - 7ms/step
Epoch 4/20
182/182 - 1s - loss: 0.6385 - accuracy: 0.6282 - val_loss: 0.6343 - val_accuracy: 0.6418 - 1s/epoch - 7ms/step
Epoch 5/20
182/182 - 1s - loss: 0.6302 - accuracy: 0.6361 - val_loss: 0.6338 - val_accuracy: 0.6480 - 1s/epoch - 7ms/step
Epoch 6/20
182/182 - 1s - loss: 0.6259 - accuracy: 0.6378 - val_loss: 0.6319 - val_accuracy: 0.6473 - 1s/epoch - 7ms/step
Epoch 7/20
182/182 - 1s - loss: 0.6187 - accuracy: 0.6473 - val_loss: 0.6258 - val_accuracy: 0.6418 - 1s/epoch - 7ms/step
Epoch 8/20
182/182 - 1s - loss: 0.6135 - accuracy: 0.6573 - val_loss: 0.6302 - val_accuracy: 0.6446 - 1s/epoch - 7ms/step
Epoch 9/20
182/182 - 1s - loss: 0.6030 - accuracy: 0.6639 - val_loss: 0.6323 - val_accuracy: 0.6432 - 1s/epoch - 7ms/step
Epoch 10/20
182/182 - 1s - loss: 0.6019 - accuracy: 0.6615 - val_loss: 0.6363 - val_accuracy: 0.6377 - 1s/epoch - 6ms/step
Epoch 11/20
182/182 - 1s - loss: 0.5919 - accuracy: 0.6703 - val_loss: 0.6393 - val_accuracy: 0.6411 - 1s/epoch - 7ms/step
Epoch 12/20
182/182 - 1s - loss: 0.5886 - accuracy: 0.6718 - val_loss: 0.6469 - val_accuracy: 0.6391 - 1s/epoch - 7ms/step
Epoch 13/20
182/182 - 1s - loss: 0.5842 - accuracy: 0.6708 - val_loss: 0.6422 - val_accuracy: 0.6404 - 1s/epoch - 7ms/step
Epoch 14/20
182/182 - 1s - loss: 0.5789 - accuracy: 0.6820 - val_loss: 0.6475 - val_accuracy: 0.6308 - 1s/epoch - 7ms/step
Epoch 15/20
182/182 - 1s - loss: 0.5740 - accuracy: 0.6789 - val_loss: 0.6483 - val_accuracy: 0.6370 - 1s/epoch - 7ms/step
Epoch 16/20
182/182 - 1s - loss: 0.5688 - accuracy: 0.6917 - val_loss: 0.6509 - val_accuracy: 0.6480 - 1s/epoch - 8ms/step
Epoch 17/20
182/182 - 1s - loss: 0.5706 - accuracy: 0.6862 - val_loss: 0.6488 - val_accuracy: 0.6501 - 1s/epoch - 7ms/step
Epoch 18/20
182/182 - 1s - loss: 0.5586 - accuracy: 0.6917 - val_loss: 0.6577 - val_accuracy: 0.6460 - 1s/epoch - 7ms/step
Epoch 19/20
182/182 - 1s - loss: 0.5592 - accuracy: 0.6901 - val_loss: 0.6670 - val_accuracy: 0.6446 - 1s/epoch - 7ms/step
Epoch 20/20
182/182 - 1s - loss: 0.5545 - accuracy: 0.6972 - val_loss: 0.6560 - val_accuracy: 0.6508 - 1s/epoch - 7ms/step
```

Poniżej wyniki dla zbiorów testowego, treningowego i wszystkich danych.

```
TEST
46/46 [==============================] - 0s 3ms/step
Accuracy: 0.6507936507936508
Confusion matrix:
 [[366 347]
 [159 577]]
Classification report:
               precision    recall  f1-score   support

           0       0.70      0.51      0.59       713
           1       0.62      0.78      0.70       736

    accuracy                           0.65      1449
   macro avg       0.66      0.65      0.64      1449
weighted avg       0.66      0.65      0.64      1449

TRAIN
182/182 [==============================] - 0s 2ms/step
Accuracy: 0.719489038494735
Confusion matrix:
 [[1679 1229]
 [ 396 2489]]
Classification report:
               precision    recall  f1-score   support

           0       0.81      0.58      0.67      2908
           1       0.67      0.86      0.75      2885

    accuracy                           0.72      5793
   macro avg       0.74      0.72      0.71      5793
weighted avg       0.74      0.72      0.71      5793

ALL
227/227 [==============================] - 0s 2ms/step
Accuracy: 0.7057442695388014
Confusion matrix:
 [[2045 1576]
 [ 555 3066]]
Classification report:
               precision    recall  f1-score   support

           0       0.79      0.56      0.66      3621
           1       0.66      0.85      0.74      3621

    accuracy                           0.71      7242
   macro avg       0.72      0.71      0.70      7242
weighted avg       0.72      0.71      0.70      7242
```

## Porównanie modeli

Jak można zauważyć, model 2 osiąga lepsze wyniki niż model 1 na danych walidacyjnych.

Model 1 osiąga jednak lepsze wyniki na danych treningowych, co może wskazywać na lepsze dopasowanie się do wzorca w danych.

| Model | Accuracy | Val Accuracy |
| ----- | -------- | ------------ |
| 1     | 0.8529   | 0.6155       |
| 2     | 0.7194   | 0.6507       |

Model drugi spełnił analityczne kryterium sukcesu. Wartość α wynosi 0.6507, czyli jest większa od założonego 0.65.

Dalsze porównanie modeli przeprowadzone zostało podczas testów A/B, opisanych poniżej.

## Strojenie hiperparametrów

Hiperparametry modelu 2 zostały dobrane w procesie strojenia.

Wykorzystaliśmy bibliotekę keras-tuner, która pozwala na automatyczne strojenie hiperparametrów.

Strojenie zostało przeprowadzone dla 4 hiperparametrów. 

- liczba warstw ukrytych
- liczba neuronów w warstwach ukrytych
- dropout (tak/nie)
- learning rate

Wykorzystany został tuner RandomSearch.

Poniżej kod odpowiedzialny za strojenie hiperparametrów.

```
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1])))

    for i in range(hp.Int("num_layers", min_value=1, max_value=3)):
        model.add(Dense(units=hp.Int(f"units_{i}", min_value=1, max_value=200, step=5), activation="relu"))
        if hp.Boolean("dropout"):
            model.add(Dropout(hp.Float("dropout_rate", min_value=0.1, max_value=0.99)))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=0.001, max_value=0.01, sampling="log")),loss='binary_crossentropy', metrics=['accuracy'])

    return model

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=50,
    executions_per_trial=1,
    overwrite=True,
    directory="tuner",
    project_name="IUM",
)

tuner.search_space_summary()
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
tuner.results_summary()
```

Wyniki strojenia:
```
{'dropout': True, 'learning_rate': 0.003, 'num_layers': 2, 'units_0': 111, 'units_1': 41}
```

## Implementacja mikroserwisu

Mikroserwis został zaimplementowany w języku Python z wykorzystaniem frameworka FastAPI.
Wykorzystana baza danych to MongoDB.

### Endpointy
##### W pliku `IUM.postman_collection.json` znajduje się kolekcja Postmana zawierająca wszystkie endpointy, gotowa do przetestowania.

#### `POST /ab_test?user_id={user_id}`
Służy do zbierania danych do testów A/B. Zwraca wynik predykcji dla podanych danych wejściowych (`genres/favourite_genres`).

Parametr w URL: `user_id` - (ID użytkownika na podstawie którego dobierana jest grupa testowa)
Przykładowe body:
```
{
    "genres": [
        "album rock",
        "art rock",
        "classic rock",
        "folk rock",
        "glam rock",
        "protopunk",
        "psychedelic rock",
        "rock"
    ],
    "favourite_genres": [
        "funk",
        "filmi",
        "metal"
    ]
}
```
Przykładowa odpowiedź: `{"skipped": true}`

#### `DELETE /ab_test/results`
Służy do czyszczenia bazy danych przechowującej wyniki testów A/B.

#### `GET /ab_test/results`
Zwraca wszystkie dane o testach A/B znajdujące się w naszej bazie danych.

#### `POST /models/{model_id}/predict`
Zwraca rezultat działania modelu oznaczonego `{model_id}` (1 - RandomForest, 2 - MLP). Dane wejściowe i wyjściowe są w takim samym formacie jak w endpoincie `POST /ab_test`.

Przykładowa odpowiedź: `{"skipped": false}`

### Deployment
Cały mikroserwis został wzdrożony pod adresem https://ium.rasztabiga.me z użyciem klastra Kubernetes.

Do uruchomienia lokalnej instancji należy mieć zainstalowanego Dockera. 
Po przejściu do folderu `microservice/` należy zainstalować wymagane pakiety Pythona poleceniem `pip install -r requirements.txt`. Następnie uruchamiamy bazę danych poleceniem `docker-compose up` oraz sam mikroserwis zbudowany z użyciem poleceniem `./run.sh`.

Aby wdrożyć mikroserwis na własny klaster Kubernetes, należy przejść do folderu `microservice/k8s` oraz uruchomić polecenie `kubectl apply -f . -n {{nazwa_namespacu}}`

## Testy A/B

Proces przeprowadzania testów A/B zawarty został w pliku `ab.ipynb`.

Do testów, ze względu na ograniczoną wydajność po stronie zdeployowanego mikroserwisu (długi czas generowania predykcji), używamy 10% z całego zestawu danych.

Dla każdego wiersza z danymi przygotowujemy i wysyłamy request na endpoint `POST /ab_test?user_id={user_id}`. Na podstawie `user_id` ustalamy grupę do której należy (`group = user_id % 2`) i zapisujemy wynik zapytania, wraz z faktyczną wartością `skipped`, do listy należącej do tej grupy.

Następnie liczymy dokładność dla każdej grupy, po czym na jej podstawie przeprowadzamy test t-Studenta i sprawdzamy, czy odrzucamy hipotezę zerową mówiącą, przez co możemy stwierdzić, czy dokładność modelu A jest większa niż modelu B.

#### Wyniki testów:
```
Group A accuracy: 0.7616438356164383
Group B accuracy: 0.6709470304975923

Wartość t-statystyki: 3.026444516084243
Wartość p-value: 0.001269366320691488
Odrzucamy hipotezę zerową. Dokładność modelu A jest większa niż modelu B.
```
