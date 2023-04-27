# IUM

### Definicja problemu biznesowego

Przewidywanie, czy dany użytkownik przewinie dany utwór czy przesłucha go do końca. Będzie to dalej wykorzystywane do odpowiedniego rozłożenia utworów w cache'u. Rozłożenie w cache'u nie jest jednak celem projektu.

### Zdefiniowanie zadania modelowania

Przygotowanie modelu klasyfikacji binarnej na podstawie logu zdarzeń użytkowników (zdarzenia typu PLAY, SKIP, itd.).

### Zdefiniowanie założeń

Wykorzystana zostanie głównie zależność między ulubionymi gatunkami użytkownika a gatunkami przypisanymi do wykonawcy.

### Zaproponowanie kryteriów sukcesu

#### Kryterium analityczne

α = liczba trafnych predykcji / liczba wszystkich predykcji
α >= 0.65

Wartość 65% została oszacowana przez zastosowanie prostego modelu klasyfikacji (random forest) na podstawie danych dla 50 użytkowników.

Procent utworów pominiętych w zbiorze treningowym wynosi 50%, więc technicznie model, który zawsze przewiduje pominięcie utworu, osiągnie 50% trafności. W związku z tym, aby model był użyteczny, musi przewidywać pominięcie utworu lepiej niż model losowy.

#### Kryterium biznesowe

Redukcja opóźnień odtwarzania utworów, w porównaniu do systemu bez predykcji, poprzez optymalizację wykorzystania pamięci podręcznej. Piosenki, które użytkownik prawdopodobnie pominie, nie będą pobierane do cache.

### Analiza danych

Znajduje się w notatnikach `notebooks/data_analysis.ipynb` oraz `notebooks/data_processing.ipynb`.
