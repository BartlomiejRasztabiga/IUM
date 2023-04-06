# IUM

### Definicja problemu biznesowego

Przewidywanie, czy dany użytkownik przewinie dany utwór czy przesłucha go do końca. Będzie to dalej wykorzystywane do odpowiedniego rozłożenia utworów w cache'u. Rozłożenie w cache'u nie jest jednak celem projektu.

### Zdefiniowanie zadania modelowania

Przygotowanie modelu klasyfikacji binarnej na podstawie logu zdarzeń użytkowników (zdarzenia typu PLAY, SKIP, itd.).

### Zdefiniowanie założeń

Wykorzystana zostanie głównie zależność między ulubionymi gatunkami użytkownika a gatunkami przypisanymi do wykonawcy.

### Zaproponowanie kryteriów sukcesu

α = liczba trafnych predykcji / liczba wszystkich predykcji
α >= 0.65

### Analiza danych

Znajduje się w notatnikach `data_analysis.ipynb` oraz `data_processin.ipynb`.