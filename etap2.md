# IUM

https://gitlab-stud.elka.pw.edu.pl/jzielins/ium/-/tree/main/
https://gitlab-stud.elka.pw.edu.pl/bkrawcz1/ium-23l

## Etap 2

## Raport z budowy modeli???


### Model 1

RandomForest(100 drzew)

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
```


### Model 2

MLP  111,41 0.003, dropout(0.5)

```Epoch 1/20
182/182 - 3s - loss: 0.6826 - accuracy: 0.5584 - val_loss: 0.6618 - val_accuracy: 0.6046 - 3s/epoch - 18ms/step
Epoch 2/20
182/182 - 1s - loss: 0.6615 - accuracy: 0.5909 - val_loss: 0.6458 - val_accuracy: 0.6563 - 1s/epoch - 8ms/step
Epoch 3/20
182/182 - 1s - loss: 0.6483 - accuracy: 0.6151 - val_loss: 0.6375 - val_accuracy: 0.6536 - 1s/epoch - 7ms/step
Epoch 4/20
182/182 - 1s - loss: 0.6379 - accuracy: 0.6263 - val_loss: 0.6362 - val_accuracy: 0.6598 - 1s/epoch - 7ms/step
Epoch 5/20
182/182 - 1s - loss: 0.6290 - accuracy: 0.6377 - val_loss: 0.6358 - val_accuracy: 0.6577 - 1s/epoch - 6ms/step
Epoch 6/20
182/182 - 1s - loss: 0.6250 - accuracy: 0.6425 - val_loss: 0.6289 - val_accuracy: 0.6591 - 1s/epoch - 7ms/step
Epoch 7/20
182/182 - 1s - loss: 0.6193 - accuracy: 0.6466 - val_loss: 0.6316 - val_accuracy: 0.6549 - 1s/epoch - 7ms/step
Epoch 8/20
182/182 - 1s - loss: 0.6125 - accuracy: 0.6605 - val_loss: 0.6273 - val_accuracy: 0.6508 - 1s/epoch - 7ms/step
Epoch 9/20
182/182 - 1s - loss: 0.6014 - accuracy: 0.6649 - val_loss: 0.6317 - val_accuracy: 0.6529 - 1s/epoch - 8ms/step
Epoch 10/20
182/182 - 1s - loss: 0.5948 - accuracy: 0.6727 - val_loss: 0.6277 - val_accuracy: 0.6556 - 1s/epoch - 8ms/step
Epoch 11/20
182/182 - 1s - loss: 0.5913 - accuracy: 0.6755 - val_loss: 0.6250 - val_accuracy: 0.6618 - 1s/epoch - 7ms/step
Epoch 12/20
182/182 - 1s - loss: 0.5816 - accuracy: 0.6805 - val_loss: 0.6368 - val_accuracy: 0.6322 - 1s/epoch - 7ms/step
Epoch 13/20
182/182 - 1s - loss: 0.5810 - accuracy: 0.6779 - val_loss: 0.6328 - val_accuracy: 0.6522 - 1s/epoch - 7ms/step
Epoch 14/20
182/182 - 1s - loss: 0.5749 - accuracy: 0.6853 - val_loss: 0.6349 - val_accuracy: 0.6411 - 1s/epoch - 7ms/step
Epoch 15/20
182/182 - 1s - loss: 0.5718 - accuracy: 0.6900 - val_loss: 0.6333 - val_accuracy: 0.6501 - 1s/epoch - 7ms/step
Epoch 16/20
182/182 - 1s - loss: 0.5553 - accuracy: 0.7007 - val_loss: 0.6409 - val_accuracy: 0.6515 - 1s/epoch - 7ms/step
Epoch 17/20
182/182 - 1s - loss: 0.5568 - accuracy: 0.7007 - val_loss: 0.6346 - val_accuracy: 0.6536 - 1s/epoch - 6ms/step
Epoch 18/20
182/182 - 1s - loss: 0.5582 - accuracy: 0.6983 - val_loss: 0.6397 - val_accuracy: 0.6370 - 1s/epoch - 6ms/step
Epoch 19/20
182/182 - 1s - loss: 0.5497 - accuracy: 0.7024 - val_loss: 0.6463 - val_accuracy: 0.6363 - 1s/epoch - 7ms/step
Epoch 20/20
182/182 - 1s - loss: 0.5499 - accuracy: 0.6965 - val_loss: 0.6509 - val_accuracy: 0.6404 - 1s/epoch - 7ms/step
```

```
TEST
46/46 [==============================] - 0s 3ms/step
Accuracy: 0.6404416839199448
Confusion matrix:
 [[414 320]
 [201 514]]
Classification report:
               precision    recall  f1-score   support

           0       0.67      0.56      0.61       734
           1       0.62      0.72      0.66       715

    accuracy                           0.64      1449
   macro avg       0.64      0.64      0.64      1449
weighted avg       0.65      0.64      0.64      1449
```

## Porównanie modeli

???

## Implementacja mikroserwisu

// TODO opis, jakie endpointy, po co, infra, itd

## Testy A/B

