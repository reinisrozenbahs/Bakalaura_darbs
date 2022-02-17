### Atskaite (17.02.22)

Visi izstrādātie faili atrodas mapē "17.02"

Failu struktūra:

- nod_8_py - numpy regresijas uzdevums Swish aktivizācijas funkciju un Huber loss funkciju
  - mapē "attēli" atrodas attēls nod_8.png, kurā ir redzami izveidotie matplotlib grafiki

- nod_9.py - pytorch uzdevums ar Swish aktivizācijas funkciju un R2-Score metriku.
  - problēma pie klases HuberLoss funkcijas backward - nav skaidrs, ko tieši šī funkcija atgriež, patlaban tā atgriež HuberLoss funkcijas atvasinājumu

- nod_10.py - numpy klasifikācijas uzdevums ar SoftMax funkciju, Cross-Entropy Loss funkciju un acc aprēķinu. Kods darbojas bez kļūdām
  - mapē "attēli" atrodas attēls nod_10.png, kurā ir redzami izveidotie matplotlib grafiki (pie parametra epoch < 50 jeb epoch % 50 = 0)