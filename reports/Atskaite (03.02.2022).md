## Atskaite (03.21.2022)

Failu saturs GIT repozitorijā

- nod-7.py atrodas refresijas uzdevums
  - par šo uzdevumu joprojām ir vairākas neskaidrības, galvenokārt problēmas ir tieši ar svaru un bias matricu izmēriem, kas pie atvasinājumu aprēķināšanas neiegūst pareizās dimensijas
  - Piemēram, funkcijā f_dW1_loss gala matricas shape ir (3,5), bet matrica W1 tika definēta kā (1,3), līdz ar to parādās ValueError kļūda

- nod-8.py atrodas OOP regresijas uzdevums
  - implementēta normalizācija un MSE/L2 loss function
  - par ReLu un NRMSE radās jautājumi (norādīti zemāk)




Jautājumi

1. Ja ReLu arguments ir matrica (np.array), tad vai tas nozīmē, ka tā algoritms (>=0 vai <0) tiek atsevišķi izpildīts katram elementam?
1. Vai tas pats attiecas uz atvasinājumu?
1. NRMSE funkcijā kas tieši ir lielumi y (un kāda atvasinājuma un parastajā funkcijā ir atškirība starp tiem y ar diakritiskajām zīmēm)?





