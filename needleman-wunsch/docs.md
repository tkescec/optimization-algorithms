# Algoritam Needleman-Wunsch

Algoritam Needleman-Wunsch je dinamički programerski algoritam koji se koristi za globalno poravnanje sekvenci, najčešće u bioinformatici za poravnanje DNK, RNK i proteinskih sekvenci. Algoritam osigurava optimalno poravnanje između dvije sekvence na način da maksimizira sličnost i minimizira kazne za neslaganja i praznine (gapove).

---

## Koraci algoritma

### 1️⃣ Inicijalizacija matrice sličnosti
- Kreira se matrica veličine \((m+1) \times (n+1)\), gdje su \(m\) i \(n\) duljine dviju sekvenci koje se poravnavaju.
- Prvi redak i prvi stupac se popunjavaju prema penalizaciji za umetanje praznina.

### 2️⃣ Popunjavanje matrice
Svaka ćelija matrice se popunjava pomoću rekurzivne formule:

\[
F(i,j) = \max
\begin{cases} 
F(i-1, j-1) + S(i,j) & \text{(ako se znakovi podudaraju ili postoji zamjena)} \\
F(i-1, j) + d & \text{(ako postoji umetanje praznine)} \\
F(i, j-1) + d & \text{(ako postoji brisanje)}
\end{cases}
\]

- \(S(i,j)\) predstavlja skor zamjene između dva znaka.
- \(d\) je kazna za umetanje praznina.

### 3️⃣ Pronalaženje optimalnog poravnanja
- Kreće se od donjeg desnog kuta matrice i prati se put unazad prema gornjem lijevom kutu kako bi se rekonstruiralo optimalno poravnanje.

---

## Primjeri upotrebe algoritma Needleman-Wunsch

### 🧬 Bioinformatika
- Poravnanje DNK sekvenci kako bi se analizirala njihova sličnost.
- Poravnanje proteinskih sekvenci kako bi se razumjela evolucijska povezanost između vrsta.

### 📝 Obrada prirodnog jezika (NLP)
- Prepoznavanje sličnih riječi ili rečenica (npr. korekcija pravopisa).

### 🔍 Računalna forenzika
- Usporedba tekstualnih dokumenata kako bi se otkrila plagijarizacija.

---

## Hiperparametri u algoritmu i kako mogu uzrokovati nekonvergenciju

Hiperparametri su unaprijed postavljene vrijednosti koje određuju ponašanje algoritma. U Needleman-Wunsch algoritmu, ključni hiperparametri su:

### 🔹 Kazna za umetanje praznina (\(d\))
- **Preniska kazna**: Algoritam će radije uvoditi praznine nego vršiti zamjene.
- **Previsoka kazna**: Algoritam će forsirati zamjene čak i kada to nije optimalno.

### 🔹 Matrica zamjena (\(S(i,j)\))
- Ako su težine neadekvatno podešene, može dovesti do poravnanja koja ne odražavaju stvarnu sličnost između sekvenci.

### 🔹 Granice veličine sekvenci
- Ako sekvence nisu usporedivih duljina, algoritam može dati nekonzistentne rezultate.

---

## 🔍 Što znači da algoritam konvergira ili ne konvergira?
- **Konvergencija** znači da algoritam proizvodi stabilno i optimalno poravnanje koje se ne mijenja značajno uz male promjene hiperparametara.
- **Nekonvergencija** se događa kada algoritam ne uspije pronaći konzistentno rješenje ili kad mala promjena hiperparametara uzrokuje drastične promjene u izlazu.

---

## Greedy vs. Exploration pristup

### Greedy (pohlepni) pristup
- Pohlepni algoritam donosi odluke temeljene na lokalno najboljem izboru u svakom koraku, bez razmatranja dugoročne optimalnosti.
- **Primjer u Needleman-Wunsch algoritmu**:
  - Ako bi algoritam uvijek birao trenutno najbolji par znakova bez evaluacije budućih poravnanja, bio bi pohlepan.
  - **Needleman-Wunsch nije pohlepan algoritam** jer koristi dinamičko programiranje i razmatra globalno optimalno rješenje.

### Exploration (istraživački) pristup
- Istraživački pristup podrazumijeva ispitivanje više mogućnosti kako bi se pronašlo najbolje rješenje.
- **Needleman-Wunsch koristi istraživački pristup** jer ispunjava cijelu matricu prije nego što odlučuje o konačnom poravnanju.

---

## 📌 Ukratko:
✅ **Greedy** – donosi odluke odmah bez provjere dugoročnih posljedica.  
✅ **Exploration** – istražuje sve mogućnosti i osigurava globalno optimalno rješenje.
