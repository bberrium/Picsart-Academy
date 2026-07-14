- [[#Naive Bayes, LDA և QDA|Naive Bayes, LDA և QDA]]
	- [[#Naive Bayes, LDA և QDA#1. Naive Bayes|1. Naive Bayes]]
	- [[#Naive Bayes, LDA և QDA#2. LDA (Linear Discriminant Analysis - Գծային Դիսկրիմինանտ Վերլուծություն)|2. LDA (Linear Discriminant Analysis - Գծային Դիսկրիմինանտ Վերլուծություն)]]
	- [[#Naive Bayes, LDA և QDA#3. QDA (Quadratic Discriminant Analysis - Քառակուսային Դիսկրիմինանտ Վերլուծություն)|3. QDA (Quadratic Discriminant Analysis - Քառակուսային Դիսկրիմինանտ Վերլուծություն)]]

# Naive Bayes, LDA և QDA 

Դասակարգման այս երեք ալգորիթմները հիմնված են հավանականությունների տեսության վրա (Probabilistic Classifiers): Նրանց հիմնական տարբերությունը կայանում է նրանում, թե տվյալների (data) և հատկանիշների (features) վերաբերյալ ինչ **ենթադրություններ (assumptions)** են նրանք անում։

## 1. Naive Bayes 

Առաջին օրինակը վերաբերում է Spam Detection-ին (սպամի հայտնաբերմանը), որտեղ որպես feature-ներ հանդես են գալիս տեքստի բառերը։

### Տեսական Ենթադրությունները

Ալգորիթմը կոչվում է «պարզամիտ» (naive), քանի որ այն անում է շատ խիստ և իրականության մեջ հազվադեպ հանդիպող ենթադրություն.

1. **Feature-ների անկախություն.** Ենթադրվում է, որ կլասը ֆիքսելու դեպքում (օրինակ՝ գիտենք, որ նամակը սպամ է), բոլոր feature-ները (բառերը կամ չափումները) **բացարձակապես անկախ են միմյանցից**: Եթե կառուցենք կովարիացիոն մատրիցը (Covariance Matrix), ապա գլխավոր անկյունագծից դուրս բոլոր արժեքները պետք է լինեն զրո։
    
2. **Բաշխման տեսակը.** Ալգորիթմը պահանջում է, որ իմանանք յուրաքանչյուր կլասի հավանականության խտության ֆունկցիան (PDF - Probability Density Function)։
    
    - Պարտադիր չէ, որ բոլոր կլասները գան _նույն_ բաշխումից, բայց գործնականում (օրինակ՝ Sklearn-ում) հեշտության համար օգտագործվում են հստակ ընտանիքներ՝ `GaussianNB` (նորմալ բաշխում անընդհատ թվերի համար), `MultinomialNB` (տեքստերի համար) կամ `BernoulliNB` (բուլյան/երկու արժեքով տվյալների համար)։
        

### Կոդի Իմպլեմենտացիա (Ըստ քննարկված տրամաբանության)

Գեներացվել են 1000 և 2000 կետերից բաղկացած տվյալներ և կիրառվել է Gaussian Naive Bayes:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 1. Տվյալների գեներացիա Գաուսյան (Նորմալ) բաշխումից
# Կլաս 0: 1000 կետ
X_class0 = np.random.normal(loc=0.0, scale=1.0, size=(1000, 2))
y_class0 = np.zeros(1000)

# Կլաս 1: 2000 կետ (այլ պարամետրերով)
X_class1 = np.random.normal(loc=3.0, scale=1.5, size=(2000, 2))
y_class1 = np.ones(2000)

# Միացնում ենք իրար
X = np.vstack((X_class0, X_class1))
y = np.concatenate((y_class0, y_class1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Մոդելի կառուցում և ուսուցում
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# 3. Գնահատում
nb_pred = nb_model.predict(X_test)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_pred):.4f}")
```

## 2. LDA (Linear Discriminant Analysis - Գծային Դիսկրիմինանտ Վերլուծություն)

Ներկայացված քննարկման ամենակարևոր հատվածներից մեկը Naive Bayes-ից LDA անցումն է:

### Տեսական Ենթադրությունները

1. **Անկախության պայմանի հեռացում.** LDA-ը ավելի մոտ է իրականությանը։ Այն _թույլատրում է_, որ feature-ները լինեն իրարից կախված (օրինակ՝ շան քաշը և հասակը կարող են կոռելացված լինել)։ Այսինքն, կովարիացիոն մատրիցի անկյունագծից դուրս արժեքները այլևս պարտադիր չէ, որ զրո լինեն։
    
2. **Ընդհանուր Կովարիացիոն Մատրից ($\Sigma$).** Չնայած ֆիչերները կարող են կախված լինել, LDA-ն անում է մեկ այլ խիստ ենթադրություն. **բոլոր կլասներն ունեն նույն կովարիացիոն մատրիցը** ($\Sigma_1 = \Sigma_2 = \Sigma$):
    
    - _Բացատրություն տեքստից._ «Ինչքանով որ կախված է [թաթի մեծությունը ականջի մեծությունից] շների դեպքում, այդ նույն չափ էլ կախված է կատուների դեպքում»:
        

### Կոդի Իմպլեմենտացիա


```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Ենթադրենք գեներացրել ենք Multivariate Normal բաշխումով տվյալներ,
# որտեղ կովարիացիոն մատրիցը (cov_matrix) ՆՈՒՅՆՆ Է երկու կլասի համար:

lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)
lda_pred = lda_model.predict(X_test)

print(f"LDA Accuracy: {accuracy_score(y_test, lda_pred):.4f}")
```

_Նշում վիդեոյից/աուդիոյից:_ Եթե տվյալների մեջ ֆիչերները իրոք կախված են միմյանցից, LDA-ը շատ ավելի բարձր ճշտություն (performance) ցույց կտա, քան Naive Bayes-ը, քանի որ Naive Bayes-ը անտեսում է այդ կախվածությունը:

## 3. QDA (Quadratic Discriminant Analysis - Քառակուսային Դիսկրիմինանտ Վերլուծություն)

Երբեմն տվյալներն այնքան բարդ են լինում, որ նույնիսկ LDA-ի «նույն կովարիացիոն մատրիցի» ենթադրությունն է սխալ դուրս գալիս:

### Տեսական Ենթադրությունները

1. **Տարբերվող Կովարիացիոն Մատրիցներ.** QDA-ը հանում է LDA-ի գլխավոր սահմանափակումը։ Այն թույլ է տալիս, որ **յուրաքանչյուր կլաս ունենա իր անհատական կովարիացիոն մատրիցը** ($\Sigma_1 \neq \Sigma_2$):
    
    - _Օրինակ._ Շների մոտ թաթի և ականջի չափերի կախվածությունը կարող է բոլորովին այլ լինել, քան կատուների մոտ։
        
2. Արդյունքում, եթե LDA-ը երկու կլասները բաժանող սահմանը գծում է **ուղիղ գծով** (linear decision boundary), ապա QDA-ը գծում է **կորով/քառակուսային ֆունկցիայով** (quadratic decision boundary)։
    

### Կոդի Իմպլեմենտացիա


```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Այստեղ տվյալները գեներացվում են ՏԱՐԲԵՐ կովարիացիոն մատրիցներով

qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train, y_train)
qda_pred = qda_model.predict(X_test)

print(f"QDA Accuracy: {accuracy_score(y_test, qda_pred):.4f}")
```

Եթե տվյալների կովարիացիոն մատրիցները իրականում տարբեր են, QDA-ը ցույց կտա ամենաբարձր արդյունքը։ Եթե կովարիացիոն մատրիցները նույնն են, QDA-ի մաթեմատիկան ավտոմատ կերպով կրճատվում և վերածվում է LDA-ի պահվածքի։

**Ամփոփում.** 
-  **Naive Bayes:** Ֆիչերներն անկախ են (Ամենապարզ և կոշտ ենթադրությունը)։
- **LDA:** Ֆիչերները կախված են, բայց կախվածության բնույթը (կովարիացիան) նույնն է բոլոր կլասների համար։
- **QDA:** Ֆիչերները կախված են, և յուրաքանչյուր կլաս ունի կախվածության իր յուրահատուկ բնույթը (Ամենաճկունը)։