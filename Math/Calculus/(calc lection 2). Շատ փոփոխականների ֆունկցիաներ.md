# Շատ փոփոխականների ֆունկցիաներ
- [Շատ փոփոխականների ֆունկցիաներ](#շատ-փոփոխականների-ֆունկցիաներ)
  - [Կուտակման կետ](#կուտակման-կետ)
  - [Շատ փոփոխականի ֆունկցիաներ](#շատ-փոփոխականի-ֆունկցիաներ)
  - [Թեորեմ: Կոշիի և Հայնեի սահմանումների համարժեքությունը](#թեորեմ-կոշիի-և-հայնեի-սահմանումների-համարժեքությունը)
    - [Ապացույց: Անհրաժեշտություն](#ապացույց-անհրաժեշտություն)
    - [Ապացույց: Բավարարություն](#ապացույց-բավարարություն)
  - [Թեորեմ: Սահմանի գոյության Կոշիի պայմանը](#թեորեմ-սահմանի-գոյության-կոշիի-պայմանը)
    - [Ապացույց: Անհրաժեշտություն](#ապացույց-անհրաժեշտություն-1)
    - [Ապացույց: Բավարարություն](#ապացույց-բավարարություն-1)


## Կուտակման կետ 

**Սահմանում:** Դիցուք $E \subseteq \mathbb{R}^m$։ $x_0 \in \mathbb{R}^m$ կետը կոչվում է $E$ բազմության **կուտակման կետ**, եթե $x_0$-ի ցանկացած շրջակայքում $E$-ի անվերջ թվով կետեր կան առկա։

**Սահմանում:** $x_0$-ն կոչվում է $E$-ի կուտակման կետ, եթե $x_0$-ի կամայական շրջակայքում առկա է $E$-ից գոնե մեկ կետ։

**Լեմմա:** Եթե $x_0$-ն $E$-ի կուտակման կետ է, ապա $\exists \{x_n\} \subset E, x_n \neq x_0$, այնպես որ $x_n \overset{n \to \infty}{\longrightarrow} x_0$։ Ճիշտ է նաև հակառակ պնդումը։

**Ապացույց:** $E$ բազմության կուտակման կետերի բազմությունը նշանակենք $E'$-ով։ Ունենք, որ $x_0 \in E' \implies \forall \varepsilon > 0 \ \exists x_{\varepsilon} \in E \text{ s.t. } x \in B(x_0, \varepsilon)$։

$\varepsilon$-ը նշանակենք $\frac{1}{n} \implies \exists x_n \in E \text{ s.t. } x_n \in B(x_0, \frac{1}{n})$։

- $n := 1 \implies \exists x_1 \in E \text{ s.t. } x_1 \in B(x_0, 1)$
    
- $n := 2 \implies \exists x_2 \in E \text{ s.t. } x_2 \in B(x_0, \frac{1}{2})$
    
    Այսպիսով՝ $(x_1, x_2, \dots, x_n) \to x_0$
    

---

## Շատ փոփոխականի ֆունկցիաներ

**Սահմանում 1:** Դիցուք $E \subseteq \mathbb{R}^m$։ $f: E \to \mathbb{R}$ արտապատկերմանը կանվանենք $m$ փոփոխականի ֆունկցիա։

**Օրինակներ:**

- $z = f(x, y) = x^2 + y^2$
    
- $V(h, r) = \pi r^2 h$
    
- $V(a, b, c) = a \cdot b \cdot c$
    

**Սահմանում 2:** Կասենք, որ $A \in \mathbb{R}$ թիվը հանդիսանում է $f: E \to \mathbb{R}$ ֆունկցիայի սահման $x_0 \in E'$ կետում, եթե $\forall \varepsilon > 0 \ \exists \delta = \delta(\varepsilon) > 0 \text{ s.t. } 0 < \|x - x_0\| < \delta, x \in E \implies |f(x) - A| < \varepsilon$։

**Սահմանում 3:** Դիցուք $x \in E, x = (x^1, x^2, \dots, x^m)$ և $x_0 \in E', x_0 = (x_0^1, x_0^2, \dots, x_0^m)$։

Կասենք, որ $A \in \mathbb{R}$ թիվը հանդիսանում է $f: E \to \mathbb{R}$ ֆունկցիայի սահման $x_0$ կետում, եթե $\forall \varepsilon > 0 \ \exists \delta' = \delta'(\varepsilon) > 0 \text{ s.t. } |x^i - x_0^i| < \delta', x \in E, x \neq x_0 \implies |f(x^1, x^2, \dots, x^m) - A| < \varepsilon$։

**Ցույց տանք, որ (3) $\implies$ (2)**

Նշանակենք $\delta = \delta'$։ Կունենանք, որ՝

$|x^i - x_0^i| \le \|x - x_0\| < \delta'$ $\implies |x^i - x_0^i| < \delta' \implies |f(x) - A| < \varepsilon$։

Այսինքն՝ $\|x - x_0\| < \delta \implies |f(x) - A| < \varepsilon$։

**Հիմա (2) $\implies$ (3)**

Ունենք (2)-ը, այսինքն՝ $\forall \varepsilon > 0 \ \exists \delta = \delta(\varepsilon) > 0 \text{ s.t. } 0 < \|x - x_0\| < \delta \implies |f(x) - A| < \varepsilon$։

Վերցնենք $\delta' := \frac{\delta}{\sqrt{m}}$։ Կունենանք՝

$\|x - x_0\| = \sqrt{(x^1 - x_0^1)^2 + (x^2 - x_0^2)^2 + \dots + (x^m - x_0^m)^2} \le \sqrt{(\frac{\delta}{\sqrt{m}})^2 + (\frac{\delta}{\sqrt{m}})^2 + \dots + (\frac{\delta}{\sqrt{m}})^2} = \sqrt{\frac{\delta^2}{m} \cdot m} = \delta$

$\implies \|x - x_0\| < \delta \implies (3)$

---

## Թեորեմ: Կոշիի և Հայնեի սահմանումների համարժեքությունը

Որպեսզի $\lim_{x \to x_0} f(x) = A$, որտեղ $f: E \to \mathbb{R} \ (E \subset \mathbb{R}^m, x_0 \in E')$, անհրաժեշտ է և բավարար, որ $\forall \{x_n\} \subset E, x_n \neq x_0, x_n \to x_0 \implies f(x_n) \to A$։

### Ապացույց: Անհրաժեշտություն

Ենթադրենք ունենք որ $\forall \varepsilon > 0 \ \exists \delta = \delta(\varepsilon) > 0 \text{ s.t. } 0 < \|x - x_0\| < \delta, x \in E \implies |f(x) - A| < \varepsilon \quad (*)$։

Դիտարկենք $\forall x_n \in E \text{ s.t. } x_n \neq x_0, x_n \to x_0$։

$\forall \varepsilon_0 > 0 \ \exists n_0(\varepsilon_0) \in \mathbb{N} \text{ s.t. } n > n_0(\varepsilon_0) \implies \|x_n - x_0\| < \varepsilon_0$։

Նշանակենք $\varepsilon_0 := \delta = \delta(\varepsilon)$։ Կունենանք, որ $\exists n_0(\varepsilon) \in \mathbb{N} \text{ s.t. } \|x_n - x_0\| < \delta$։

Օգտվելով $(*)$-ից՝ կունենանք, որ $\|x_n - x_0\| < \delta \implies |f(x_n) - A| < \varepsilon \implies f(x_n) \overset{n \to \infty}{\longrightarrow} A$։

### Ապացույց: Բավարարություն

Դիցուք $\forall \{x_n\} \subset E, x_n \neq x_0, x_n \to x_0 \implies f(x_n) \to A$։ Ցույց տանք, որ՝

$\forall \varepsilon > 0 \ \exists \delta = \delta(\varepsilon) > 0 \text{ s.t. } 0 < \|x - x_0\| < \delta, x \in E \implies |f(x) - A| < \varepsilon$։

Կատարենք հակասող ենթադրություն. $\exists \varepsilon > 0, \forall \delta > 0, \exists x_{\delta} \in E \text{ s.t. } 0 < \|x - x_0\| < \delta, x \in E \implies |f(x) - A| \ge \varepsilon$։

$\delta$-ն ընտրելով $\frac{1}{n}$, կունենանք՝

$n = 1 \implies \exists \varepsilon > 0, \exists x_1 \text{ s.t. } 0 < \|x_1 - x_0\| < 1, x_1 \in E \implies |f(x_1) - A| \ge \varepsilon$

$n = 2 \implies \exists \varepsilon > 0, \exists x_2 \text{ s.t. } 0 < \|x_2 - x_0\| < \frac{1}{2}, x_2 \in E \implies |f(x_2) - A| \ge \varepsilon$

Այսպիսով՝ $x_1, x_2, \dots, x_n \to x_0$, բայց $f(x_1), f(x_2), \dots, f(x_n)$ չի ձգտում $A$-ի, ինչը հակասում է թեորեմի պայմանին։

---

## Թեորեմ: Սահմանի գոյության Կոշիի պայմանը

Որպեսզի $\lim_{x \to x_0} f(x) = A$, որտեղ $f: E \to \mathbb{R} \ (E \subset \mathbb{R}^m, x_0 \in E')$, անհրաժեշտ է և բավարար, որ՝

$\forall \varepsilon > 0 \ \exists \delta = \delta(\varepsilon) > 0 \text{ s.t. } 0 < \|x' - x_0\| < \delta, 0 < \|x'' - x_0\| < \delta, x', x'' \in E \implies |f(x') - f(x'')| < \varepsilon \quad (**)$

### Ապացույց: Անհրաժեշտություն

Դիցուք $\lim_{x \to x_0} f(x) = A$: Ցույց տանք, որ $(**)$-ը տեղի ունի այդ դեպքում։

$\lim_{x \to x_0} f(x) = A \implies \forall \varepsilon > 0 \ \exists \delta = \delta(\varepsilon) > 0 \text{ s.t. } 0 < \|x - x_0\| < \delta, x \in E \implies |f(x) - A| < \frac{\varepsilon}{2}$
Այսինքն՝

$0 < \|x' - x_0\| < \delta \implies |f(x') - A| < \frac{\varepsilon}{2}$

$0 < \|x'' - x_0\| < \delta \implies |f(x'') - A| < \frac{\varepsilon}{2}$

Այժմ դիտարկենք հետևյալ արտահայտությունը. $|f(x') - f(x'')| = |f(x') - A + A - f(x'')| \le |f(x') - A| + |f(x'') - A| < \frac{\varepsilon}{2} + \frac{\varepsilon}{2} = \varepsilon$։

$\implies |f(x') - f(x'')| < \varepsilon$։

### Ապացույց: Բավարարություն

Ունենք, որ $(**)$-ը բավարարված է: Ցույց տանք, որ սահմանը գոյություն ունի։

Վերցնենք $\forall \{x_n\} \subset E, x_n \neq x_0, x_n \to x_0$։

Կունենանք, որ $\forall \delta > 0 \ \exists n_0(\delta) \in \mathbb{N} \text{ s.t. } n > n_0(\delta) \implies 0 < \|x_n - x_0\| < \delta$։

$0 < \|x_m - x_0\| < \delta, x_m, x_n \in E \implies |f(x_m) - f(x_n)| < \varepsilon$։

Օգտվելով Կոշիի հաջորդականության սկզբունքից թվային հաջորդականությունների համար, կարող ենք պնդել, որ $f(x_n)$-ը զուգամետ է, քանի որ այն ֆունդամենտալ է։ Այսինքն՝ $\exists A \in \mathbb{R} \text{ s.t. } f(x_n) \overset{n \to \infty}{\longrightarrow} A$։

Այժմ ցույց տանք, որ $A \in \mathbb{R}$ թիվը կախված չէ $\{x_n\}$ հաջորդականության ընտրությունից։ Կատարենք հետևյալ ենթադրությունը՝

$x_n \to x_0 \implies f(x_n) \to A$

$y_n \to x_0 \implies f(y_n) \to B$

$x_n \neq y_n, \forall n, y_n \in E$

Կառուցենք $x_0$-ին ձգտող հետևյալ հաջորդականությունը. $z_n = x_1, y_1, x_2, y_2, \dots, x_n, y_n \to x_0$։

$z_{2n} \to x_0, f(z_{2n}) \to B$

$z_{2n-1} \to x_0, f(z_{2n-1}) \to A$

Քանի որ $f(z_n) \overset{n \to \infty}{\longrightarrow} A'$, ապա $\implies B = A$։