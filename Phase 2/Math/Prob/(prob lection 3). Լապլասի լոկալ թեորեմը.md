- [Լապլասի լոկալ թեորեմը](#լապլասի-լոկալ-թեորեմը)
  - [Թեորեմ](#թեորեմ)
  - [Ապացույց](#ապացույց)
  - [Օրինակ](#օրինակ)

## Լապլասի լոկալ թեորեմը

**Դիտարկենք $n$ անկախ փորձ:**
$P_n(m) = C_n^m p^m q^{n-m} = \frac{n!}{m!(n-m)!} p^m q^{n-m}$
$A$-ի $m$ անգամ հանդես գալը:

Ֆակտորիալը մեծ թվերի համար հարմար չէ, դրա համար օգտվում ենք **Ստիրլինգի թեորեմից**:

### Թեորեմ

Եթե $n$ անկախ փորձերում $A$ պատահույթի $p$ հավանականությունը տարբեր է $0$-ից և $1$-ից ($0 < p < 1$), ապա՝


$$\lim_{n \to \infty} \left[ P_n(m) : \frac{1}{\sqrt{2\pi npq}} \cdot e^{-\frac{1}{2} \left( \frac{m-np}{\sqrt{npq}} \right)^2} \right] = 1$$


որտեղ $q = 1 - p$:

$$P_n(m) \approx \frac{1}{\sqrt{2\pi npq}} e^{-\frac{1}{2} \left( \frac{m-np}{\sqrt{npq}} \right)^2}$$

---

### Ապացույց

Թեորեմը ապացուցելու համար օգտվենք Ստիրլինգի հետևյալ բանաձևից.


$$\lim_{x \to \infty} \frac{x!}{x^x e^{-x} \sqrt{2\pi x}} = 1$$

Այսինքն՝
$x! \approx x^x e^{-x} \sqrt{2\pi x} \cdot (1 + \alpha_x) \quad \alpha_x \to 0$ (անվերջ փոքր)

Համաձայն Բեռնուլիի բանաձևի.
$P_n(m) = \frac{n! p^m q^{n-m}}{m!(n-m)!} \quad (*)$

$(*)$-ում բոլոր ֆակտորիալները փոխարինենք (1)-ի իրենց տեսքով.


$$P_n(m) = \frac{n^n e^{-n} \sqrt{2\pi n} \cdot p^m q^{n-m}}{m^m e^{-m} \sqrt{2\pi m} (n-m)^{n-m} e^{-(n-m)} \sqrt{2\pi(n-m)}}$$

$$= \frac{n^n p^m q^{n-m} \sqrt{2\pi n}}{m^m (n-m)^{n-m} \sqrt{2\pi m} \sqrt{2\pi(n-m)}} = \frac{n^n p^m q^{n-m} \sqrt{n}}{m^m (n-m)^{n-m} \sqrt{2\pi m(n-m)}}$$

$$= \left( \frac{np}{m} \right)^m \cdot \left( \frac{nq}{n-m} \right)^{n-m} \cdot \sqrt{\frac{n}{2\pi m(n-m)}} \quad (**)$$

**Նշանակենք՝**


$$H := \left( \frac{np}{m} \right)^m \cdot \left( \frac{nq}{n-m} \right)^{n-m} \implies \frac{1}{H} = \left( \frac{m}{np} \right)^m \cdot \left( \frac{n-m}{nq} \right)^{n-m}$$

Կատարենք հետևյալ նշանակումը.
$t := \frac{m-np}{\sqrt{npq}}$, որտեղից $m = np + t\sqrt{npq}$, իսկ
$n-m = n - (np + t\sqrt{npq}) = n(1-p) - t\sqrt{npq} = nq - t\sqrt{npq}$

$m$-ի և $n-m$-ի $t$-ով արտահայտված արժեքները տեղադրենք $(**)$-ում.


$$\frac{1}{H} = \left( \frac{np + t\sqrt{npq}}{np} \right)^{np + t\sqrt{npq}} \cdot \left( \frac{nq - t\sqrt{npq}}{nq} \right)^{nq - t\sqrt{npq}}$$

$$= \left( 1 + t\sqrt{\frac{q}{np}} \right)^{np + t\sqrt{npq}} \cdot \left( 1 - t\sqrt{\frac{p}{nq}} \right)^{nq - t\sqrt{npq}} \quad (1)$$

Լոգարիթմելով $(1)$-ի երկու կողմերը՝ կունենանք.


$$\ln \frac{1}{H} = \ln \left( 1 + t\sqrt{\frac{q}{np}} \right)^{np + t\sqrt{npq}} + \ln \left( 1 - t\sqrt{\frac{p}{nq}} \right)^{nq - t\sqrt{npq}} = $$

$$= (np + t\sqrt{npq}) \ln \left( 1 + t\sqrt{\frac{q}{np}} \right) + (nq - t\sqrt{npq}) \ln \left( 1 - t\sqrt{\frac{p}{nq}} \right) \quad (2)$$

Օգտվելով $\ln(1+x)$ ֆունկցիայի Թեյլորի շարքից.
$\ln(1+x) = x - \frac{x^2}{2} + \alpha_x \cdot x$
$\ln(1-x) = -x - \frac{x^2}{2} + \beta_x \cdot x$

$(2)$-ում լոգարիթմները փոխարինելով իրենց բացվածքներով՝ կունենանք.


$$-\ln H = (np + t\sqrt{npq}) \cdot \left( t\sqrt{\frac{q}{np}} - \frac{1}{2} t^2 \frac{q}{np} \right) + (nq - t\sqrt{npq}) \cdot \left( -t\sqrt{\frac{p}{nq}} - \frac{1}{2} t^2 \frac{p}{nq} \right) = $$

$$= t\sqrt{npq} - \frac{1}{2} t^2 q + t^2 q - \frac{1}{2} t^3 q \sqrt{\frac{q}{np}} - t\sqrt{npq} - \frac{1}{2} t^2 p + t^2 p + \frac{1}{2} t^3 p \sqrt{\frac{p}{nq}} = $$

$$= \frac{1}{2} t^2 q - \frac{1}{2} t^3 q \sqrt{\frac{q}{np}} + \frac{1}{2} t^2 p + \frac{1}{2} t^3 p \sqrt{\frac{p}{nq}} = $$

$$= \frac{1}{2} t^2 (p+q) - \frac{1}{2} t^3 q \sqrt{\frac{q}{np}} + \frac{1}{2} t^3 p \sqrt{\frac{p}{nq}} = \frac{1}{2} t^2 - \frac{1}{2} t^3 q \sqrt{\frac{q}{np}} + \frac{1}{2} t^3 p \sqrt{\frac{p}{nq}} \quad (3)$$

$m$-ի բոլոր այն արժեքների համար, որտեղ $t = \frac{m-np}{\sqrt{npq}}$ մեծությունը սահմանափակ է, $(3)$-ում անցնելով սահմանի՝ կունենանք.
$\lim_{n \to \infty} \left[ \frac{1}{2} t^2 - \frac{1}{2} t^3 q \sqrt{\frac{q}{np}} + \frac{1}{2} t^3 p \sqrt{\frac{p}{nq}} \right] = \frac{1}{2} t^2$

Ստացվեց որ՝ $-\ln H = \frac{1}{2} t^2 \implies \ln H = -\frac{1}{2} t^2 \implies H = e^{-\frac{1}{2} t^2}$
$t$-ի փոխարեն տեղադրելով.
$H = e^{-\frac{1}{2} \left( \frac{m-np}{\sqrt{npq}} \right)^2}$ (սա $(**)$-ի I կտորն էր)

Այժմ $(**)$-ի II կտորում $m$ և $(n-m)$-ի փոխարեն տեղադրենք իրենց արժեքները $t$-ով արտահայտված.


$$\sqrt{\frac{n}{2\pi m(n-m)}} = \sqrt{\frac{n}{2\pi (np + t\sqrt{npq})(nq - t\sqrt{npq})}} = $$

$$= \sqrt{\frac{n}{2\pi n^2 pq (1 + t\sqrt{\frac{q}{np}})(1 - t\sqrt{\frac{p}{nq}})}} = \frac{1}{\sqrt{2\pi npq}} \cdot \frac{1}{\sqrt{(1+t\sqrt{\frac{q}{np}})(1-t\sqrt{\frac{p}{nq}})}} \to \frac{1}{\sqrt{2\pi npq}} \cdot (1 + \beta_n)$$

Ընդհանրացնելով կունենանք.


$$P_n(m) \approx \frac{1}{\sqrt{2\pi npq}} e^{-\frac{1}{2} \left( \frac{m-np}{\sqrt{npq}} \right)^2}$$

**Լապլասի փոքր ֆունկցիա.**
Կատարենք հետևյալ նշանակումը՝ $\varphi(t) := \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2} t^2}$, որին կանվանենք Լապլասի փոքր ֆունկցիա։
Նկատենք որ $\varphi(-t) = \varphi(t)$, այսինքն՝ զույգ ֆունկցիա է։

Այսպիսով, $t = \frac{m-np}{\sqrt{npq}}$ նշանակումից ստացվում է.


$$P_n(m) \approx \frac{\varphi(t)}{\sqrt{npq}}$$

---

### Օրինակ

$n = 2400$ անկախ փորձ, $m = 1400$
$p = 0.6$
$q = 0.4$
$P_n(m) - ?$
$P_{2400}(1400) = \frac{2400!}{1400! 1000!} (0.6)^{1400} (0.4)^{1000}$

$P_{2400}(1400) = \frac{1}{\sqrt{2400 \cdot 0.6 \cdot 0.4}} \cdot \exp \left( -\frac{1}{2} \left( \frac{1400 - 2400 \cdot 0.6}{\sqrt{2400 \cdot 0.6 \cdot 0.4}} \right)^2 \right)$


Քանի որ $\varphi(t)$-ն զույգ ֆունկցիա է, $\varphi(-1.67) = \varphi(1.67)$
Օգտվելով Լապլասի ֆունկցիայի աղյուսակից.

$\varphi(1.66) \approx 0.1006$

$\varphi(1.67) \approx 0.0989$

$t$-ի հաշվարկը
$$t = \frac{m - np}{\sqrt{npq}} = \frac{1400 - 1440}{24} = -\frac{40}{24} = -\frac{5}{3} \approx -1.666...$$

$t = -\frac{5}{3} \approx -1.6$
Տեղադրենք արժեքները $P_n(m) \approx \frac{\varphi(t)}{\sqrt{npq}}$ բանաձևի մեջ.

$$P_{2400}(1400) \approx \frac{0.0989}{24} \approx 0.00412$$

