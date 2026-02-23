- [Դիֆերենցելիությունից բխող հատկությունները](#դիֆերենցելիությունից-բխող-հատկությունները)
- [Համադրույթի (բարդ ֆունկցիայի) դիֆերենցելիությունը](#համադրույթի-բարդ-ֆունկցիայի-դիֆերենցելիությունը)
- [Շղթայական կանոնի ստացումը](#շղթայական-կանոնի-ստացումը)
- [Դիֆերենցիալի սահմանումը](#դիֆերենցիալի-սահմանումը)


Համաձայն արդեն ապացուցված թեորեմի. Եթե $(x_0, y_0)$ կետում գոյություն ունեն $f'_x(x_0, y_0)$ և $f'_y(x_0, y_0)$ մասնակի ածանցյալները և դրանք անընդհատ են, ապա $f$ ֆունկցիան դիֆերենցելի է այդ կետում։

Նշանակենք՝

$$A = \frac{\partial f}{\partial x}(x_0, y_0)$$

$$B = \frac{\partial f}{\partial y}(x_0, y_0)$$

Այդ դեպքում **ֆունկցիայի աճը** կլինի.

$$\Delta f = \frac{\partial f}{\partial x}(x_0, y_0)\Delta x + \frac{\partial f}{\partial y}(x_0, y_0)\Delta y + o(\sqrt{\Delta x^2 + \Delta y^2})$$

կամ նույնն է՝

$$\Delta f = A\Delta x + B\Delta y + o(\sqrt{\Delta x^2 + \Delta y^2})$$

Նշենք, որ հակառակ պնդումը ճիշտ չէ։ Այսինքն՝ ֆունկցիան կարող է լինել դիֆերենցելի ինչ-որ կետում, սակայն այդ կետում մասնակի ածանցյալները կարող են անընդհատ չլինել։

Օրինակ՝

$$f(x, y) = \begin{cases} x^2 \sin \frac{1}{x} & \text{երբ } x \neq 0 \\ 0 & \text{երբ } (x, y) = (0, 0) \end{cases}$$

Մասնակի ածանցյալը $x \neq 0$ դեպքում.

$$f'_x(x, y) = 2x \sin \frac{1}{x} + x^2 \cos \frac{1}{x} \left(-\frac{1}{x^2}\right) = 2x \sin \frac{1}{x} - \cos \frac{1}{x}$$

Մասնակի ածանցյալը $(0, 0)$ կետում՝ ըստ սահմանման.

$$\frac{\partial f}{\partial x}(0, 0) = \lim_{\Delta x \to 0} \frac{f(0 + \Delta x, 0) - f(0, 0)}{\Delta x} = \lim_{\Delta x \to 0} \frac{\Delta x^2 \sin \frac{1}{\Delta x} - 0}{\Delta x} = \lim_{\Delta x \to 0} \left(\Delta x \sin \frac{1}{\Delta x}\right) = 0$$

(Այստեղ կիրառված է սահմանափակ ֆունկցիայի և անվերջ փոքրի արտադրյալի հատկությունը)։

Հասկանանք, որ ածանցյալն անընդհատ չէ.

$$\lim_{x \to 0} \frac{\partial f}{\partial x} = \lim_{x \to 0} \left(2x \sin \frac{1}{x} - \cos \frac{1}{x}\right) = 2 \lim_{x \to 0} x \sin \frac{1}{x} - \lim_{x \to 0} \cos \frac{1}{x}$$

Քանի որ $\lim_{x \to 0} \cos \frac{1}{x}$ սահմանը գոյություն չունի, ապա ածանցյալը $(0, 0)$ կետում անընդհատ չէ։

**Եզրակացություն**: Որպեսզի ֆունկցիան լինի դիֆերենցելի, մասնակի ածանցյալի անընդհատությունը պարտադիր չէ։

### Դիֆերենցելիությունից բխող հատկությունները

**Թեորեմ**: Եթե $f$ ֆունկցիան դիֆերենցելի է $(x_0, y_0)$ կետում, ապա գոյություն ունեն $\frac{\partial f}{\partial x}(x_0, y_0)$ և $\frac{\partial f}{\partial y}(x_0, y_0)$ մասնակի ածանցյալները, ընդ որում՝ $\frac{\partial f}{\partial x}(x_0, y_0) = A$ և $\frac{\partial f}{\partial y}(x_0, y_0) = B$:

**Ապացույց:**

Դիցուք $f$-ը դիֆերենցելի է $(x_0, y_0)$-ում, այդ դեպքում գոյություն ունեն $A, B \in \mathbb{R}$ այնպիսին, որ.

$$\Delta f(x_0, y_0) = A \Delta x + B \Delta y + o(\sqrt{\Delta x^2 + \Delta y^2}) \quad (*)$$

Դիտարկենք մասնակի ածանցյալի սահմանումը $x$-ի համար.

$$\frac{\partial f}{\partial x}(x_0, y_0) = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x, y_0) - f(x_0, y_0)}{\Delta x} \quad (1)$$

Օգտվելով $(*)$ բանաձևից և տեղադրելով $(1)$-ի մեջ (երբ $\Delta y = 0$), կստանանք.

$$\lim_{\Delta x \to 0} \frac{A \Delta x + B \cdot 0 + o(\sqrt{\Delta x^2 + 0^2})}{\Delta x} = \lim_{\Delta x \to 0} \left( A + \frac{o(|\Delta x|)}{\Delta x} \right) = A \Rightarrow \frac{\partial f}{\partial x} = A$$

Նույն կերպ $y$-ի համար.

$$\frac{\partial f}{\partial y}(x_0, y_0) = \lim_{\Delta y \to 0} \frac{f(x_0, y_0 + \Delta y) - f(x_0, y_0)}{\Delta y} = \lim_{\Delta y \to 0} \frac{A \cdot 0 + B \Delta y + o(|\Delta y|)}{\Delta y} = B \Rightarrow \frac{\partial f}{\partial y} = B$$

**Հատկություն**: Եթե $f$ ֆունկցիան դիֆերենցելի է $(x_0, y_0)$ կետում, ապա այն նաև անընդհատ է այդ կետում։

**Ապացույց:**
Ֆունկցիայի անընդհատությունը նշանակում է, որ $\lim_{\substack{x \to x_0 \\ y \to y_0}} f(x, y) = f(x_0, y_0)$:

Նկատենք, որ $f(x, y) = f(x, y) - f(x_0, y_0) + f(x_0, y_0)$:

Քանի որ $f$-ը դիֆերենցելի է, ապա աճը ($f(x, y) - f(x_0, y_0)$) կարող ենք ներկայացնել այսպես.

$$f(x, y) = A(x - x_0) + B(y - y_0) + o(\sqrt{(x-x_0)^2 + (y-y_0)^2}) + f(x_0, y_0)$$

Անցնելով սահմանի, երբ $x \to x_0$ և $y \to y_0$, տեսնում ենք, որ առաջին երեք գումարելիները ձգտում են $0$-ի.

$$\lim_{\substack{x \to x_0 \\ y \to y_0}} f(x, y) = 0 + 0 + f(x_0, y_0) + 0 = f(x_0, y_0)$$

### Համադրույթի (բարդ ֆունկցիայի) դիֆերենցելիությունը

Դիցուք ունենք.

$f: D \to \mathbb{R}$, որտեղ $D \subset \mathbb{R}^m$

$g: D' \to D$, որտեղ $D' \subset \mathbb{R}^k$

$g(t) = (g^1(t), g^2(t), \dots, g^m(t))$, որտեղ յուրաքանչյուր $g^i$ կախված է $(t^1, \dots, t^k)$ փոփոխականներից։

**Թեորեմ:** Եթե $f$ ֆունկցիան դիֆերենցելի է $x_0 = g(t_0)$ կետում և $g^1, \dots, g^m$ կոորդինատային ֆունկցիաները դիֆերենցելի են $t_0 \in D'$ կետում, ապա բարդ ֆունկցիան՝ $F(t) = f(g(t))$, նույնպես դիֆերենցելի կլինի $t_0$ կետում։

**Ապացույց:**
Քանի որ $f$-ը դիֆերենցելի է $x_0$ կետում.

$$\Delta f = \sum_{i=1}^{m} \frac{\partial f(x_0)}{\partial x^i} \Delta x^i + o(\|\Delta x\|) \quad (1)$$

որտեղ $\Delta x^i = g^i(t) - g^i(t_0)$։

$g^i$ ֆունկցիաների դիֆերենցելիությունից $t_0$ կետում.

$$\Delta g^i = \sum_{j=1}^{k} \frac{\partial g^i(t_0)}{\partial t^j} (t^j - t_0^j) + o(\|t - t_0\|) \quad (2)$$

Դիտարկենք $F$ ֆունկցիայի աճը.

$$\Delta F = f(g(t)) - f(g(t_0)) \quad (3)$$

Տեղադրելով $(1)$-ի մեջ.

$$\Delta F = \sum_{i=1}^{m} \frac{\partial f(x_0)}{\partial x^i} (g^i(t) - g^i(t_0)) + o(\|g(t) - g(t_0)\|) \quad (4)$$

Այժմ $(2)$-ը տեղադրենք $(4)$-ի մեջ.

$$\Delta F = \sum_{i=1}^{m} \frac{\partial f(x_0)}{\partial x^i} \left[ \sum_{j=1}^{k} \frac{\partial g^i(t_0)}{\partial t^j} (t^j - t_0^j) + o(\|t - t_0\|) \right] + o(\|g(t) - g(t_0)\|) \quad (5)$$

### Շղթայական կանոնի ստացումը

Բացելով $(5)$ բանաձևի փակագծերը և խմբավորելով ըստ $(t^j - t_0^j)$ աճերի.

$$\Delta F = \sum_{j=1}^{k} (t^j - t_0^j) \left[ \sum_{i=1}^{m} \frac{\partial f(x_0)}{\partial x^i} \cdot \frac{\partial g^i(t_0)}{\partial t^j} \right] + o(\|t - t_0\|) + o(\|g(t) - g(t_0)\|) \quad (6)$$

Նշանակենք $A_j$-ով $(t^j - t_0^j)$-ի գործակիցը.

$$A_j = \sum_{i=1}^{m} \frac{\partial f}{\partial x^i} \cdot \frac{\partial g^i}{\partial t^j}$$

Այդ դեպքում.

$$\Delta F = \sum_{j=1}^{k} A_j (t^j - t_0^j) + o(\|t - t_0\|) + o(\|g(t) - g(t_0)\|) \quad (7)$$

Այժմ ցույց տանք, որ վերջին մնացորդային անդամը նույնպես հանդիսանում է $o(\|t - t_0\|)$.

$$\frac{\|g(t) - g(t_0)\|^2}{\|t - t_0\|^2} = \sum_{i=1}^{m} \frac{(g^i(t) - g^i(t_0))^2}{\|t - t_0\|^2}$$

Յուրաքանչյուր $g^i$-ի համար.

$$\frac{|g^i(t) - g^i(t_0)|}{\|t - t_0\|} = \frac{|\sum \frac{\partial g^i}{\partial t^j} \Delta t^j + o(\dots)|}{\|t - t_0\|} \leq \sum \left| \frac{\partial g^i}{\partial t^j} \right| + \alpha_i$$

Քանի որ սա սահմանափակ է, ապա $o(\|g(t) - g(t_0)\|) = o(\|t - t_0\|)$։

**Վերջնական տեսքը**.

$$\Delta F = \sum_{j=1}^{k} A_j (t^j - t_0^j) + o(\|t - t_0\|)$$

Սա նշանակում է, որ բարդ ֆունկցիան դիֆերենցելի է։

### Դիֆերենցիալի սահմանումը

Դիցուք $f: D \to \mathbb{R}$, որտեղ $D \subseteq \mathbb{R}^2$ և $f$-ը դիֆերենցելի է $M_0(x_0, y_0) \in D$ կետում.

$$\Delta f = \frac{\partial f}{\partial x}(M_0) \Delta x + \frac{\partial f}{\partial y}(M_0) \Delta y + o(\sqrt{\Delta x^2 + \Delta y^2}) \quad (*)$$

**Սահմանում:** Ֆունկցիայի աճի գլխավոր մասը (այն մասը, որը գծային է $\Delta x$ և $\Delta y$ աճերի նկատմամբ) կոչվում է ֆունկցիայի դիֆերենցիալ։

Այն նշանակվում է $df(M_0)$ կամ $df(x_0, y_0)$ և հավասար է $(*)$ բանաձևի առաջին երկու գումարելիների գումարին.

$$df(x_0, y_0) = \frac{\partial f}{\partial x}(x_0, y_0) \Delta x + \frac{\partial f}{\partial y}(x_0, y_0) \Delta y$$

