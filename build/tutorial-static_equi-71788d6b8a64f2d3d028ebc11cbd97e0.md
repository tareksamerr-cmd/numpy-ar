---
short_title: الاتزان الساكن
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# تحديد الاتزان الساكن (Static Equilibrium) في NumPy

عند تحليل الهياكل الفيزيائية، من الأهمية بمكان فهم الميكانيكا التي تحافظ على استقرارها. القوى المطبقة (Applied forces) على أرضية، أو عارضة (beam)، أو أي هيكل آخر، تخلق قوى رد فعل (reaction forces) وعزوم (moments). هذه التفاعلات هي مقاومة الهيكل للحركة دون أن ينكسر. في الحالات التي لا تتحرك فيها الهياكل على الرغم من وجود قوى مطبقة عليها، ينص [قانون نيوتن الثاني](https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion#Newton%27s_second_law) على أن كلاً من التسارع (acceleration) ومجموع القوى في جميع الاتجاهات في النظام يجب أن يكون صفرًا. يمكنك تمثيل وحل هذا المفهوم باستخدام مصفوفات NumPy (NumPy arrays).

## ما ستفعله:
- في هذا الدرس التعليمي (tutorial)، ستستخدم NumPy لإنشاء المتجهات (vectors) والعزوم (moments) باستخدام NumPy arrays.
- حل المشكلات التي تتضمن الكابلات والأرضيات التي تدعم الهياكل.
- كتابة مصفوفات NumPy (NumPy matrices) لعزل المجاهيل (unknowns).
- استخدام دوال NumPy (NumPy functions) لإجراء عمليات الجبر الخطي (linear algebra operations).

## ما ستتعلمه:
- كيفية تمثيل النقاط (points)، و vectors، و moments باستخدام NumPy.
- كيفية إيجاد [العمودي على المتجهات (normal of vectors)](https://en.wikipedia.org/wiki/Normal_(geometry)).
- استخدام NumPy لحساب عمليات المصفوفات (matrix calculations).

## ما ستحتاجه:
- NumPy
- [Matplotlib](https://matplotlib.org/)

يتم استيرادها بالأوامر التالية:

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
```

في هذا tutorial، ستستخدم أدوات NumPy التالية:

* [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) : تحدد هذه الدالة مقياس حجم المتجه (vector magnitude).
* [`np.cross`](https://numpy.org/doc/stable/reference/generated/numpy.cross.html) : تأخذ هذه الدالة مصفوفتين وتنتج حاصل الضرب الاتجاهي (cross product).

+++

## حل الاتزان باستخدام قانون نيوتن الثاني

يتكون نموذجك من عارضة (beam) تحت مجموع من القوى والعزوم. يمكنك البدء في تحليل هذا النظام بقانون نيوتن الثاني:

$$\sum{\text{force}} = \text{mass} \times \text{acceleration}.$$

لتبسيط الأمثلة التي تم النظر فيها، افترض أنها ساكنة (static)، مع acceleration $=0$. نظرًا لوجود نظامنا في ثلاثة أبعاد، ضع في اعتبارك القوى المطبقة في كل من هذه الأبعاد. هذا يعني أنه يمكنك تمثيل هذه القوى كـ vectors. تصل إلى نفس النتيجة لـ [moments](https://en.wikipedia.org/wiki/Moment_(physics))، والتي تنتج عن تطبيق القوى على مسافة معينة بعيدًا عن مركز كتلة الجسم (center of mass).

افترض أن القوة $F$ ممثلة كـ vector ثلاثي الأبعاد (three-dimensional vector)

$$F = (F_x, F_y, F_z)$$

حيث يمثل كل مكون من المكونات الثلاثة حجم القوة المطبقة في كل اتجاه مقابل. افترض أيضًا أن كل مكون في الـ vector

$$r = (r_x, r_y, r_z)$$

هو المسافة بين النقطة التي يتم فيها تطبيق كل مكون من مكونات القوة ومركز النظام (centroid of the system). ثم يمكن حساب الـ moment بواسطة

$$r \times F = (r_x, r_y, r_z) \times (F_x, F_y, F_z).$$

ابدأ ببعض الأمثلة البسيطة لـ vectors القوة

```{code-cell}
forceA = np.array([1, 0, 0])
forceB = np.array([0, 1, 0])
print("Force A =", forceA)
print("Force B =", forceB)
```

هذا يحدد `forceA` كـ vector بحجم 1 في الاتجاه $x$ و `forceB` بحجم 1 في الاتجاه $y$.

قد يكون من المفيد تصور هذه القوى لفهم أفضل لكيفية تفاعلها مع بعضها البعض.
Matplotlib هي مكتبة (library) تحتوي على أدوات تصور (visualization tools) يمكن استخدامها لهذا الغرض.
سيتم استخدام مخططات السهم (Quiver plots) لإظهار [vectors ثلاثية الأبعاد](https://matplotlib.org/gallery/mplot3d/quiver3d.html)، ولكن يمكن أيضًا استخدام الـ library لـ [عروض ثنائية الأبعاد](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiver.html).

```{code-cell}
fig = plt.figure()

d3 = fig.add_subplot(projection="3d")

d3.set_xlim(-1, 1)
d3.set_ylim(-1, 1)
d3.set_zlim(-1, 1)

x, y, z = np.array([0, 0, 0])  # تحديد نقطة التطبيق. اجعلها الأصل

u, v, w = forceA  # تقسيم vector القوة إلى مكونات فردية
d3.quiver(x, y, z, u, v, w, color="r", label="forceA")

u, v, w = forceB
d3.quiver(x, y, z, u, v, w, color="b", label="forceB")

plt.legend()
plt.show()
```

هناك قوتان تنبعثان من نقطة واحدة. لتبسيط هذه المشكلة، يمكنك جمعهما معًا لإيجاد مجموع القوى. لاحظ أن كلاً من `forceA` و `forceB` هما three-dimensional vectors، ممثلتان بواسطة NumPy كـ arrays بثلاثة مكونات. نظرًا لأن NumPy تهدف إلى تبسيط وتحسين العمليات بين vectors، يمكنك بسهولة حساب مجموع هذين الـ vectors على النحو التالي:

```{code-cell}
forceC = forceA + forceB
print("Force C =", forceC)
```

تعمل `forceC` الآن كقوة واحدة تمثل كلاً من A و B.
يمكنك رسمها لرؤية النتيجة.

```{code-cell}
fig = plt.figure()

d3 = fig.add_subplot(projection="3d")

d3.set_xlim(-1, 1)
d3.set_ylim(-1, 1)
d3.set_zlim(-1, 1)

x, y, z = np.array([0, 0, 0])

u, v, w = forceA
d3.quiver(x, y, z, u, v, w, color="r", label="forceA")
u, v, w = forceB
d3.quiver(x, y, z, u, v, w, color="b", label="forceB")
u, v, w = forceC
d3.quiver(x, y, z, u, v, w, color="g", label="forceC")

plt.legend()
plt.show()
```

ومع ذلك، الهدف هو equilibrium.
هذا يعني أنك تريد أن يكون مجموع القوى لديك $(0, 0, 0)$ وإلا فإن جسمك سيتعرض لـ acceleration.
لذلك، يجب أن تكون هناك قوة أخرى تعادل القوى السابقة.

يمكنك كتابة هذه المشكلة كـ $A+B+R=0$، حيث $R$ هي reaction force التي تحل المشكلة.

في هذا المثال، سيعني هذا:

$$(1, 0, 0) + (0, 1, 0) + (R_x, R_y, R_z) = (0, 0, 0)$$

مقسمة إلى مكونات $x$ و $y$ و $z$، يعطيك هذا:

$$\begin{cases}
1+0+R_x=0\\
0+1+R_y=0\\
0+0+R_z=0
\end{cases}$$

حل $R_x$ و $R_y$ و $R_z$ يعطيك vector $R$ من $(-1, -1, 0)$.

إذا تم رسمها، يجب أن يتم إلغاء القوى التي شوهدت في الأمثلة السابقة.
فقط إذا لم تكن هناك قوة متبقية يعتبر النظام في equilibrium.

```{code-cell}
R = np.array([-1, -1, 0])

fig = plt.figure()

d3.set_xlim(-1, 1)
d3.set_ylim(-1, 1)
d3.set_zlim(-1, 1)

d3 = fig.add_subplot(projection="3d")

x, y, z = np.array([0, 0, 0])

u, v, w = forceA + forceB + R  # اجمعها كلها معًا لمجموع القوى
d3.quiver(x, y, z, u, v, w)

plt.show()
```

الرسم البياني الفارغ يدل على عدم وجود قوى خارجية. هذا يشير إلى نظام في equilibrium.


## حل الاتزان كمجموع من العزوم (sum of moments)

بعد ذلك، دعنا ننتقل إلى تطبيق أكثر تعقيدًا.
عندما لا يتم تطبيق جميع القوى في نفس النقطة، يتم إنشاء moments.

على غرار القوى، يجب أن يكون مجموع هذه moments صفرًا، وإلا سيحدث تسارع دوراني (rotational acceleration). على غرار مجموع القوى، يؤدي هذا إلى إنشاء معادلة خطية (linear equation) لكل اتجاه من الاتجاهات الإحداثية الثلاثة في الفضاء.

مثال بسيط على ذلك سيكون من قوة مطبقة على عمود ثابت مثبت في الأرض.
العمود لا يتحرك، لذلك يجب أن يطبق reaction force.
العمود أيضًا لا يدور، لذلك يجب أن يخلق أيضًا reaction moment.
حل كل من reaction force و moments.

لنفترض أن قوة 5N مطبقة بشكل عمودي (perpendicularly) على بعد 2 متر فوق قاعدة العمود.

```{code-cell}
f = 5  # Force in newtons
L = 2  # Length of the pole

R = 0 - f
M = 0 - f * L
print("Reaction force =", R)
print("Reaction moment =", M)
```

## إيجاد القيم ذات الخصائص الفيزيائية (physical properties)

لنفترض أنه بدلاً من قوة تعمل بشكل perpendicularly على الـ beam، تم تطبيق قوة على عمودنا من خلال سلك كان متصلاً بالأرض أيضًا.
نظرًا للشد (tension) في هذا السلك، كل ما تحتاجه لحل هذه المشكلة هو المواقع الفيزيائية لهذه الكائنات.

![Image representing the problem](_static/static_eqbm-fig01.png)

استجابة للقوى المؤثرة على العمود، ولدت القاعدة reaction forces في اتجاهي x و y، بالإضافة إلى reaction moment.

حدد قاعدة العمود على أنها الأصل (origin).
الآن، لنفترض أن السلك متصل بالأرض على بعد 3 أمتار في الاتجاه x ومتصل بالعمود على بعد 2 متر لأعلى، في الاتجاه z.

حدد هذه النقاط في الفضاء كـ NumPy arrays، ثم استخدم هذه الـ arrays لإيجاد vectors الاتجاهية (directional vectors).

```{code-cell}
poleBase = np.array([0, 0, 0])
cordBase = np.array([3, 0, 0])
cordConnection = np.array([0, 0, 2])

poleDirection = cordConnection - poleBase
print("Pole direction =", poleDirection)
cordDirection = cordBase - cordConnection
print("Cord direction =", cordDirection)
```

لاستخدام هذه الـ vectors فيما يتعلق بالقوى، تحتاج إلى تحويلها إلى vectors وحدة (unit vectors).
`unit vectors` لها حجم واحد، وتنقل فقط اتجاه القوى.

```{code-cell}
cordUnit = cordDirection / np.linalg.norm(cordDirection)
print("Cord unit vector =", cordUnit)
```

يمكنك بعد ذلك ضرب هذا الاتجاه بحجم القوة لإيجاد vector القوة.

لنفترض أن السلك لديه tension قدره 5N:

```{code-cell}
cordTension = 5
forceCord = cordUnit * cordTension
print("Force from the cord =", forceCord)
```

لإيجاد الـ moment، تحتاج إلى حاصل الضرب الاتجاهي (cross product) لـ vector القوة والـ radius.

```{code-cell}
momentCord = np.cross(forceCord, poleDirection)
print("Moment from the cord =", momentCord)
```

الآن كل ما عليك فعله هو إيجاد reaction force و moment.

```{code-cell}
equilibrium = np.array([0, 0, 0])
R = equilibrium - forceCord
M = equilibrium - momentCord
print("Reaction force =", R)
print("Reaction moment =", M)
```

### مثال آخر
دعنا نلقي نظرة على نموذج أكثر تعقيدًا قليلاً. في هذا المثال، ستلاحظ عارضة (beam) بها كابلان وقوة مطبقة. هذه المرة تحتاج إلى إيجاد كل من tension في الأسلاك و reaction forces للـ beam. *(المصدر: [Vector Mechanics for Engineers: Statics and Dynamics](https://www.mheducation.com/highered/product/Vector-Mechanics-for-Engineers-Statics-and-Dynamics-Beer.html)، المشكلة 4.106)*


![image.png](_static/problem4.png)

حدد المسافة *a* بـ 3 أمتار


كما كان من قبل، ابدأ بتحديد موقع كل نقطة ذات صلة كـ array.

```{code-cell}
A = np.array([0, 0, 0])
B = np.array([0, 3, 0])
C = np.array([0, 6, 0])
D = np.array([1.5, 0, -3])
E = np.array([1.5, 0, 3])
F = np.array([-3, 0, 2])
```

من هذه المعادلات، تبدأ بتحديد اتجاهات الـ vector باستخدام unit vectors.

```{code-cell}
AB = B - C
AC = C - A
BD = D - B
BE = E - B
CF = F - C

UnitBD = BD / np.linalg.norm(BD)
UnitBE = BE / np.linalg.norm(BE)
UnitCF = CF / np.linalg.norm(CF)

RadBD = np.cross(AB, UnitBD)
RadBE = np.cross(AB, UnitBE)
RadCF = np.cross(AC, UnitCF)
```

يتيح لك هذا تمثيل tension (T) و reaction (R) forces المؤثرة على النظام كـ

$$\left[
\begin{array}
~1/3 & 1/3 & 1 & 0 & 0\\
-2/3 & -2/3 & 0 & 1 & 0\\
-2/3 & 2/3 & 0 & 0 & 1\\
\end{array}
\right]
\left[
\begin{array}
~T_{BD}\\
T_{BE}\\
R_{x}\\
R_{y}\\
R_{z}\\
\end{array}
\right]
=
\left[
\begin{array}
~195\\
390\\
-130\\
\end{array}
\right]$$

والـ moments كـ

$$\left[
\begin{array}
~2 & -2\\
1 & 1\\
\end{array}
\right]
\left[
\begin{array}
~T_{BD}\\
T_{BE}\\
\end{array}
\right]
=
\left[
\begin{array}
~780\\
1170\\
\end{array}
\right]$$

حيث $T$ هو tension في السلك المعني و $R$ هو reaction force في الاتجاه المعني. ثم لديك ست معادلات فقط:


$\sum F_{x} = 0 = T_{BE}/3+T_{BD}/3-195+R_{x}$

$\sum F_{y} = 0 = (-\frac{2}{3})T_{BE}-\frac{2}{3}T_{BD}-390+R_{y}$

$\sum F_{z} = 0 = (-\frac{2}{3})T_{BE}+\frac{2}{3}T_{BD}+130+R_{z}$

$\sum M_{x} = 0 = 780+2T_{BE}-2T_{BD}$

$\sum M_{z} = 0 = 1170-T_{BE}-T_{BD}$


لديك الآن خمسة unknowns بخمس معادلات، ويمكنك حلها لـ:

$\ T_{BD} = 780N$

$\ T_{BE} = 390N$

$\ R_{x} = -195N$

$\ R_{y} = 1170N$

$\ R_{z} = 130N$

+++

## الخلاصة (Wrapping up)

لقد تعلمت كيفية استخدام arrays لتمثيل points، و forces، و moments في الفضاء ثلاثي الأبعاد (three dimensional space). يمكن استخدام كل إدخال في array لتمثيل physical property مقسمة إلى مكونات اتجاهية (directional components). يمكن بعد ذلك معالجتها بسهولة باستخدام NumPy functions.

### تطبيقات إضافية (Additional Applications)

يمكن تطبيق هذه العملية نفسها على المشكلات الحركية (kinetic problems) أو في أي عدد من الأبعاد. افترضت الأمثلة التي تم إجراؤها في هذا tutorial مشكلات ثلاثية الأبعاد في static equilibrium. يمكن استخدام هذه الطرق بسهولة في مشكلات أكثر تنوعًا. تتطلب الأبعاد الأكثر أو الأقل arrays أكبر أو أصغر للتمثيل. في الأنظمة التي تشهد acceleration، يمكن تمثيل السرعة (velocity) و acceleration بالمثل كـ vectors أيضًا.

### المراجع (References)

1. [Vector Mechanics for Engineers: Statics and Dynamics (Beer & Johnston & Mazurek & et al.)](https://www.mheducation.com/highered/product/Vector-Mechanics-for-Engineers-Statics-and-Dynamics-Beer.html)
2. [NumPy Reference](https://numpy.org/doc/stable/reference/)
