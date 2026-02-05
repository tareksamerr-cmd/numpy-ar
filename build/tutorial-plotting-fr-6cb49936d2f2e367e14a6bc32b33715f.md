---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# رسم الكسوريات (Plotting Fractals)

+++

![صورة كسورية](tutorial-plotting-fractals/fractal.png)

+++

الكسوريات (Fractals) هي أشكال رياضية جميلة وجذابة يمكن إنشاؤها غالباً من مجموعة بسيطة نسبياً من التعليمات. في الطبيعة، يمكن العثور عليها في أماكن مختلفة، مثل الخطوط الساحلية، والأصداف البحرية، والسرخس، وحتى أنها استُخدمت في إنشاء أنواع معينة من الهوائيات. كانت الفكرة الرياضية للكسوريات معروفة منذ فترة طويلة، ولكن بدأ تقديرها حقاً في السبعينيات مع تقدم رسومات الكمبيوتر وبعض الاكتشافات العرضية التي قادت باحثين مثل [بينوا ماندلبروت](https://en.wikipedia.org/wiki/Benoit_Mandelbrot) إلى العثور على التصورات المذهلة حقاً التي تمتلكها الكسوريات.

اليوم سنتعلم كيفية رسم هذه التصورات الجميلة وسنبدأ في الاستكشاف بأنفسنا بينما نكتسب دراية بالرياضيات وراء الكسوريات وسنستخدم دوال NumPy العامة القوية دائماً لإجراء الحسابات اللازمة بكفاءة.

+++

## ما ستفعله

- كتابة دالة (Function) لرسم مجموعات جوليا (Julia sets) المتنوعة
- إنشاء تصور لمجموعة ماندلبروت (Mandelbrot set)
- كتابة دالة تحسب كسوريات نيوتن (Newton fractals)
- التجربة مع تنويعات من أنواع الكسوريات العامة

+++

## ما ستتعلمه

- حدس أفضل لكيفية عمل الكسوريات رياضياً
- فهم أساسي حول دوال NumPy العامة (Universal Functions - ufuncs) والفهرسة البولينية (Boolean Indexing)
- أساسيات العمل مع الأعداد المركبة (Complex Numbers) في NumPy
- كيفية إنشاء تصورات كسورية فريدة خاصة بك

+++

## ما ستحتاجه

- [Matplotlib](https://matplotlib.org/)
- دالة `make_axis_locatable` من واجهة برمجة تطبيقات (API) `mpl_toolkits`

والتي يمكن استيرادها كما يلي:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
```

- بعض الدراية بلغة Python و NumPy و matplotlib
- فكرة عن الدوال الرياضية الأولية، مثل [الأسس](https://en.wikipedia.org/wiki/Exponential_function)، و [الجيب](https://en.wikipedia.org/wiki/Sine) (sin)، و [كثيرات الحدود](https://en.wikipedia.org/wiki/Polynomial) إلخ
- سيكون من المفيد وجود فهم أساسي جداً لـ [الأعداد المركبة](https://en.wikipedia.org/wiki/Complex_number)
- قد تكون المعرفة بـ [المشتقات](https://en.wikipedia.org/wiki/Derivative) (Derivatives) مفيدة

+++

## إحماء (Warmup)

لاكتساب بعض الحدس حول ماهية الكسوريات، سنبدأ بمثال.

تأمل المعادلة التالية:

$f(z) = z^2 -1 $

حيث `z` هو عدد مركب (أي من الشكل $a + bi$)

لراحتنا، سنكتب دالة Python لها:

```{code-cell} ipython3
def f(z):
    return np.square(z) - 1
```

لاحظ أن دالة التربيع التي استخدمناها هي مثال على **[دالة NumPy عامة](https://numpy.org/doc/stable/reference/ufuncs.html)** (NumPy Universal Function)؛ سنعود إلى أهمية هذا القرار قريباً.

لاكتساب بعض الحدس حول سلوك الدالة، يمكننا محاولة إدخال بعض القيم المختلفة.

بالنسبة لـ $z = 0$ ، نتوقع الحصول على $-1$:

```{code-cell} ipython3
f(0)
```

بما أننا استخدمنا دالة عامة في تصميمنا، يمكننا حساب مدخلات (Inputs) متعددة في نفس الوقت:

```{code-cell} ipython3
z = [4, 1-0.2j, 1.6]
f(z)
```

بعض القيم تكبر، وبعضها يصغر، وبعضها لا يطرأ عليه تغيير كبير.

لرؤية سلوك الدالة على نطاق أوسع، يمكننا تطبيق الدالة على مجموعة فرعية من المستوى المركب (Complex Plane) ورسم النتيجة. لإنشاء مجموعتنا الفرعية (أو الشبكة - Mesh)، يمكننا الاستفادة من دالة [**meshgrid**](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html).

```{code-cell} ipython3
x, y = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
mesh = x + (1j * y)  # إنشاء شبكة من المستوى المركب
```

الآن سنطبق دالتنا على كل قيمة موجودة في الـ Mesh. بما أننا استخدمنا دالة عامة في تصميمنا، فهذا يعني أنه يمكننا تمرير الشبكة بالكامل دفعة واحدة. هذا مريح للغاية لسببين: فهو يقلل من كمية الكود (Code) المطلوب كتابتها ويزيد الكفاءة بشكل كبير (حيث تستخدم الدوال العامة برمجة لغة C على مستوى النظام في حساباتها).

هنا نقوم برسم القيمة المطلقة (Absolute Value) (أو المقياس - Modulus) لكل عنصر في الشبكة بعد "تكرار" (Iteration) واحد للدالة باستخدام [**مخطط تشتت ثلاثي الأبعاد**](https://matplotlib.org/stable/users/explain/toolkits/mplot3d.html#scatter-plots) (3D Scatterplot):

```{code-cell} ipython3
output = np.abs(f(mesh))  # أخذ القيمة المطلقة للمخرج (لأغراض الرسم)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter(x, y, output, alpha=0.2)

ax.set_xlabel('Real axis')
ax.set_ylabel('Imaginary axis')
ax.set_zlabel('Absolute value')
ax.set_title('One Iteration: $ f(z) = z^2 - 1$');
```

يعطينا هذا فكرة تقريبية عما يفعله تكرار واحد للدالة. تظل مناطق معينة (لاسيما المناطق الأقرب إلى $(0,0i)$) صغيرة نوعاً ما بينما تنمو مناطق أخرى بشكل كبير جداً. لاحظ أننا نفقد معلومات حول المخرج بأخذ القيمة المطلقة، لكنها الطريقة الوحيدة التي تمكننا من إنشاء رسم بياني.

دعونا نرى ما يحدث عندما نطبق تكرارين على الـ Mesh:

```{code-cell} ipython3
output = np.abs(f(f(mesh)))

ax = plt.axes(projection='3d')

ax.scatter(x, y, output, alpha=0.2)

ax.set_xlabel('Real axis')
ax.set_ylabel('Imaginary axis')
ax.set_zlabel('Absolute value')
ax.set_title('Two Iterations: $ f(z) = z^2 - 1$');
```

مرة أخرى، نرى أن القيم حول نقطة الأصل تظل صغيرة، والقيم ذات القيمة المطلقة الأكبر "تنفجر".

من الانطباع الأول، يبدو سلوكها طبيعياً، وقد يبدو عادياً. تميل الكسوريات إلى امتلاك ما هو أكثر مما تراه العين؛ يظهر السلوك الغريب عندما نبدأ في تطبيق المزيد من الـ Iterations.

+++

تأمل ثلاثة أعداد مركبة:

$z_1 = 0.4 + 0.4i $,

$z_2 = z_1 + 0.1$,

$z_3 = z_1 + 0.1i$

بالنظر إلى شكل أول رسمين بيانيين، نتوقع أن تظل هذه القيم قريبة من نقطة الأصل مع تطبيق التكرارات عليها. دعونا نرى ما يحدث عندما نطبق 10 تكرارات على كل قيمة:

```{code-cell} ipython3
selected_values = np.array([0.4 + 0.4j, 0.41 + 0.4j, 0.4 + 0.41j])
num_iter = 9

outputs = np.zeros((num_iter+1, selected_values.shape[0]), dtype=complex)
outputs[0] = selected_values

for i in range(num_iter):
    outputs[i+1] = f(outputs[i])  # تطبيق 10 تكرارات، وحفظ كل مخرج
```

لدهشتنا، لم يقترب سلوك الدالة من مطابقة فرضيتنا. هذا مثال صارخ على السلوك الفوضوي (Chaotic Behaviour) الذي تمتلكه الكسوريات. في أول رسمين، "انفجرت" القيمة في التكرار الأخير، قافزة بعيداً عن المنطقة التي كانت محتواة فيها سابقاً. من ناحية أخرى، ظل الرسم الثالث محصوراً في منطقة صغيرة قريبة من نقطة الأصل، مما أدى إلى سلوك مختلف تماماً رغم التغيير الضئيل في القيمة.

يقودنا هذا إلى سؤال مهم للغاية: **كم عدد التكرارات التي يمكن تطبيقها على كل قيمة قبل أن تتباعد ("تنفجر")؟**

كما رأينا من أول رسمين، كلما كانت القيم أبعد عن نقطة الأصل، انفجرت بشكل أسرع عموماً. على الرغم من أن السلوك غير مؤكد للقيم الأصغر (مثل $z_1, z_2, z_3$)، يمكننا افتراض أنه إذا تجاوزت القيمة مسافة معينة من نقطة الأصل (لنقل 2) فإنها محكوم عليها بالتباعد (Diverge). سنسمي هذا الحد بـ **نصف القطر** (Radius).

يسمح لنا هذا بتحديد سلوك الدالة لقيمة معينة دون الحاجة إلى إجراء العديد من الحسابات. بمجرد تجاوز نصف القطر، يُسمح لنا بالتوقف عن التكرار، مما يعطينا طريقة للإجابة على السؤال الذي طرحناه. إذا قمنا بحساب عدد الحسابات التي طُبقت قبل التباعد، فإننا نكتسب رؤية حول سلوك الدالة سيكون من الصعب تتبعها بخلاف ذلك.

بالطبع، يمكننا القيام بما هو أفضل بكثير وتصميم دالة تنفذ الإجراء على Mesh كاملة.

```{code-cell} ipython3
def divergence_rate(mesh, num_iter=10, radius=2):

    z = mesh.copy()
    diverge_len = np.zeros(mesh.shape)  # الاحتفاظ بسجل لعدد التكرارات

    # التكرار على العنصر إذا وفقط إذا كان |العنصر| < نصف القطر (وإلا افترض التباعد)
    for i in range(num_iter):
        conv_mask = np.abs(z) < radius
        diverge_len[conv_mask] += 1
        z[conv_mask] = f(z[conv_mask])

    return diverge_len
```

قد يبدو سلوك هذه الدالة مربكاً للوهلة الأولى، لذا سيساعد شرح بعض التدوينات.

هدفنا هو التكرار على كل قيمة في الشبكة وحساب عدد التكرارات قبل أن تتباعد القيمة. بما أن بعض القيم ستتباعد بشكل أسرع من غيرها، فنحن بحاجة إلى إجراء يكرر فقط على القيم التي لها قيمة مطلقة صغيرة بما يكفي. نريد أيضاً التوقف عن حساب القيم بمجرد تجاوزها لنصف القطر. لهذا، يمكننا استخدام **[الفهرسة البولينية](https://numpy.org/devdocs/reference/arrays.indexing.html#boolean-array-indexing)** (Boolean Indexing)، وهي ميزة في NumPy تكون لا تُهزم عند دمجها مع الدوال العامة. تسمح الفهرسة البولينية بإجراء العمليات بشكل مشروط على مصفوفة NumPy دون الحاجة إلى اللجوء إلى التكرار (Looping) والتحقق من كل قيمة في المصفوفة بشكل فردي.

في حالتنا، نستخدم حلقة (Loop) لتطبيق التكرارات على دالتنا $f(z) = z^2 -1 $ ونحتفظ بالسجل. باستخدام Boolean Indexing، نطبق التكرارات فقط على القيم التي لها قيمة مطلقة أقل من 2.

مع توضيح ذلك، يمكننا البدء في رسم أول كسرية لنا! سنستخدم دالة [**imshow**](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html) لإنشاء تصور مرمّز بالألوان للسجلات.

```{code-cell} ipython3
x, y = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
mesh = x + (1j * y)

output = divergence_rate(mesh)

fig = plt.figure(figsize=(5, 5))
ax = plt.axes()

ax.set_title('$f(z) = z^2 -1$')
ax.set_xlabel('Real axis')
ax.set_ylabel('Imaginary axis')

im = ax.imshow(output, extent=[-2, 2, -2, 2])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax, label='Number of iterations');
```

(محتوى محذوف للاختصار...)

ماذا يحدث إذا قمنا بتركيب دالتنا المعرفة داخل دالة الجيب (Sine)؟

دعونا نحاول تعريف

$g(z) = sin(f(z)) = sin(tan(z^2))$

```{code-cell} ipython3
def g(z):
    return np.sin(f(z))
```

```{code-cell} ipython3
output = general_julia(mesh, f=g, num_iter=15, radius=2.1)
kwargs = {'title': 'g(z) = sin(tan(z^2))', 'cmap': 'plasma_r'}

plot_fractal(output, **kwargs);
```

بعد ذلك، لننشئ دالة تطبق كلاً من f و g على المدخلات في كل تكرار وتجمع النتيجة معاً:

$h(z) = f(z) + g(z) = tan(z^2) + sin(tan(z^2))$

```{code-cell} ipython3
def h(z):
    return f(z) + g(z)
```

```{code-cell} ipython3
output = general_julia(small_mesh, f=h, num_iter=10, radius=2.1)
kwargs = {'title': 'h(z) = tan(z^2) + sin(tan(z^2))', 'figsize': (7, 7), 'extent': [-1, 1, -1, 1], 'cmap': 'jet'}

plot_fractal(output, **kwargs);
```

يمكنك حتى إنشاء كسوريات جميلة من خلال أخطائك الخاصة. إليك واحدة تم إنشاؤها بالصدفة عن طريق ارتكاب خطأ في حساب مشتق (Derivative) لكسرية نيوتن:

```{code-cell} ipython3
def accident(z):
    return z - (2 * np.power(np.tan(z), 2) / (np.sin(z) * np.cos(z)))
```

```{code-cell} ipython3
output = general_julia(mesh, f=accident, num_iter=15, c=0, radius=np.pi)
kwargs = {'title': 'Accidental \\ fractal', 'cmap': 'Blues'}

plot_fractal(output, **kwargs);
```

وغني عن القول، هناك إمداد لا نهائي تقريباً من الإبداعات الكسورية المثيرة للاهتمام التي يمكن صنعها بمجرد اللعب بمجموعات مختلفة من دوال NumPy العامة والعبث بالمعلمات (Parameters).

+++

## في الختام (In conclusion)

تعلمنا الكثير عن توليد الكسوريات اليوم. رأينا كيف يمكن حساب الكسوريات المعقدة التي تتطلب العديد من التكرارات بكفاءة باستخدام الدوال العامة. استفدنا أيضاً من Boolean Indexing، مما سمح بإجراء حسابات أقل دون الحاجة إلى التحقق من كل قيمة بشكل فردي. أخيراً، تعلمنا الكثير عن الكسوريات نفسها. كملخص:

- يتم إنشاء الصور الكسورية عن طريق تكرار دالة على مجموعة من القيم، والاحتفاظ بسجل للمدة التي تستغرقها كل قيمة لتجاوز حد معين
- تتوافق الألوان في الصورة مع عدد السجلات للقيم
- تتكون مجموعة جوليا المملوءة (Filled-in Julia set) لـ $c$ من جميع الأعداد المركبة `z` التي تتقارب فيها $f(z) = z^2 + c$
- مجموعة جوليا لـ $c$ هي مجموعة الأعداد المركبة التي تشكل حدود مجموعة جوليا المملوءة
- مجموعة ماندلبروت هي جميع القيم $c$ التي تتقارب فيها $f(z) = z^2 + c$ عند 0
- تستخدم كسوريات نيوتن دوالاً من الشكل $f(z) = z - \frac{p(z)}{p'(z)}$
- يمكن أن تختلف الصور الكسورية مع ضبط عدد التكرارات، ونصف قطر التقارب، وحجم الشبكة، والألوان، واختيار الدالة واختيار المعلمات

+++

## بمفردك (On your own)

- العب بمعلمات دالة مجموعة جوليا المعممة، جرب اللعب بالقيمة الثابتة، وعدد التكرارات، واختيار الدالة، ونصف القطر، واختيار اللون.

- قم بزيارة صفحة ويكيبيديا "List of fractals by Hausdorff dimension" (الرابط موجود في قسم القراءة الإضافية) وحاول كتابة دالة لكسرية لم تُذكر في هذا الدليل التعليمي.

+++

## قراءة إضافية (Further reading)

- [مزيد من المعلومات حول النظرية وراء الكسوريات](https://en.wikipedia.org/wiki/Fractal)
- [قراءة إضافية حول مجموعات جوليا](https://en.wikipedia.org/wiki/Julia_set)
- [مزيد من التفاصيل حول مجموعة ماندلبروت](https://en.wikipedia.org/wiki/Mandelbrot_set)
- [معالجة أكثر اكتمالاً لكسوريات نيوتن](https://en.wikipedia.org/wiki/Newton_fractal)
- [قائمة بالكسوريات المختلفة](https://en.wikipedia.org/wiki/List_of_fractals_by_Hausdorff_dimension)
