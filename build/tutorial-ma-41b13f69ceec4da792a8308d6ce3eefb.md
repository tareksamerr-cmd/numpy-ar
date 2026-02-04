---
jupytext:
  formats: ipynb,md:myst
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

# المصفوفات المقنعة (Masked Arrays)

## ماذا ستفعل (What you'll do)

استخدم وحدة المصفوفات المقنعة (masked arrays) من NumPy لتحليل بيانات COVID-19 والتعامل مع القيم المفقودة (missing values).

## ماذا ستتعلم (What you'll learn)

- ستفهم ما هي masked arrays وكيف يمكن إنشاؤها.
- سترى كيفية الوصول إلى البيانات وتعديلها لـ masked arrays.
- ستكون قادرًا على تحديد متى يكون استخدام masked arrays مناسبًا في بعض تطبيقاتك.

## ماذا ستحتاج (What you'll need)

- معرفة أساسية بلغة Python. إذا كنت ترغب في تحديث ذاكرتك، ألقِ نظرة على [البرنامج التعليمي لـ Python](https://docs.python.org/dev/tutorial/index.html).
- معرفة أساسية بـ NumPy.
- لتشغيل الرسوم البيانية (plots) على جهاز الكمبيوتر الخاص بك، تحتاج إلى [matplotlib](https://matplotlib.org).

+++

***

+++

## ما هي masked arrays؟

فكر في المشكلة التالية. لديك مجموعة بيانات (dataset) تحتوي على إدخالات مفقودة (missing) أو غير صالحة (invalid). إذا كنت تقوم بأي نوع من المعالجة على هذه البيانات، وترغب في *تخطي* أو وضع علامة على هذه الإدخالات غير المرغوب فيها دون حذفها، فقد تضطر إلى استخدام الشروط (conditionals) أو تصفية بياناتك بطريقة ما. توفر وحدة [numpy.ma](https://numpy.org/devdocs/reference/maskedarray.generic.html#module-numpy.ma) بعض الوظائف نفسها لـ [NumPy ndarrays](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) مع بنية إضافية لضمان عدم استخدام الإدخالات غير الصالحة في الحساب.

من [الدليل المرجعي (Reference Guide)](https://numpy.org/devdocs/reference/maskedarray.generic.html#module-numpy.ma):

> المصفوفة المقنعة (A masked array) هي مزيج من [numpy.ndarray](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) قياسية و **قناع (mask)**. القناع إما `nomask`، مما يشير إلى أن لا توجد قيمة للمصفوفة المرتبطة غير صالحة، أو مصفوفة من القيم المنطقية (booleans) تحدد لكل عنصر من عناصر المصفوفة المرتبطة ما إذا كانت القيمة صالحة أم لا. عندما يكون عنصر القناع `False`، يكون العنصر المقابل للمصفوفة المرتبطة صالحًا ويقال إنه غير مقنع (unmasked). عندما يكون عنصر القناع `True`، يقال إن العنصر المقابل للمصفوفة المرتبطة مقنع (masked) (غير صالح).

يمكننا التفكير في [MaskedArray](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray) كمزيج من:

- البيانات (Data)، كمصفوفة `numpy.ndarray` عادية بأي شكل أو نوع بيانات (datatype)؛
- قناع منطقي (boolean mask) بنفس شكل البيانات؛
- `fill_value`، وهي قيمة يمكن استخدامها لاستبدال الإدخالات غير الصالحة من أجل إرجاع `numpy.ndarray` قياسية.

## متى يمكن أن تكون مفيدة؟ (When can they be useful?)

هناك بعض الحالات التي يمكن أن تكون فيها masked arrays أكثر فائدة من مجرد إزالة الإدخالات غير الصالحة من المصفوفة:

- عندما تريد الاحتفاظ بالقيم التي قمت بتقنيعها لمعالجتها لاحقًا، دون نسخ المصفوفة؛
- عندما تضطر إلى التعامل مع العديد من المصفوفات، لكل منها قناعها الخاص. إذا كان القناع جزءًا من المصفوفة، فإنك تتجنب الأخطاء وقد يكون الكود أكثر إحكامًا؛
- عندما يكون لديك علامات مختلفة للقيم المفقودة أو غير الصالحة، وترغب في الاحتفاظ بهذه العلامات دون استبدالها في مجموعة البيانات الأصلية، ولكن استبعادها من العمليات الحسابية؛
- إذا لم تتمكن من تجنب أو إزالة القيم المفقودة، ولكن لا ترغب في التعامل مع قيم [NaN (Not a Number)](https://numpy.org/devdocs/reference/constants.html#numpy.nan) في عملياتك.

تعد masked arrays أيضًا فكرة جيدة لأن وحدة `numpy.ma` تأتي أيضًا بتطبيق محدد لمعظم [دوال NumPy الشاملة (ufuncs)](https://numpy.org/devdocs/glossary.html#term-ufunc)، مما يعني أنه لا يزال بإمكانك تطبيق دوال وعمليات متجهة سريعة على البيانات المقنعة. يكون الناتج بعد ذلك masked array. سنرى بعض الأمثلة على كيفية عمل ذلك عمليًا أدناه.

+++

## استخدام masked arrays لعرض بيانات COVID-19

من [Kaggle](https://www.kaggle.com/atilamadai/covid19) من الممكن تنزيل مجموعة بيانات (dataset) تحتوي على بيانات أولية حول تفشي COVID-19 في بداية عام 2020. سننظر في مجموعة فرعية صغيرة من هذه البيانات، الموجودة في الملف `who_covid_19_sit_rep_time_series.csv`. *(لاحظ أنه تم استبدال هذا الملف بإصدار بدون بيانات مفقودة في وقت ما في أواخر عام 2020.)*

```{code-cell}
import numpy as np
import os

# تعيد الدالة os.getcwd() المجلد الحالي؛ يمكنك تغيير
# المتغير filepath ليشير إلى المجلد الذي حفظت فيه ملف .csv
filepath = os.getcwd()
filename = os.path.join(filepath, "who_covid_19_sit_rep_time_series.csv")
```

يحتوي ملف البيانات على بيانات من أنواع مختلفة وهو منظم على النحو التالي:

- الصف الأول هو سطر رأس (header line) يصف (في الغالب) البيانات في كل عمود يليه في الصفوف أدناه، وبدءًا من العمود الرابع، يكون الرأس هو تاريخ الملاحظة.
- الصفوف من الثاني إلى السابع تحتوي على بيانات ملخصة (summary data) من نوع مختلف عما سنقوم بفحصه، لذلك سنحتاج إلى استبعادها من البيانات التي سنعمل بها.
- تبدأ البيانات الرقمية التي نرغب في العمل بها من العمود 4، الصف 8، وتمتد من هناك إلى أقصى اليمين وأسفل صف.

دعنا نستكشف البيانات داخل هذا الملف لأول 14 يومًا من السجلات. لجمع البيانات من ملف `.csv`، سنستخدم دالة [numpy.genfromtxt](https://numpy.org/devdocs/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt)، مع التأكد من أننا نختار فقط الأعمدة التي تحتوي على أرقام فعلية بدلاً من الأعمدة الأربعة الأولى التي تحتوي على بيانات الموقع. نتخطى أيضًا أول 6 صفوف من هذا الملف، لأنها تحتوي على بيانات أخرى لا تهمنا. بشكل منفصل، سنستخرج المعلومات حول التواريخ والموقع لهذه البيانات.

```{code-cell}
# لاحظ أننا نستخدم skip_header و usecols لقراءة أجزاء فقط من
# ملف البيانات في كل متغير.
# اقرأ فقط التواريخ للأعمدة 4-18 من الصف الأول
dates = np.genfromtxt(
    filename,
    dtype=np.str_,
    delimiter=",",
    max_rows=1,
    usecols=range(4, 18),
    encoding="utf-8-sig",
)
# اقرأ أسماء المواقع الجغرافية من العمودين الأولين،
# مع تخطي الصفوف الستة الأولى
locations = np.genfromtxt(
    filename,
    dtype=np.str_,
    delimiter=",",
    skip_header=6,
    usecols=(0, 1),
    encoding="utf-8-sig",
)
# اقرأ البيانات الرقمية لأول 14 يومًا فقط
nbcases = np.genfromtxt(
    filename,
    dtype=np.int_,
    delimiter=",",
    skip_header=6,
    usecols=range(4, 18),
    encoding="utf-8-sig",
)
```

ضمن استدعاء دالة `numpy.genfromtxt`، اخترنا [numpy.dtype](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype) لكل مجموعة فرعية من البيانات (إما عدد صحيح - `numpy.int_` - أو سلسلة من الأحرف - `numpy.str_`). استخدمنا أيضًا الوسيط `encoding` لاختيار `utf-8-sig` كترميز للملف (اقرأ المزيد حول الترميز في [وثائق Python الرسمية](https://docs.python.org/3/library/codecs.html#encodings-and-unicode)). يمكنك قراءة المزيد حول دالة `numpy.genfromtxt` من [الوثائق المرجعية (Reference Documentation)](https://numpy.org/devdocs/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt) أو من [البرنامج التعليمي الأساسي للإدخال/الإخراج (Basic IO tutorial)](https://numpy.org/devdocs/user/basics.io.genfromtxt.html).

+++

## استكشاف البيانات (Exploring the data)

أولاً وقبل كل شيء، يمكننا رسم مجموعة البيانات الكاملة التي لدينا ونرى كيف تبدو. من أجل الحصول على رسم بياني قابل للقراءة، نختار فقط عددًا قليلاً من التواريخ لعرضها في [علامات المحور السيني (x-axis ticks)](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.xticks.html#matplotlib.pyplot.xticks). لاحظ أيضًا أننا في أمر الرسم البياني الخاص بنا، نستخدم `nbcases.T` (منقولة مصفوفة `nbcases`) لأن هذا يعني أننا سنرسم كل صف من الملف كخط منفصل. نختار رسم خط متقطع (باستخدام نمط الخط `'-'`). راجع وثائق [matplotlib](https://matplotlib.org/) لمزيد من المعلومات حول هذا.

```{code-cell}
import matplotlib.pyplot as plt

selected_dates = [0, 3, 11, 13]
plt.plot(dates, nbcases.T, "--")
plt.xticks(selected_dates, dates[selected_dates])
plt.title("COVID-19 cumulative cases from Jan 21 to Feb 3 2020")
```

يحتوي الرسم البياني على شكل غريب من 24 يناير إلى 1 فبراير. سيكون من المثير للاهتمام معرفة مصدر هذه البيانات. إذا نظرنا إلى مصفوفة `locations` التي استخرجناها من ملف `.csv`، يمكننا أن نرى أن لدينا عمودين، حيث يحتوي الأول على المناطق ويحتوي الثاني على اسم البلد. ومع ذلك، فإن الصفوف القليلة الأولى فقط تحتوي على بيانات للعمود الأول (أسماء المقاطعات في الصين). بعد ذلك، لدينا فقط أسماء البلدان. لذلك سيكون من المنطقي تجميع جميع البيانات من الصين في صف واحد. لهذا، سنختار من مصفوفة `nbcases` فقط الصفوف التي يتوافق فيها الإدخال الثاني لمصفوفة `locations` مع الصين. بعد ذلك، سنستخدم دالة [numpy.sum](https://numpy.org/devdocs/reference/generated/numpy.sum.html#numpy.sum) لجمع جميع الصفوف المختارة (`axis=0`). لاحظ أيضًا أن الصف 35 يتوافق مع الإجماليات للبلد بأكمله لكل تاريخ. نظرًا لأننا نريد حساب المجموع بأنفسنا من بيانات المقاطعات، يجب علينا إزالة هذا الصف أولاً من كل من `locations` و `nbcases`:

```{code-cell}
totals_row = 35
locations = np.delete(locations, (totals_row), axis=0)
nbcases = np.delete(nbcases, (totals_row), axis=0)

china_total = nbcases[locations[:, 1] == "China"].sum(axis=0)
china_total
```

هناك خطأ ما في هذه البيانات - لا ينبغي أن يكون لدينا قيم سالبة في مجموعة بيانات تراكمية. ما الذي يحدث؟

+++

## البيانات المفقودة (Missing data)

بالنظر إلى البيانات، هذا ما نجده: هناك فترة بها **بيانات مفقودة (missing data)**:

```{code-cell}
nbcases
```

جميع قيم `-1` التي نراها تأتي من محاولة `numpy.genfromtxt` قراءة البيانات المفقودة من ملف `.csv` الأصلي. من الواضح أننا لا نرغب في حساب البيانات المفقودة على أنها `-1` - نريد فقط تخطي هذه القيمة حتى لا تتداخل في تحليلنا. بعد استيراد وحدة `numpy.ma`، سنقوم بإنشاء مصفوفة جديدة، هذه المرة بتقنيع القيم غير الصالحة:

```{code-cell}
from numpy import ma

nbcases_ma = ma.masked_values(nbcases, -1)
```

إذا نظرنا إلى masked array `nbcases_ma`، فهذا ما لدينا:

```{code-cell}
nbcases_ma
```

يمكننا أن نرى أن هذا نوع مختلف من المصفوفات. كما ذكرنا في المقدمة، لها ثلاث سمات (`data` و `mask` و `fill_value`). ضع في اعتبارك أن السمة `mask` لها قيمة `True` للعناصر المقابلة للبيانات **غير الصالحة** (الممثلة بشرطتين في السمة `data`).

+++

دعنا نحاول ونرى كيف تبدو البيانات باستثناء الصف الأول (بيانات من مقاطعة هوبي في الصين) حتى نتمكن من النظر إلى البيانات المفقودة عن كثب:

```{code-cell}
plt.plot(dates, nbcases_ma[1:].T, "--")
plt.xticks(selected_dates, dates[selected_dates])
plt.title("COVID-19 cumulative cases from Jan 21 to Feb 3 2020")
```

الآن بعد أن تم تقنيع بياناتنا، دعنا نحاول جمع جميع الحالات في الصين:

```{code-cell}
china_masked = nbcases_ma[locations[:, 1] == "China"].sum(axis=0)
china_masked
```

لاحظ أن `china_masked` هي masked array، لذا فهي تحتوي على بنية بيانات مختلفة عن مصفوفة NumPy العادية. الآن، يمكننا الوصول إلى بياناتها مباشرة باستخدام السمة `.data`:

```{code-cell}
china_total = china_masked.data
china_total
```

هذا أفضل: لا توجد قيم سالبة بعد الآن. ومع ذلك، لا يزال بإمكاننا أن نرى أنه في بعض الأيام، يبدو أن العدد التراكمي للحالات ينخفض (من 835 إلى 10، على سبيل المثال)، وهو ما لا يتفق مع تعريف "البيانات التراكمية". إذا نظرنا عن كثب إلى البيانات، يمكننا أن نرى أنه في الفترة التي كانت فيها بيانات مفقودة في البر الرئيسي للصين، كانت هناك بيانات صالحة لهونغ كونغ وتايوان وماكاو ومناطق "غير محددة" في الصين. ربما يمكننا إزالة هذه من المجموع الكلي للحالات في الصين، للحصول على فهم أفضل للبيانات.

أولاً، سنحدد مؤشرات المواقع في البر الرئيسي للصين:

```{code-cell}
china_mask = (
    (locations[:, 1] == "China")
    & (locations[:, 0] != "Hong Kong")
    & (locations[:, 0] != "Taiwan")
    & (locations[:, 0] != "Macau")
    & (locations[:, 0] != "Unspecified*")
)
```

```{code-cell}
china_mask.nonzero()
```

الآن يمكننا جمع الإدخالات بشكل صحيح للبر الرئيسي للصين:

```{code-cell}
china_total = nbcases_ma[china_mask].sum(axis=0)
china_total
```

يمكننا استبدال البيانات بهذه المعلومات ورسم رسم بياني جديد، مع التركيز على البر الرئيسي للصين:

```{code-cell}
plt.plot(dates, china_total.T, "--")
plt.xticks(selected_dates, dates[selected_dates])
plt.title("COVID-19 cumulative cases from Jan 21 to Feb 3 2020 - Mainland China")
```

من الواضح أن masked arrays هي الحل الصحيح هنا. لا يمكننا تمثيل البيانات المفقودة دون تشويه تطور المنحنى.

+++

## مطابقة البيانات (Fitting Data)

أحد الاحتمالات التي يمكننا التفكير فيها هو استيفاء البيانات المفقودة (interpolate the missing data) لتقدير عدد الحالات في أواخر يناير. لاحظ أنه يمكننا تحديد العناصر المقنعة باستخدام السمة `.mask`:

```{code-cell}
china_total.mask
invalid = china_total[china_total.mask]
invalid
```

يمكننا أيضًا الوصول إلى الإدخالات الصالحة باستخدام النفي المنطقي (logical negation) لهذا القناع:

```{code-cell}
valid = china_total[~china_total.mask]
valid
```

الآن، إذا أردنا إنشاء تقريب بسيط جدًا لهذه البيانات، فيجب أن نأخذ في الاعتبار الإدخالات الصالحة حول الإدخالات غير الصالحة. لذلك أولاً دعنا نختار التواريخ التي تكون فيها البيانات صالحة. لاحظ أنه يمكننا استخدام القناع من masked array `china_total` لفهرسة مصفوفة التواريخ:

```{code-cell}
dates[~china_total.mask]
```

أخيرًا، يمكننا استخدام [وظيفة المطابقة (fitting functionality) لحزمة numpy.polynomial](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html) لإنشاء نموذج متعدد الحدود من الدرجة الثالثة (cubic polynomial model) يتناسب مع البيانات بأفضل شكل ممكن:

```{code-cell}
t = np.arange(len(china_total))
model = np.polynomial.Polynomial.fit(t[~china_total.mask], valid, deg=3)
plt.plot(t, china_total)
plt.plot(t, model(t), "--")
```

هذا الرسم البياني ليس قابلاً للقراءة بشكل كبير حيث تبدو الخطوط متراكبة، لذلك دعنا نلخص في رسم بياني أكثر تفصيلاً. سنرسم البيانات الحقيقية عندما تكون متاحة، ونعرض المطابقة التكعيبية (cubic fit) للبيانات غير المتاحة، باستخدام هذه المطابقة لحساب تقدير للعدد الملاحظ للحالات في 28 يناير 2020، بعد 7 أيام من بداية السجلات:

```{code-cell}
plt.plot(t, china_total)
plt.plot(t[china_total.mask], model(t)[china_total.mask], "--", color="orange")
plt.plot(7, model(7), "r*")
plt.xticks([0, 7, 13], dates[[0, 7, 13]])
plt.yticks([0, model(7), 10000, 17500])
plt.legend(["Mainland China", "Cubic estimate", "7 days after start"])
plt.title(
    "COVID-19 cumulative cases from Jan 21 to Feb 3 2020 - Mainland China\n"
    "Cubic estimate for 7 days after start"
)
```

## عمليًا (In practice)

+++

- إضافة `-1` إلى البيانات المفقودة ليست مشكلة مع `numpy.genfromtxt`؛ في هذه الحالة بالذات، قد يكون استبدال القيمة المفقودة بـ `0` مقبولًا، لكننا سنرى لاحقًا أن هذا بعيد عن الحل العام. أيضًا، من الممكن استدعاء دالة `numpy.genfromtxt` باستخدام المعامل `usemask`. إذا كان `usemask=True`، فإن `numpy.genfromtxt` تعيد تلقائيًا masked array.

+++

## قراءات إضافية (Further reading)

يمكن العثور على الموضوعات غير المشمولة في هذا البرنامج التعليمي في الوثائق:

- [Hardmasks](https://numpy.org/devdocs/reference/generated/numpy.ma.harden_mask.html#numpy.ma.harden_mask) مقابل [softmasks](https://numpy.org/devdocs/reference/generated/numpy.ma.soften_mask.html#numpy.ma.soften_mask)
- [وحدة numpy.ma](https://numpy.org/devdocs/reference/maskedarray.generic.html#maskedarray-generic)

### المراجع (Reference)

- Ensheng Dong, Hongru Du, Lauren Gardner, *An interactive web-based dashboard to track COVID-19 in real time*, The Lancet Infectious Diseases, Volume 20, Issue 5, 2020, Pages 533-534, ISSN 1473-3099, [https://doi.org/10.1016/S1473-3099(20)30120-1](https://doi.org/10.1016/S1473-3099(20)30120-1).
