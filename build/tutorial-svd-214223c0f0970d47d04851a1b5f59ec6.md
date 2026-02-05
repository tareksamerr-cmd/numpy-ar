---
title: الجبر الخطي على المصفوفات n-الأبعاد
short_title: الجبر الخطي على المصفوفات n-الأبعاد
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

# الجبر الخطي على المصفوفات n-الأبعاد (Linear algebra on n-dimensional arrays)

+++

## المتطلبات الأساسية (Prerequisites)

قبل قراءة هذا البرنامج التعليمي، يجب أن تكون لديك معرفة بسيطة بلغة Python. إذا كنت ترغب في تحديث ذاكرتك، ألقِ نظرة على [البرنامج التعليمي لـ Python](https://docs.python.org/3/tutorial/).

إذا كنت ترغب في تشغيل الأمثلة في هذا البرنامج التعليمي، يجب أن يكون لديك أيضًا [matplotlib](https://matplotlib.org/) و [SciPy](https://scipy.org) مثبتين على جهاز الكمبيوتر الخاص بك.

## ملف تعريف المتعلم (Learner profile)

هذا البرنامج التعليمي مخصص للأشخاص الذين لديهم فهم أساسي للجبر الخطي (linear algebra) والمصفوفات (arrays) في NumPy ويرغبون في فهم كيفية تمثيل المصفوفات n-الأبعاد ($n>=2$) وكيف يمكن التعامل معها. على وجه الخصوص، إذا كنت لا تعرف كيفية تطبيق الدوال الشائعة على المصفوفات n-الأبعاد (دون استخدام الحلقات التكرارية (for-loops))، أو إذا كنت ترغب في فهم خصائص المحور (axis) والشكل (shape) للمصفوفات n-الأبعاد، فقد يكون هذا البرنامج التعليمي مفيدًا.

## أهداف التعلم (Learning Objectives)

بعد هذا البرنامج التعليمي، يجب أن تكون قادرًا على:

- فهم الفرق بين المصفوفات أحادية وثنائية و n-الأبعاد في NumPy؛
- فهم كيفية تطبيق بعض عمليات الجبر الخطي على المصفوفات n-الأبعاد دون استخدام for-loops؛
- فهم خصائص axis و shape للمصفوفات n-الأبعاد.

## المحتوى (Content)

في هذا البرنامج التعليمي، سنستخدم **تحليل المصفوفة (matrix decomposition)** من linear algebra، وهو **تحليل القيمة المفردة (Singular Value Decomposition)**، لإنشاء تقريب مضغوط لصورة. سنستخدم صورة `face` من وحدة [scipy.datasets](https://docs.scipy.org/doc/scipy/reference/datasets.html):

```{code-cell}
from scipy.datasets import face

img = face()
```

```{note}
إذا كنت تفضل ذلك، يمكنك استخدام صورتك الخاصة أثناء العمل في هذا البرنامج التعليمي.
لتحويل صورتك إلى مصفوفة NumPy يمكن التعامل معها، يمكنك استخدام دالة `imread` من الوحدة الفرعية [matplotlib.pyplot](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot).
بدلاً من ذلك، يمكنك استخدام دالة [imageio.imread](https://imageio.readthedocs.io/en/stable/_autosummary/imageio.v3.imread.html) من مكتبة `imageio`.
كن على دراية بأنه إذا كنت تستخدم صورتك الخاصة، فمن المحتمل أن تحتاج إلى تكييف الخطوات أدناه.
لمزيد من المعلومات حول كيفية معالجة الصور عند تحويلها إلى مصفوفات NumPy، راجع [دورة مكثفة حول NumPy للصور](https://scikit-image.org/docs/stable/user_guide/numpy_images.html) من وثائق `scikit-image`.
```

+++

الآن، `img` هي مصفوفة NumPy، كما نرى عند استخدام دالة `type`:

```{code-cell}
type(img)
```

يمكننا رؤية الصورة باستخدام دالة [matplotlib.pyplot.imshow](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html#matplotlib.pyplot.imshow) وأمر iPython الخاص، `%matplotlib inline` لعرض الرسوم البيانية (plots) مضمنة:

```{code-cell}
import matplotlib.pyplot as plt

%matplotlib inline
```

```{code-cell}
plt.imshow(img)
plt.show()
```

### خصائص الشكل (Shape)، المحور (axis) والمصفوفة (array)

لاحظ أنه في linear algebra، يشير بُعد المتجه (vector) إلى عدد الإدخالات في مصفوفة. في NumPy، فإنه يحدد بدلاً من ذلك عدد المحاور (axes). على سبيل المثال، مصفوفة 1D هي vector مثل `[1, 2, 3]`، ومصفوفة 2D هي مصفوفة (matrix)، وهكذا.

أولاً، دعنا نتحقق من shape البيانات في مصفوفتنا. نظرًا لأن هذه الصورة ثنائية الأبعاد (تشكل البكسلات في الصورة مستطيلاً)، فقد نتوقع مصفوفة ثنائية الأبعاد لتمثيلها (matrix). ومع ذلك، فإن استخدام خاصية `shape` لهذه المصفوفة NumPy يعطينا نتيجة مختلفة:

```{code-cell}
img.shape
```

الناتج هو [tuple](https://docs.python.org/dev/tutorial/datastructures.html#tut-tuples) بثلاثة عناصر، مما يعني أن هذه مصفوفة ثلاثية الأبعاد. نظرًا لأن هذه صورة ملونة، وقد استخدمنا دالة `imread` لقراءتها، يتم تنظيم البيانات كشبكة 768×1024 من البكسلات، حيث يحتوي كل بكسل على 3 قيم تمثل قنوات الألوان (الأحمر والأخضر والأزرق - RGB). يمكنك رؤية ذلك من خلال النظر إلى shape، حيث يتوافق الرقم الأيسر مع المحور الخارجي (ارتفاع الصورة)، والرقم الأوسط مع المحور التالي (عرض الصورة) والرقم الأيمن مع المحور الداخلي (قنوات الألوان).

علاوة على ذلك، باستخدام خاصية `ndim` لهذه المصفوفة، يمكننا أن نرى أن

```{code-cell}
img.ndim
```

يشير NumPy إلى كل بُعد على أنه *axis*. نظرًا لكيفية عمل `imread`، فإن *الفهرس الأول في المحور الثالث* هو بيانات البكسل الأحمر لصورتنا. يمكننا الوصول إلى هذا باستخدام بناء الجملة (syntax):

```{code-cell}
img[:, :, 0]
```

من الناتج أعلاه، يمكننا أن نرى أن كل قيمة في `img[:, :, 0]` هي قيمة عدد صحيح (integer value) بين 0 و 255، تمثل مستوى اللون الأحمر في كل بكسل صورة مقابل (ضع في اعتبارك أن هذا قد يكون مختلفًا إذا كنت تستخدم صورتك الخاصة بدلاً من [scipy.datasets.face](https://docs.scipy.org/doc/scipy/reference/generated/scipy.datasets.face.html)).

كما هو متوقع، هذه مصفوفة 768x1024:

```{code-cell}
img[:, :, 0].shape
```

نظرًا لأننا سنقوم بإجراء عمليات linear algebra على هذه البيانات، فقد يكون من المثير للاهتمام أن يكون لدينا أرقام حقيقية بين 0 و 1 في كل إدخال من المصفوفات لتمثيل قيم RGB. يمكننا القيام بذلك عن طريق تعيين

```{code-cell}
img_array = img / 255
```

تعمل هذه العملية، وهي قسمة مصفوفة على scalar، بسبب [قواعد البث (broadcasting rules)](https://numpy.org/devdocs/user/theory.broadcasting.html#array-broadcasting-in-numpy) الخاصة بـ NumPy.

```{tip}
في التطبيقات الواقعية، قد يكون من الأفضل استخدام، على سبيل المثال، دالة المساعدة [img_as_float](https://scikit-image.org/docs/stable/api/skimage.html#skimage.img_as_float) من `scikit-image`.
```

يمكنك التحقق من أن ما ورد أعلاه يعمل عن طريق إجراء بعض الاختبارات؛ على سبيل المثال، الاستعلام عن القيم القصوى والدنيا لهذه المصفوفة:

```{code-cell}
img_array.min(), img_array.max()
```

أو التحقق من نوع البيانات في المصفوفة:

```{code-cell}
img_array.dtype
```

لاحظ أنه يمكننا تعيين كل قناة لونية لمصفوفة منفصلة باستخدام syntax الشرائح (slice syntax):

```{code-cell}
red_array = img_array[:, :, 0]
green_array = img_array[:, :, 1]
blue_array = img_array[:, :, 2]
```

### العمليات على محور (Operations on an axis)

من الممكن استخدام طرق من linear algebra لتقريب مجموعة بيانات موجودة. هنا، سنستخدم SVD (Singular Value Decomposition) لمحاولة إعادة بناء صورة تستخدم معلومات قيمة مفردة (singular value information) أقل من الصورة الأصلية، مع الاحتفاظ ببعض ميزاتها.

+++

```{note}
سنستخدم وحدة linear algebra الخاصة بـ NumPy، [numpy.linalg](https://numpy.org/devdocs/reference/routines.linalg.html#module-numpy.linalg)، لإجراء العمليات في هذا البرنامج التعليمي.
يمكن العثور على معظم دوال linear algebra في هذه الوحدة أيضًا في [scipy.linalg](https://docs.scipy.org/doc/scipy/reference/linalg.html#module-scipy.linalg)، ويتم تشجيع المستخدمين على استخدام وحدة [scipy](https://docs.scipy.org/doc/scipy/reference/index.html#module-scipy) للتطبيقات الواقعية.
ومع ذلك، فإن بعض الدوال في وحدة [scipy.linalg](https://docs.scipy.org/doc/scipy/reference/linalg.html#module-scipy.linalg)، مثل دالة SVD، تدعم فقط المصفوفات ثنائية الأبعاد.
لمزيد من المعلومات حول هذا، تحقق من [صفحة scipy.linalg](https://docs.scipy.org/doc/scipy/tutorial/linalg.html).
```

+++

من أجل استخراج المعلومات من matrix معينة، يمكننا استخدام SVD للحصول على 3 مصفوفات يمكن ضربها للحصول على matrix الأصلية. من نظرية linear algebra، بالنظر إلى matrix $A$، يمكن حساب المنتج التالي:

$$U \Sigma V^T = A$$

حيث $U$ و $V^T$ مربعان و $\Sigma$ بنفس حجم $A$. $\Sigma$ هي matrix قطرية (diagonal matrix) وتحتوي على [القيم المفردة (singular values)](https://en.wikipedia.org/wiki/Singular_value) لـ $A$، منظمة من الأكبر إلى الأصغر. هذه القيم دائمًا غير سالبة ويمكن استخدامها كمؤشر على "أهمية" بعض الميزات الممثلة بواسطة matrix $A$.

دعنا نرى كيف يعمل هذا عمليًا مع matrix واحدة فقط أولاً. لاحظ أنه وفقًا [لقياس الألوان (colorimetry)](https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale)، من الممكن الحصول على نسخة تدرج رمادي (grayscale version) معقولة جدًا من صورتنا الملونة إذا طبقنا الصيغة:

$$Y = 0.2126 R + 0.7152 G + 0.0722 B$$

حيث $Y$ هي المصفوفة التي تمثل صورة grayscale، و $R$ و $G$ و $B$ هي مصفوفات القنوات الحمراء والخضراء والزرقاء التي كانت لدينا في الأصل. لاحظ أنه يمكننا استخدام عامل التشغيل `@` (عامل تشغيل ضرب المصفوفات (matrix multiplication operator) لمصفوفات NumPy، انظر [numpy.matmul](https://numpy.org/devdocs/reference/generated/numpy.matmul.html#numpy.matmul)) لهذا:

```{code-cell}
img_gray = img_array @ [0.2126, 0.7152, 0.0722]
```

الآن، `img_gray` لها shape:

```{code-cell}
img_gray.shape
```

لمعرفة ما إذا كان هذا منطقيًا في صورتنا، يجب أن نستخدم خريطة ألوان (colormap) من `matplotlib` تتوافق مع اللون الذي نرغب في رؤيته في صورتنا (وإلا، ستستخدم `matplotlib` خريطة ألوان افتراضية لا تتوافق مع البيانات الحقيقية).

في حالتنا، نحن نقرب جزء grayscale من الصورة، لذلك سنستخدم colormap `gray`:

```{code-cell}
plt.imshow(img_gray, cmap="gray")
plt.show()
```

الآن، بتطبيق دالة [linalg.svd](https://numpy.org/devdocs/reference/generated/numpy.linalg.svd.html#numpy.linalg.svd) على هذه matrix، نحصل على التحليل التالي:

```{code-cell}
import numpy as np
U, s, Vt = np.linalg.svd(img_gray)
```

```{note}
إذا كنت تستخدم صورتك الخاصة، فقد يستغرق هذا الأمر بعض الوقت للتشغيل، اعتمادًا على حجم صورتك وأجهزتك.
لا تقلق، هذا طبيعي! يمكن أن يكون SVD حسابًا مكثفًا جدًا.
```

+++

دعنا نتحقق مما إذا كان هذا ما توقعناه:

```{code-cell}
U.shape, s.shape, Vt.shape
```

لاحظ أن `s` لها shape خاص: لها بُعد واحد فقط. هذا يعني أن بعض دوال linear algebra التي تتوقع مصفوفات ثنائية الأبعاد قد لا تعمل. على سبيل المثال، من النظرية، قد يتوقع المرء أن تكون `s` و `Vt` متوافقتين للضرب. ومع ذلك، هذا ليس صحيحًا لأن `s` ليس لها محور ثانٍ:

```{code-cell}
:tags: [raises-exception]
s @ Vt
```

ينتج عنه `ValueError`. يحدث هذا لأن وجود مصفوفة أحادية الأبعاد لـ `s`، في هذه الحالة، أكثر اقتصادًا بكثير عمليًا من بناء diagonal matrix بنفس البيانات. لإعادة بناء matrix الأصلية، يمكننا إعادة بناء diagonal matrix $\Sigma$ بعناصر `s` في قطرها وبالأبعاد المناسبة للضرب: في حالتنا، يجب أن تكون $\Sigma$ 768x1024 نظرًا لأن `U` هي 768x768 و `Vt` هي 1024x1024. من أجل إضافة singular values إلى قطر `Sigma`، سنستخدم دالة [fill_diagonal](https://numpy.org/devdocs/reference/generated/numpy.fill_diagonal.html) من NumPy:

```{code-cell}
Sigma = np.zeros((U.shape[1], Vt.shape[0]))
np.fill_diagonal(Sigma, s)
```

الآن، نريد التحقق مما إذا كانت `U @ Sigma @ Vt` المعاد بناؤها قريبة من matrix `img_gray` الأصلية.

+++

## التقريب (Approximation)

```{code-cell}
np.linalg.norm(img_gray - U @ Sigma @ Vt)
```

(قد تختلف النتيجة الفعلية لهذه العملية اعتمادًا على بنيتك وإعداد linear algebra الخاص بك. بغض النظر، يجب أن ترى رقمًا صغيرًا.)

يمكننا أيضًا استخدام دالة [numpy.allclose](https://numpy.org/devdocs/reference/generated/numpy.allclose.html#numpy.allclose) للتأكد من أن المنتج المعاد بناؤه هو، في الواقع، *قريب* من matrix الأصلية (الفرق بين المصفوفتين صغير):

```{code-cell}
np.allclose(img_gray, U @ Sigma @ Vt)
```

لمعرفة ما إذا كان التقريب معقولًا، يمكننا التحقق من القيم في `s`:

```{code-cell}
plt.plot(s)
plt.show()
```

في الرسم البياني، يمكننا أن نرى أنه على الرغم من أن لدينا 768 singular values في `s`، فإن معظمها (بعد الإدخال 150 أو نحو ذلك) صغيرة جدًا. لذلك قد يكون من المنطقي استخدام المعلومات المتعلقة بـ *singular values* الخمسين الأولى فقط (على سبيل المثال) لبناء تقريب أكثر اقتصادية لصورتنا.

الفكرة هي اعتبار جميع singular values في `Sigma` باستثناء `k` الأولى (وهي نفس الموجودة في `s`) أصفارًا، مع الحفاظ على `U` و `Vt` سليمتين، وحساب ناتج هذه المصفوفات كتقريب.

على سبيل المثال، إذا اخترنا

```{code-cell}
k = 10
```

يمكننا بناء التقريب عن طريق القيام بـ

```{code-cell}
approx = U @ Sigma[:, :k] @ Vt[:k, :]
```

لاحظ أنه كان علينا استخدام الصفوف `k` الأولى فقط من `Vt`، نظرًا لأن جميع الصفوف الأخرى سيتم ضربها بالأصفار المقابلة لـ singular values التي أزلناها من هذا التقريب.

```{code-cell}
plt.imshow(approx, cmap="gray")
plt.show()
```

الآن، يمكنك المضي قدمًا وتكرار هذه التجربة بقيم أخرى لـ `k`، ويجب أن تمنحك كل تجربة من تجاربك صورة أفضل قليلاً (أو أسوأ) اعتمادًا على القيمة التي تختارها.

+++

### التطبيق على جميع الألوان (Applying to all colors)

الآن نريد القيام بنفس نوع العملية، ولكن لجميع الألوان الثلاثة. قد تكون غريزتنا الأولى هي تكرار نفس العملية التي قمنا بها أعلاه لكل matrix لونية على حدة. ومع ذلك، فإن *broadcasting* الخاص بـ NumPy يتولى هذا الأمر نيابة عنا.

إذا كانت مصفوفتنا تحتوي على أكثر من بُعدين، فيمكن تطبيق SVD على جميع المحاور في وقت واحد. ومع ذلك، تتوقع دوال linear algebra في NumPy رؤية مصفوفة من الشكل `(n, M, N)`، حيث يمثل المحور الأول `n` عدد المصفوفات `MxN` في المكدس (stack).

في حالتنا،

```{code-cell}
img_array.shape
```

لذلك نحتاج إلى تبديل المحور (permutating the axis) في هذه المصفوفة للحصول على shape مثل `(3, 768, 1024)`. لحسن الحظ، يمكن لدالة [numpy.transpose](https://numpy.org/devdocs/reference/generated/numpy.transpose.html#numpy.transpose) القيام بذلك نيابة عنا:

```{code-cell}
# تشير القيم في الـ tuple إلى البعد الأصلي، والترتيب هو المحور الجديد
# لذا المحور 2 -> 0، 0 -> 1، و 1 -> 2
img_array_transposed = np.transpose(img_array, (2, 0, 1))
img_array_transposed.shape
```

الآن نحن جاهزون لتطبيق SVD:

```{code-cell}
U, s, Vt = np.linalg.svd(img_array_transposed)
```

أخيرًا، للحصول على الصورة المقربة الكاملة، نحتاج إلى إعادة تجميع هذه المصفوفات في التقريب. الآن، لاحظ أن

```{code-cell}
U.shape, s.shape, Vt.shape
```

لبناء matrix التقريب النهائية، يجب أن نفهم كيف يعمل الضرب عبر المحاور المختلفة.

+++

### المنتجات مع المصفوفات n-الأبعاد (Products with n-dimensional arrays)

إذا كنت قد عملت من قبل مع مصفوفات أحادية أو ثنائية الأبعاد فقط في NumPy، فقد تستخدم [numpy.dot](https://numpy.org/devdocs/reference/generated/numpy.dot.html#numpy.dot) و [numpy.matmul](https://numpy.org/devdocs/reference/generated/numpy.matmul.html#numpy.matmul) (أو عامل التشغيل `@`) بالتبادل. ومع ذلك، بالنسبة للمصفوفات n-الأبعاد، فإنها تعمل بطرق مختلفة جدًا. لمزيد من التفاصيل، تحقق من الوثائق حول [numpy.matmul](https://numpy.org/devdocs/reference/generated/numpy.matmul.html#numpy.matmul).

الآن، لبناء تقريبنا، نحتاج أولاً إلى التأكد من أن singular values جاهزة للضرب، لذلك نبني matrix `Sigma` الخاصة بنا بشكل مشابه لما فعلناه من قبل. يجب أن تحتوي مصفوفة `Sigma` على أبعاد `(3, 768, 1024)`. من أجل إضافة singular values إلى قطر `Sigma`، سنستخدم مرة أخرى دالة [fill_diagonal](https://numpy.org/devdocs/reference/generated/numpy.fill_diagonal.html)، باستخدام كل من الصفوف الثلاثة في `s` كقطر لكل من المصفوفات الثلاث في `Sigma`:

```{code-cell}
Sigma = np.zeros((3, 768, 1024))
for j in range(3):
    np.fill_diagonal(Sigma[j, :, :], s[j, :])
```

الآن، إذا أردنا إعادة بناء SVD الكامل (بدون تقريب)، يمكننا القيام بـ

```{code-cell}
reconstructed = U @ Sigma @ Vt
```

لاحظ أن

```{code-cell}
reconstructed.shape
```

يجب أن تكون الصورة المعاد بناؤها لا يمكن تمييزها عن الصورة الأصلية، باستثناء الاختلافات الناتجة عن أخطاء النقطة العائمة (floating point errors) من إعادة البناء. تذكر أن صورتنا الأصلية تتكون من قيم floating point في النطاق `[0., 1.]`. يمكن أن يؤدي تراكم floating point error من إعادة البناء إلى قيم خارج هذا النطاق الأصلي قليلاً:

```{code-cell}
reconstructed.min(), reconstructed.max()
```

نظرًا لأن `imshow` تتوقع قيمًا في النطاق، يمكننا استخدام `clip` لإزالة floating point error:

```{code-cell}
reconstructed = np.clip(reconstructed, 0, 1)
plt.imshow(np.transpose(reconstructed, (1, 2, 0)))
plt.show()
```

```{note}
في الواقع، تقوم `imshow` بإجراء هذا القص (clipping) ضمنيًا، لذلك إذا تخطيت السطر الأول في خلية الكود السابقة، فقد ترى رسالة تحذير تقول `"Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers)."`
```

الآن، لإجراء التقريب، يجب أن نختار فقط singular values `k` الأولى لكل قناة لونية. يمكن القيام بذلك باستخدام بناء الجملة التالي:

```{code-cell}
approx_img = U @ Sigma[..., :k] @ Vt[..., :k, :]
```

يمكنك أن ترى أننا اخترنا فقط المكونات `k` الأولى من المحور الأخير لـ `Sigma` (وهذا يعني أننا استخدمنا فقط الأعمدة `k` الأولى لكل من المصفوفات الثلاث في stack)، وأننا اخترنا فقط المكونات `k` الأولى في المحور الثاني من الأخير لـ `Vt` (وهذا يعني أننا اخترنا فقط الصفوف `k` الأولى من كل matrix في stack `Vt` وجميع الأعمدة). إذا لم تكن على دراية بـ ellipsis syntax، فهو عنصر نائب (placeholder) للمحاور الأخرى. لمزيد من التفاصيل، راجع الوثائق حول [الفهرسة (Indexing)](https://numpy.org/devdocs/user/basics.indexing.html#basics-indexing).

الآن،

```{code-cell}
approx_img.shape
```

وهو ليس shape الصحيح لعرض الصورة. أخيرًا، بإعادة ترتيب المحاور إلى shape الأصلي `(768, 1024, 3)`، يمكننا رؤية تقريبنا:

```{code-cell}
plt.imshow(np.transpose(np.clip(approx_img, 0, 1), (1, 2, 0)))
plt.show()
```

على الرغم من أن الصورة ليست حادة بنفس القدر، إلا أن استخدام عدد صغير من singular values `k` (مقارنة بالمجموعة الأصلية المكونة من 768 قيمة)، يمكننا استعادة العديد من الميزات المميزة من هذه الصورة.

+++

### كلمات أخيرة (Final words)

بالطبع، هذه ليست أفضل طريقة *لتقريب* صورة. ومع ذلك، هناك، في الواقع، نتيجة في linear algebra تقول إن التقريب الذي بنيناه أعلاه هو الأفضل الذي يمكننا الحصول عليه لـ matrix الأصلية من حيث norm الفرق. لمزيد من المعلومات، راجع *G. H. Golub and C. F. Van Loan, Matrix Computations, Baltimore, MD, Johns Hopkins University Press, 1985*.

## قراءات إضافية (Further reading)

- [برنامج Python التعليمي](https://docs.python.org/dev/tutorial/index.html)
- [مرجع NumPy](https://numpy.org/devdocs/reference/index.html#reference)
- [برنامج SciPy التعليمي](https://docs.scipy.org/doc/scipy/tutorial/index.html)
- [ملاحظات محاضرات SciPy](https://scipy-lectures.org)
- [قاموس matlab, R, IDL, NumPy/SciPy](http://mathesaurus.sf.net/)
