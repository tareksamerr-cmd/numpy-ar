---
short_title: مشاركة بيانات المصفوفة
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

# حفظ ومشاركة مصفوفات NumPy الخاصة بك

## ماذا ستتعلم (What you'll learn)

ستقوم بحفظ مصفوفات NumPy الخاصة بك كملفات مضغوطة (zipped files) وملفات نصية قابلة للقراءة البشرية مفصولة بفواصل (comma-delimited files) أي `*.csv`. ستتعلم أيضًا كيفية تحميل كلا النوعين من الملفات مرة أخرى إلى مساحات عمل NumPy.

## ماذا ستفعل (What you'll do)

ستتعلم طريقتين لحفظ وقراءة الملفات - كملفات مضغوطة (compressed) وملفات نصية (text files) - والتي ستلبي معظم احتياجات التخزين الخاصة بك في NumPy.

* ستنشئ مصفوفتين أحاديتي الأبعاد (1D arrays) ومصفوفة ثنائية الأبعاد (2D array)
* ستقوم بحفظ هذه المصفوفات في ملفات
* ستقوم بإزالة المتغيرات من مساحة عملك
* ستقوم بتحميل المتغيرات من ملفك المحفوظ
* ستقارن الملفات الثنائية المضغوطة (zipped binary files) بالملفات المحددة القابلة للقراءة البشرية (human-readable delimited files)
* ستنهي بمهارات حفظ وتحميل ومشاركة مصفوفات NumPy


## ماذا ستحتاج (What you'll need)

* NumPy
* إذن قراءة وكتابة (read-write access) إلى دليل العمل الخاص بك

قم بتحميل الدوال الضرورية باستخدام الأمر التالي.

```{code-cell}
import numpy as np
```

في هذا البرنامج التعليمي، ستستخدم دوال Python و IPython magic و NumPy التالية:

* [`np.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html)
* [`np.savez`](https://numpy.org/doc/stable/reference/generated/numpy.savez.html)
* [`del`](https://docs.python.org/3/reference/simple_stmts.html#del)
* [`whos`](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-whos)
* [`np.load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html)
* [`np.block`](https://numpy.org/doc/stable/reference/generated/numpy.block.html)
* [`np.newaxis`](https://numpy.org/doc/stable/reference/constants.html?highlight=newaxis#numpy.newaxis)
* [`np.savetxt`](https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html)
* [`np.loadtxt`](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html)

+++

---

## إنشاء مصفوفاتك (Create your arrays)

الآن بعد أن قمت باستيراد مكتبة NumPy، يمكنك إنشاء مصفوفتين؛ لنبدأ بمصفوفتين أحاديتي الأبعاد (1D arrays)، `x` و `y`، حيث `y = x**2`. ستقوم بتعيين `x` للأعداد الصحيحة من 0 إلى 9 باستخدام [`np.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html).

```{code-cell}
x = np.arange(10)
y = x ** 2
print(x)
print(y)
```

## حفظ مصفوفاتك باستخدام [`savez`](https://numpy.org/doc/stable/reference/generated/numpy.savez.html?highlight=savez#numpy.savez) من NumPy

الآن لديك مصفوفتان في مساحة عملك،

`x: [0 1 2 3 4 5 6 7 8 9]`

`y: [ 0  1  4  9 16 25 36 49 64 81]`

أول شيء ستفعله هو حفظهما في ملف كمصفوفات مضغوطة (zipped arrays) باستخدام [`savez`](https://numpy.org/doc/stable/reference/generated/numpy.savez.html?highlight=savez#numpy.savez). ستستخدم خيارين لتسمية المصفوفات في الملف،

1. `x_axis = x`: هذا الخيار يعين الاسم `x_axis` للمتغير `x`
2. `y_axis = y`: هذا الخيار يعين الاسم `y_axis` للمتغير `y`

```{code-cell}
np.savez("x_y-squared.npz", x_axis=x, y_axis=y)
```

## إزالة المصفوفات المحفوظة وتحميلها مرة أخرى باستخدام [`load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load) من NumPy

في دليل العمل الحالي الخاص بك، يجب أن يكون لديك ملف جديد بالاسم `x_y-squared.npz`. هذا الملف هو ثنائي مضغوط (zipped binary) للمصفوفتين، `x` و `y`. دعنا نمسح مساحة العمل ونحمل القيم مرة أخرى. يحتوي ملف `x_y-squared.npz` هذا على ملفين بتنسيق [NPY format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format). تنسيق NPY هو [تنسيق ثنائي أصلي (native binary format)](https://en.wikipedia.org/wiki/Binary_file). لا يمكنك قراءة الأرقام في محرر نصوص قياسي أو جدول بيانات.

1. إزالة `x` و `y` من مساحة العمل باستخدام [`del`](https://docs.python.org/3/reference/simple_stmts.html#del)
2. تحميل المصفوفات إلى مساحة العمل في قاموس (dictionary) باستخدام [`np.load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load)

لمعرفة المتغيرات الموجودة في مساحة العمل، استخدم أمر Jupyter/IPython "magic" [`whos`](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-whos).

```{code-cell}
del x, y
```

```{code-cell}
%whos
```

```{code-cell}
load_xy = np.load("x_y-squared.npz")

print(load_xy.files)
```

```{code-cell}
%whos
```

## إعادة تعيين مصفوفات NpzFile إلى `x` و `y`

لقد أنشأت الآن القاموس بنوع `NpzFile`. الملفات المضمنة هي `x_axis` و `y_axis` التي قمت بتعريفها في أمر `savez` الخاص بك. يمكنك إعادة تعيين `x` و `y` لملفات `load_xy`.

```{code-cell}
x = load_xy["x_axis"]
y = load_xy["y_axis"]
print(x)
print(y)
```

## نجاح (Success)
لقد قمت بإنشاء وحفظ وحذف وتحميل المتغيرات `x` و `y` باستخدام `savez` و `load`. عمل رائع.

## خيار آخر: الحفظ إلى csv قابل للقراءة البشرية (Another option: saving to human-readable csv)
دعنا نفكر في سيناريو آخر، تريد مشاركة `x` و `y` مع أشخاص آخرين أو برامج أخرى. قد تحتاج إلى ملف نصي قابل للقراءة البشرية (human-readable text file) يسهل مشاركته. بعد ذلك، ستستخدم [`savetxt`](https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html#numpy.savetxt) لحفظ `x` و `y` في ملف قيم مفصولة بفواصل (comma separated value file)، `x_y-squared.csv`. يتكون ملف csv الناتج من أحرف ASCII. يمكنك تحميل الملف مرة أخرى إلى NumPy أو قراءته ببرامج أخرى.

## إعادة ترتيب البيانات في مصفوفة ثنائية الأبعاد واحدة (Rearrange the data into a single 2D array)
أولاً، يجب عليك إنشاء مصفوفة ثنائية الأبعاد واحدة من مصفوفتيك أحاديتي الأبعاد. نوع ملف csv هو مجموعة بيانات على غرار جدول البيانات (spreadsheet-style dataset). يرتب csv الأرقام في صفوف - مفصولة بأسطر جديدة - وأعمدة - مفصولة بفواصل. إذا كانت البيانات أكثر تعقيدًا، على سبيل المثال، مصفوفات ثنائية الأبعاد متعددة أو مصفوفات ذات أبعاد أعلى، فمن الأفضل استخدام `savez`. هنا، ستستخدم دالتين من NumPy لتنسيق البيانات:

1. [`np.block`](https://numpy.org/doc/stable/reference/generated/numpy.block.html?highlight=block#numpy.block): هذه الدالة تلحق المصفوفات معًا في مصفوفة ثنائية الأبعاد

2. [`np.newaxis`](https://numpy.org/doc/stable/reference/constants.html?highlight=newaxis#numpy.newaxis): هذه الدالة تجبر المصفوفة أحادية الأبعاد على أن تكون متجه عمود ثنائي الأبعاد (2D column vector) بـ 10 صفوف وعمود واحد.

```{code-cell}
array_out = np.block([x[:, np.newaxis], y[:, np.newaxis]])
print("the output array has shape ", array_out.shape, " with values:")
print(array_out)
```

## حفظ البيانات في ملف csv باستخدام [`savetxt`](https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html#numpy.savetxt)

ستستخدم `savetxt` مع ثلاثة خيارات لجعل ملفك أسهل في القراءة:

* `X = array_out`: هذا الخيار يخبر `savetxt` بحفظ مصفوفتك ثنائية الأبعاد، `array_out`، في الملف `x_y-squared.csv`
* `header = 'x, y'`: هذا الخيار يكتب رأسًا قبل أي بيانات تسمي أعمدة csv
* `delimiter = ','`: هذا الخيار يخبر `savetxt` بوضع فاصلة بين كل عمود في الملف

```{code-cell}
np.savetxt("x_y-squared.csv", X=array_out, header="x, y", delimiter=",")
```

افتح الملف، `x_y-squared.csv`، وسترى ما يلي:

```{code-cell}
!head x_y-squared.csv
```

## مصفوفاتنا كملف csv (Our arrays as a csv file)

هناك ميزتان يجب أن تلاحظهما هنا:

1. تستخدم NumPy `#` لتجاهل العناوين عند استخدام `loadtxt`. إذا كنت تستخدم [`loadtxt`](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html) مع ملفات csv أخرى، يمكنك تخطي صفوف الرأس باستخدام `skiprows = <number_of_header_lines>`.
2. تمت كتابة الأعداد الصحيحة في التدوين العلمي (scientific notation). *يمكنك* تحديد تنسيق النص باستخدام خيار `savetxt`، [`fmt = `](https://docs.python.org/3/library/string.html#formatstrings)، لكنه سيظل مكتوبًا بأحرف ASCII. بشكل عام، لا يمكنك الحفاظ على نوع أرقام ASCII كـ `float` أو `int`.


الآن، احذف `x` و `y` مرة أخرى وقم بتعيينهما لأعمدتك في `x-y_squared.csv`.

```{code-cell}
del x, y
```

```{code-cell}
load_xy = np.loadtxt("x_y-squared.csv", delimiter=",")
```

```{code-cell}
load_xy.shape
```

```{code-cell}
x = load_xy[:, 0]
y = load_xy[:, 1]
print(x)
print(y)
```

## نجاح، ولكن تذكر أنواعك (Success, but remember your types)

عندما قمت بحفظ المصفوفات في ملف csv، لم تحافظ على نوع `int`. عند تحميل المصفوفات مرة أخرى إلى مساحة عملك، ستكون العملية الافتراضية هي تحميل ملف csv كمصفوفة نقطة عائمة ثنائية الأبعاد (2D floating point array) على سبيل المثال `load_xy.dtype == 'float64'` و `load_xy.shape == (10, 2)`.

+++

## تلخيص (Wrapping up)

في الختام، يمكنك إنشاء وحفظ وتحميل المصفوفات في NumPy. يجعل حفظ المصفوفات مشاركة عملك والتعاون أسهل بكثير. هناك طرق أخرى يمكن لـ Python من خلالها حفظ البيانات في ملفات، مثل [pickle](https://docs.python.org/3/library/pickle.html)، ولكن `savez` و `savetxt` ستلبي معظم احتياجات التخزين الخاصة بك لعمل NumPy المستقبلي والمشاركة مع الآخرين، على التوالي.

__الخطوات التالية__: يمكنك استيراد البيانات ذات القيم المفقودة من [الاستيراد باستخدام genfromtext](https://numpy.org/devdocs/user/basics.io.genfromtxt.html) أو معرفة المزيد حول الإدخال/الإخراج العام لـ NumPy باستخدام [قراءة وكتابة الملفات](https://numpy.org/devdocs/user/how-to-io.html).
