---
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

# معالجة صور الأشعة السينية (X-ray image processing)

+++

يوضح هذا البرنامج التعليمي كيفية قراءة ومعالجة صور الأشعة السينية (X-ray images) باستخدام NumPy و imageio و Matplotlib و SciPy. ستتعلم كيفية تحميل الصور الطبية، والتركيز على أجزاء معينة، ومقارنتها بصرياً باستخدام مرشحات (filters) [Gaussian](https://en.wikipedia.org/wiki/Gaussian_filter) و [Laplacian-Gaussian](https://en.wikipedia.org/wiki/Laplace_distribution) و [Sobel](https://en.wikipedia.org/wiki/Sobel_operator) و [Canny](https://en.wikipedia.org/wiki/Canny_edge_detector) لـ كشف الحواف (edge detection).

يمكن أن يكون تحليل X-ray images جزءاً من تحليل البيانات و [سير عمل تعلم الآلة (machine learning workflow)](https://www.sciencedirect.com/science/article/pii/S235291481930214X) عندما تقوم، على سبيل المثال، ببناء خوارزمية تساعد في [كشف الالتهاب الرئوي (detect pneumonia)](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) كجزء من [مسابقة (competition)](https://www.kaggle.com/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen) على منصة [Kaggle](https://www.kaggle.com). في صناعة الرعاية الصحية، تعد معالجة الصور الطبية وتحليلها أمراً مهماً بشكل خاص عندما تشير التقديرات إلى أن الصور تمثل [90% على الأقل](https://www-03.ibm.com/press/us/en/pressrelease/51146.wss) من جميع البيانات الطبية.

ستعمل مع صور الأشعة من مجموعة بيانات [ChestX-ray8](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community) المقدمة من [المعاهد الوطنية للصحة (NIH)](http://nih.gov). تحتوي ChestX-ray8 على أكثر من 100,000 صورة X-ray مجهولة الهوية بتنسيق PNG لأكثر من 30,000 مريض. يمكنك العثور على ملفات ChestX-ray8 في [مستودع (repository)](https://nihcc.app.box.com/v/ChestXray-NIHCC) Box العام التابع لـ NIH في مجلد `/images`. (لمزيد من التفاصيل، راجع [الورقة البحثية (paper)](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) المنشورة في مؤتمر CVPR لرؤية الكمبيوتر في عام 2017).

لراحتك، تم حفظ عدد صغير من صور PNG في repository هذا البرنامج التعليمي تحت `tutorial-x-ray-image-processing/` ، نظرًا لأن ChestX-ray8 تحتوي على غيغابايت من البيانات وقد تجد صعوبة في تحميلها على دفعات.

![A series of 9 x-ray images of the same region of a patient's chest is shown with different types of image processing filters applied to each image. Each x-ray shows different types of biological detail.](_static/tutorial-x-ray-image-processing.png)

+++

## المتطلبات المسبقة (Prerequisites)

+++

يجب أن يكون لدى القارئ بعض المعرفة بلغة Python و مصفوفات NumPy (NumPy arrays) و Matplotlib. لتنشيط الذاكرة، يمكنك مراجعة دروس [Python](https://docs.python.org/dev/tutorial/index.html) و Matplotlib [PyPlot](https://matplotlib.org/tutorials/introductory/pyplot.html) و [البداية السريعة (quickstart)](https://numpy.org/devdocs/user/quickstart.html) لـ NumPy.

يتم استخدام الحزم (packages) التالية في هذا البرنامج التعليمي:

- [imageio](https://imageio.github.io) لقراءة وكتابة بيانات الصور. تعمل صناعة الرعاية الصحية عادةً مع تنسيق [DICOM](https://en.wikipedia.org/wiki/DICOM) للتصوير الطبي، ويجب أن يكون [imageio](https://imageio.readthedocs.io/en/stable/format_dicom.html) مناسباً تماماً لقراءة هذا التنسيق. للتبسيط، ستعمل في هذا الدرس مع ملفات PNG.
- [Matplotlib](https://matplotlib.org/) لـ تصور البيانات (data visualization).
- [SciPy](https://www.scipy.org) لـ معالجة الصور متعددة الأبعاد (multi-dimensional image processing) عبر [`ndimage`](https://docs.scipy.org/doc/scipy/reference/ndimage.html).

يمكن تشغيل هذا البرنامج التعليمي محلياً في بيئة معزولة، مثل [Virtualenv](https://virtualenv.pypa.io/en/stable/) أو [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). يمكنك استخدام [Jupyter Notebook أو JupyterLab](https://jupyter.org/install) لتشغيل كل خلية في المفكرة.

+++

## جدول المحتويات (Table of contents)

+++

1. فحص صورة X-ray باستخدام `imageio`
2. دمج الصور في مصفوفة متعددة الأبعاد لإظهار التقدم
3. كشف الحواف باستخدام مرشحات Laplacian-Gaussian و Gaussian gradient و Sobel و Canny
4. تطبيق الأقنعة (masks) على صور X-ray باستخدام `np.where()`
5. مقارنة النتائج

---

+++

## فحص صورة X-ray باستخدام `imageio` (Examine an X-ray with `imageio`)

+++

لنبدأ بمثال بسيط باستخدام صورة X-ray واحدة فقط من مجموعة بيانات ChestX-ray8.

تم تحميل الملف — `00000011_001.png` — وحفظه في مجلد `/tutorial-x-ray-image-processing`.

+++

**1.** قم بتحميل الصورة باستخدام `imageio`:

```{code-cell}
import os
import imageio

DIR = "tutorial-x-ray-image-processing"

xray_image = imageio.v3.imread(os.path.join(DIR, "00000011_001.png"))
```

**2.** تأكد من أن أبعادها (shape) هي 1024x1024 بكسل وأن المصفوفة تتكون من أعداد صحيحة 8-بت (8-bit integers):

```{code-cell}
print(xray_image.shape)
print(xray_image.dtype)
```

**3.** استورد `matplotlib` واعرض الصورة بتدرج رمادي (grayscale colormap):

```{code-cell}
import matplotlib.pyplot as plt

plt.imshow(xray_image, cmap="gray")
plt.axis("off")
plt.show()
```

## دمج الصور في مصفوفة متعددة الأبعاد لإظهار التقدم (Combine images into a multidimensional array to demonstrate progression)

+++

في المثال التالي، بدلاً من صورة واحدة، ستستخدم 9 صور X-ray بأبعاد 1024x1024 بكسل من مجموعة بيانات ChestX-ray8. وهي مرقمة من `...000.png` إلى `...008.png` ولنفترض أنها تنتمي لنفس المريض.

**1.** استورد NumPy، واقرأ كل صورة من صور X-ray، وأنشئ مصفوفة ثلاثية الأبعاد (three-dimensional array) حيث يتوافق البعد الأول مع رقم الصورة:

```{code-cell}
import numpy as np
num_imgs = 9

combined_xray_images_1 = np.array(
    [imageio.v3.imread(os.path.join(DIR, f"00000011_00{i}.png")) for i in range(num_imgs)]
)
```

**2.** تحقق من أبعاد مصفوفة صور X-ray الجديدة التي تحتوي على 9 صور مكدسة:

```{code-cell}
combined_xray_images_1.shape
```

لاحظ أن shape في البعد الأول يطابق `num_imgs` ، لذا يمكن تفسير مصفوفة `combined_xray_images_1` على أنها حزمة من الصور ثنائية الأبعاد (2D images).

**3.** يمكنك الآن عرض "تقدم الحالة الصحية" من خلال رسم كل إطار بجانب الآخر باستخدام Matplotlib:

```{code-cell} ipython3
fig, axes = plt.subplots(nrows=1, ncols=num_imgs, figsize=(30, 30))

for img, ax in zip(combined_xray_images_1, axes):
    ax.imshow(img, cmap='gray')
    ax.axis('off')
```

**4.** بالإضافة إلى ذلك، قد يكون من المفيد إظهار التقدم كرسوم متحركة (animation). لنقم بإنشاء ملف GIF باستخدام `imageio.mimwrite()` وعرض النتيجة في المفكرة:

```{code-cell} ipython3
GIF_PATH = os.path.join(DIR, "xray_image.gif")
imageio.mimwrite(GIF_PATH, combined_xray_images_1, format= ".gif", duration=1000)
```

مما يعطينا:
![An animated gif repeatedly cycles through a series of 8 x-rays, showing the same viewpoint of the patient's chest at different points in time. The patient's bones and internal organs can be visually compared from frame to frame.](tutorial-x-ray-image-processing/xray_image.gif)

## كشف الحواف باستخدام مرشحات Laplacian-Gaussian و Gaussian gradient و Sobel و Canny (Edge detection using the Laplacian-Gaussian, Gaussian gradient, Sobel, and Canny filters)

+++

عند معالجة البيانات الطبية الحيوية (biomedical data)، قد يكون من المفيد التأكيد على "الحواف" (edges) ثنائية الأبعاد للتركيز على سمات معينة في الصورة. للقيام بذلك، يمكن أن يكون استخدام [تدرجات الصور (image gradients)](https://en.wikipedia.org/wiki/Image_gradient) مفيداً بشكل خاص عند اكتشاف تغير شدة لون البكسل.

+++

### مرشح Laplace مع المشتقات الثانية لـ Gaussian (The Laplace filter with Gaussian second derivatives)

لنبدأ بمرشح [Laplace](https://en.wikipedia.org/wiki/Laplace_distribution) متعدد الأبعاد ("Laplacian-Gaussian") الذي يستخدم مشتقات [Gaussian](https://en.wikipedia.org/wiki/Normal_distribution) الثانية. تركز طريقة Laplacian هذه على البكسلات ذات التغير السريع في الشدة (intensity) وتدمج مع تنعيم Gaussian لـ [إزالة الضجيج (remove noise)](https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm). لنفحص كيف يمكن أن يكون ذلك مفيداً في تحليل صور X-ray ثنائية الأبعاد.

+++

- تنفيذ مرشح Laplacian-Gaussian بسيط نسبياً: 1) استيراد وحدة `ndimage` من SciPy؛ و 2) استدعاء [`scipy.ndimage.gaussian_laplace()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_laplace.html) مع معامل سيجما (sigma)، والذي يؤثر على الانحرافات المعيارية (standard deviations) لمرشح Gaussian (ستستخدم `1` في المثال أدناه):

```{code-cell}
from scipy import ndimage

xray_image_laplace_gaussian = ndimage.gaussian_laplace(xray_image, sigma=1)
```

اعرض X-ray الأصلية وتلك التي تم تطبيق مرشح Laplacian-Gaussian عليها:

```{code-cell}
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

axes[0].set_title("Original")
axes[0].imshow(xray_image, cmap="gray")
axes[1].set_title("Laplacian-Gaussian (edges)")
axes[1].imshow(xray_image_laplace_gaussian, cmap="gray")
for i in axes:
    i.axis("off")
plt.show()
```

### طريقة مقدار تدرج Gaussian (The Gaussian gradient magnitude method)

طريقة أخرى لـ edge detection يمكن أن تكون مفيدة هي مرشح Gaussian (التدرج). يقوم بحساب مقدار التدرج متعدد الأبعاد بمشتقات Gaussian ويساعد في إزالة مكونات الصورة [عالية التردد (high-frequency)](https://www.cs.cornell.edu/courses/cs6670/2011sp/lectures/lec02_filter.pdf).

+++

**1.** استدعِ [`scipy.ndimage.gaussian_gradient_magnitude()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_gradient_magnitude.html) مع معامل sigma (للانحرافات المعيارية؛ ستستخدم `2` في المثال أدناه):

```{code-cell}
x_ray_image_gaussian_gradient = ndimage.gaussian_gradient_magnitude(xray_image, sigma=2)
```

**2.** اعرض X-ray الأصلية وتلك التي تم تطبيق مرشح Gaussian gradient عليها:

```{code-cell}
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

axes[0].set_title("Original")
axes[0].imshow(xray_image, cmap="gray")
axes[1].set_title("Gaussian gradient (edges)")
axes[1].imshow(x_ray_image_gaussian_gradient, cmap="gray")
for i in axes:
    i.axis("off")
plt.show()
```

### عامل Sobel-Feldman (مرشح Sobel) (The Sobel-Feldman operator (the Sobel filter))

للعثور على مناطق ذات تردد مكاني عالٍ (الحواف أو خرائط الحواف) على طول المحاور الأفقية والرأسية لصورة X-ray ثنائية الأبعاد، يمكنك استخدام تقنية [عامل Sobel-Feldman (مرشح Sobel)](https://en.wikipedia.org/wiki/Sobel_operator). يطبق مرشح Sobel مصفوفتي نواة (kernel matrices) بحجم 3x3 — واحدة لكل محور — على X-ray من خلال [التفاف (convolution)](https://en.wikipedia.org/wiki/Kernel_(image_processing)#Convolution). بعد ذلك، يتم دمج هاتين النقطتين (التدرجات) باستخدام [نظرية فيثاغورس (Pythagorean theorem)](https://en.wikipedia.org/wiki/Pythagorean_theorem) لإنتاج مقدار التدرج (gradient magnitude).

+++

**1.** استخدم مرشحات Sobel — ([`scipy.ndimage.sobel()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sobel.html)) — على محوري x و y للصورة. ثم احسب المسافة بين `x` و `y` باستخدام Pythagorean theorem ودالة [`np.hypot()`](https://numpy.org/doc/stable/reference/generated/numpy.hypot.html) من NumPy للحصول على المقدار. أخيراً، قم بـ [تطبيع (normalize)](https://en.wikipedia.org/wiki/Normalization_%28image_processing%29) الصورة الناتجة لتكون قيم البكسل بين 0 و 255.

يتبع [تطبيع الصورة (Image normalization)](https://en.wikipedia.org/wiki/Normalization_%28image_processing%29) الصيغة التالية: `output_channel = 255.0 * (input_channel - min_value) / (max_value - min_value)`. نظراً لأنك تستخدم صورة grayscale، فأنت بحاجة إلى normalize قناة واحدة فقط.

```{code-cell}
x_sobel = ndimage.sobel(xray_image, axis=0)
y_sobel = ndimage.sobel(xray_image, axis=1)

xray_image_sobel = np.hypot(x_sobel, y_sobel)

xray_image_sobel *= 255.0 / np.max(xray_image_sobel)
```

**2.** قم بتغيير نوع بيانات مصفوفة الصورة الجديدة إلى تنسيق الفاصلة العائمة 32-بت (32-bit floating-point) لـ [جعلها متوافقة](https://github.com/matplotlib/matplotlib/issues/15432) مع Matplotlib:

```{code-cell}
print("The data type - before: ", xray_image_sobel.dtype)

xray_image_sobel = xray_image_sobel.astype("float32")

print("The data type - after: ", xray_image_sobel.dtype)
```

**3.** اعرض X-ray الأصلية وتلك التي تم تطبيق مرشح Sobel عليها. لاحظ استخدام تدرجات الألوان grayscale و `CMRmap` للمساعدة في التأكيد على الحواف:

```{code-cell}
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))

axes[0].set_title("Original")
axes[0].imshow(xray_image, cmap="gray")
axes[1].set_title("Sobel (edges) - grayscale")
axes[1].imshow(xray_image_sobel, cmap="gray")
axes[2].set_title("Sobel (edges) - CMRmap")
axes[2].imshow(xray_image_sobel, cmap="CMRmap")
for i in axes:
    i.axis("off")
plt.show()
```

### مرشح Canny (The Canny filter)

يمكنك أيضاً التفكير في استخدام مرشح معروف آخر لـ edge detection يسمى [مرشح Canny (Canny filter)](https://en.wikipedia.org/wiki/Canny_edge_detector).

أولاً، تقوم بتطبيق مرشح Gaussian لإزالة الضجيج في الصورة. في هذا المثال، تستخدم مرشح [Fourier](https://en.wikipedia.org/wiki/Fourier_transform) الذي ينعم X-ray من خلال عملية convolution. بعد ذلك، تقوم بتطبيق [مرشح Prewitt (Prewitt filter)](https://en.wikipedia.org/wiki/Prewitt_operator) على كل من محوري الصورة للمساعدة في اكتشاف بعض الحواف. في النهاية، تقوم بحساب المقدار بين التدرجين باستخدام Pythagorean theorem وتقوم بـ normalize للصور كما فعلنا سابقاً.

+++

**1.** استخدم مرشحات Fourier من SciPy — [`scipy.ndimage.fourier_gaussian()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.fourier_gaussian.html) — مع قيمة `sigma` صغيرة لإزالة بعض الضجيج. ثم احسب تدرجين باستخدام [`scipy.ndimage.prewitt()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.prewitt.html). بعد ذلك، قم بقياس المسافة بين التدرجات باستخدام `np.hypot()`. أخيراً، قم بـ normalize للصورة الناتجة.

```{code-cell}
fourier_gaussian = ndimage.fourier_gaussian(xray_image, sigma=0.05)

x_prewitt = ndimage.prewitt(fourier_gaussian, axis=0)
y_prewitt = ndimage.prewitt(fourier_gaussian, axis=1)

xray_image_canny = np.hypot(x_prewitt, y_prewitt)

xray_image_canny *= 255.0 / np.max(xray_image_canny)

print("The data type - ", xray_image_canny.dtype)
```

**2.** ارسم صورة X-ray الأصلية وتلك التي تم اكتشاف حوافها بمساعدة تقنية Canny filter. يمكن التأكيد على الحواف باستخدام تدرجات الألوان `prism` و `nipy_spectral` و `terrain` في Matplotlib.

```{code-cell}
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 15))

axes[0].set_title("Original")
axes[0].imshow(xray_image, cmap="gray")
axes[1].set_title("Canny (edges) - prism")
axes[1].imshow(xray_image_canny, cmap="prism")
axes[2].set_title("Canny (edges) - nipy_spectral")
axes[2].imshow(xray_image_canny, cmap="nipy_spectral")
axes[3].set_title("Canny (edges) - terrain")
axes[3].imshow(xray_image_canny, cmap="terrain")
for i in axes:
    i.axis("off")
plt.show()
```

## تطبيق الأقنعة على صور X-ray باستخدام `np.where()` (Apply masks to X-rays with `np.where()`)

+++

لحجب بكسلات معينة فقط في صور X-ray للمساعدة في اكتشاف سمات معينة، يمكنك تطبيق الأقنعة (masks) باستخدام دالة [`np.where(condition, x, y)`](https://numpy.org/doc/stable/reference/generated/numpy.where.html) من NumPy التي تعيد `x` عندما يكون الشرط `True` و `y` عندما يكون `False`.

يمكن أن يكون تحديد مناطق الاهتمام (regions of interest) — مجموعات معينة من البكسلات في الصورة — مفيداً، وتعمل الأقنعة كمصفوفات بولية (boolean arrays) لها نفس أبعاد الصورة الأصلية.

+++

**1.** استخرج بعض الإحصائيات الأساسية حول قيم البكسل في صورة X-ray الأصلية:

```{code-cell}
print("The data type of the X-ray image is: ", xray_image.dtype)
print("The minimum pixel value is: ", np.min(xray_image))
print("The maximum pixel value is: ", np.max(xray_image))
print("The average pixel value is: ", np.mean(xray_image))
print("The median pixel value is: ", np.median(xray_image))
```

**2.** نوع بيانات المصفوفة هو `uint8` وتشير نتائج القيم الدنيا/القصوى إلى أن جميع الألوان الـ 256 (من `0` إلى `255`) مستخدمة. لنقم بتصور *توزيع شدة البكسل (pixel intensity distribution)* للصورة الأصلية باستخدام `ndimage.histogram()` و Matplotlib:

```{code-cell}
pixel_intensity_distribution = ndimage.histogram(
    xray_image, min=np.min(xray_image), max=np.max(xray_image), bins=256
)

plt.plot(pixel_intensity_distribution)
plt.title("Pixel intensity distribution")
plt.show()
```

كما يشير pixel intensity distribution، هناك العديد من قيم البكسل المنخفضة (بين 0 و 20 تقريباً) والعالية جداً (بين 200 و 240 تقريباً).

**3.** يمكنك إنشاء أقنعة شرطية مختلفة باستخدام `np.where()` — على سبيل المثال، لنحتفظ فقط بتلك القيم من الصورة التي تتجاوز فيها البكسلات عتبة (threshold) معينة:

```{code-cell}
# العتبة هي "أكبر من 150"
# أعد الصورة الأصلية إذا كان الشرط صحيحاً، و `0` خلاف ذلك
xray_image_mask_noisy = np.where(xray_image > 150, xray_image, 0)

plt.imshow(xray_image_mask_noisy, cmap="gray")
plt.axis("off")
plt.show()
```

```{code-cell}
# العتبة هي "أكبر من 150"
# أعد `1` إذا كان الشرط صحيحاً، و `0` خلاف ذلك
xray_image_mask_less_noisy = np.where(xray_image > 150, 1, 0)

plt.imshow(xray_image_mask_less_noisy, cmap="gray")
plt.axis("off")
plt.show()
```

## مقارنة النتائج (Compare the results)

+++

لنقم بعرض بعض نتائج صور X-ray المعالجة التي عملت عليها حتى الآن:

```{code-cell}
fig, axes = plt.subplots(nrows=1, ncols=9, figsize=(30, 30))

axes[0].set_title("Original")
axes[0].imshow(xray_image, cmap="gray")
axes[1].set_title("Laplace-Gaussian (edges)")
axes[1].imshow(xray_image_laplace_gaussian, cmap="gray")
axes[2].set_title("Gaussian gradient (edges)")
axes[2].imshow(x_ray_image_gaussian_gradient, cmap="gray")
axes[3].set_title("Sobel (edges) - grayscale")
axes[3].imshow(xray_image_sobel, cmap="gray")
axes[4].set_title("Sobel (edges) - hot")
axes[4].imshow(xray_image_sobel, cmap="hot")
axes[5].set_title("Canny (edges) - prism)")
axes[5].imshow(xray_image_canny, cmap="prism")
axes[6].set_title("Canny (edges) - nipy_spectral)")
axes[6].imshow(xray_image_canny, cmap="nipy_spectral")
axes[7].set_title("Mask (> 150, noisy)")
axes[7].imshow(xray_image_mask_noisy, cmap="gray")
axes[8].set_title("Mask (> 150, less noisy)")
axes[8].imshow(xray_image_mask_less_noisy, cmap="gray")
for i in axes:
    i.axis("off")
plt.show()
```

## الخطوات التالية (Next steps)

+++

إذا كنت ترغب في استخدام عيناتك الخاصة، يمكنك استخدام [هذه الصورة](https://openi.nlm.nih.gov/detailedresult?img=CXR3666_IM-1824-1001&query=chest%20infection&it=xg&req=4&npos=32) أو البحث عن صور أخرى متنوعة في قاعدة بيانات [_Openi_](https://openi.nlm.nih.gov). تحتوي Openi على العديد من الصور الطبية الحيوية ويمكن أن تكون مفيدة بشكل خاص إذا كان لديك عرض نطاق ترددي منخفض و/أو كنت مقيداً بكمية البيانات التي يمكنك تحميلها.

لمعرفة المزيد حول معالجة الصور في سياق بيانات الصور الطبية الحيوية أو ببساطة edge detection، قد تجد المواد التالية مفيدة:

- [معالجة وتقسيم DICOM في Python](https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/) باستخدام Scikit-Image و pydicom (Radiology Data Quest)
- [التلاعب بالصور ومعالجتها باستخدام Numpy و Scipy](https://scipy-lectures.org/advanced/image_processing/index.html) (Scipy Lecture Notes)
- [قيم الشدة (Intensity values)](https://s3.amazonaws.com/assets.datacamp.com/production/course_7032/slides/chapter2.pdf) (عرض تقديمي، DataCamp)
- [كشف الكائنات باستخدام Raspberry Pi و Python](https://makersportal.com/blog/2019/4/23/image-processing-with-raspberry-pi-and-python-part-ii-spatial-statistics-and-correlations) (Maker Portal)
- [تحضير وتقسيم بيانات X-ray](https://www.kaggle.com/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen) باستخدام التعلم العميق (مفكرة Jupyter مستضافة على Kaggle)
- [تصفية الصور (Image filtering)](https://www.cs.cornell.edu/courses/cs6670/2011sp/lectures/lec02_filter.pdf) (شرائح محاضرة، CS6670: رؤية الكمبيوتر، جامعة كورنيل)
- [كشف الحواف في Python](https://scikit-image.org/docs/0.25.x/auto_examples/edges/plot_edge_filter.html)
- [كشف الحواف](https://datacarpentry.github.io/image-processing/edge-detection.html) باستخدام Scikit-Image (Data Carpentry)
- [تدرجات الصور وتصفية التدرج](https://www.cs.cmu.edu/~16385/s17/Slides/4.0_Image_Gradients_and_Gradient_Filtering.pdf) (شرائح محاضرة، 16-385 رؤية الكمبيوتر، جامعة كارنيجي ميلون)
