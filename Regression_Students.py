{"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},"cells":[{"cell_type":"markdown","source":["Импорт библиотек"],"metadata":{"id":"9eFThHrPlCFg"}},{"cell_type":"code","execution_count":null,"metadata":{"id":"zqaeN2z8v3qR"},"outputs":[],"source":["import pandas as pd\n","import seaborn as sns\n","from sklearn.model_selection import train_test_split \n","from sklearn.decomposition import PCA\n","from sklearn.linear_model import LinearRegression\n","from sklearn.metrics import r2_score"]},{"cell_type":"markdown","source":["Всего датасет содержит 159 записей. Мы предварительно зарезервировали часть датасета для задания по преодолению бейслайна (baseline). В настоящем задании подразумевается работа с датасетом, содержащим 119 записей."],"metadata":{"id":"QTq1SRHEzF6J"}},{"cell_type":"markdown","source":["# 0. Применение полиномиальной регрессии для предсказания непрерывного параметра"],"metadata":{"id":"bjyhxYK-eoem"}},{"cell_type":"markdown","source":["Представленный набор данных — это набор данных о семи различных типах рыб, продаваемых в некоторой рыбной лавке. Наша цель заключается в том, чтобы предсказать массу рыбы по каким-то косвенным признакам, известным о рыбе. Сами признаки, быть может, нужно синтезировать из тех, что известны."],"metadata":{"id":"VpkXz4mpeygZ"}},{"cell_type":"markdown","source":["При помощи <code>train_test_split()</code> разбейте набор данных на обучающую и тестовую выборки с параметрами, указанными в вашем задании. Используйте стратификацию по колонке <code>Species</code>. Стратификация позволит сохранить доли представленных объектов (по представителям типов рыб) в тренировочной и тестовой выборках."],"metadata":{"id":"K497OQtR0cbN"}},{"cell_type":"code","source":["# < ENTER YOUR CODE HERE > "],"metadata":{"id":"KXoU2OWH0fn5"},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":["Вычислите выборочное среднее колонки <code>Width</code> полученной тренировочной выборки."],"metadata":{"id":"euRpYppY1xuR"}},{"cell_type":"code","source":["# < ENTER YOUR CODE HERE > "],"metadata":{"id":"3q_2UVQ_2Avj"},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":["# 1. Построение базовой модели\n","\n","Избавьтесь от категориальных признаков и обучите модель линейной регрессии (<code>LinearRegression()</code>) на тренировочном наборе данных. Выполните предсказания для тестового набора данных. Оцените модель при помощи метрики <code>r2_score()</code>."],"metadata":{"id":"cAQId_950UCZ"}},{"cell_type":"code","source":["# < ENTER YOUR CODE HERE > "],"metadata":{"id":"J9QyfzCx5FrA"},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":["# 2. Добавление предварительной обработки признаков."],"metadata":{"id":"qPEFdTMe56b9"}},{"cell_type":"markdown","source":["## Использование PCA\n","\n","При помощи, например, <code>sns.heatmap()</code>, постройте матрицу корреляций признаков тренировочного набора данных и определите тройку наиболее коррелированных между собой признаков."],"metadata":{"id":"kKrNXjxTCS1D"}},{"cell_type":"code","source":["# < ENTER YOUR CODE HERE > "],"metadata":{"id":"HFtdKOdG62hS"},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":["Линейные модели достаточно плохо реагируют на коррелированные признаки, поэтому от таких признаков имеет смысл избавиться еще до начала обучения."],"metadata":{"id":"Cx1RHggagRLr"}},{"cell_type":"markdown","source":["Для уменьшения количества неинформативных измерений используйте метод главных компонент. \n","\n","1) Примените метод главных компонент (<code>PCA(n_components=3, svd_solver='full')</code>) для трех найденных наиболее коррелированных признаков. \n","\n","2) Вычислите долю объясненной дисперсии при использовании только первой главной компоненты. \n","\n","3) Замените три наиболее коррелированных признака на новый признак <code>Lengths</code>, значения которого совпадают со значениями счетов первой главной компоненты."],"metadata":{"id":"e6AquRcu7Gvb"}},{"cell_type":"code","source":["# < ENTER YOUR CODE HERE > "],"metadata":{"id":"VlPLHO2U8PiA"},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":["Примените полученное преобразование для тех же признаков в тестовом наборе данных. Обратите внимание, что заново обучать преобразование `PCA` не нужно. Аналогично предыдущему этапу замените три рассмотренных признака на один."],"metadata":{"id":"kDrK2-Nd_ogF"}},{"cell_type":"code","source":["# < ENTER YOUR CODE HERE > "],"metadata":{"id":"xEOAkMFf-T5m"},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":["Обучите базовую модель линейной регресси на полученных тренировочных данных, снова выбросив категориальные признаки. Выполните предсказания для тестовых данных, оцените при помощи <code>r2_score()</code>."],"metadata":{"id":"NGiPC64DAmRb"}},{"cell_type":"code","source":["# < ENTER YOUR CODE HERE > "],"metadata":{"id":"EVu8wL6QA1Ci"},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":["Видно, что точность значительно не изменилась."],"metadata":{"id":"BFPWkxOnieem"}},{"cell_type":"markdown","source":["## Модификация признаков"],"metadata":{"id":"hJm3yq--BcT7"}},{"cell_type":"markdown","source":["Постройте графики зависимостей признаков от целевой переменной, например, при помощи <code>sns.pairplot()</code>."],"metadata":{"id":"T11p0ltmEFwR"}},{"cell_type":"code","source":["# < ENTER YOUR CODE HERE > "],"metadata":{"id":"0rifKCoWEQYj"},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":["Видно, что масса, вообще говоря, нелинейно зависит от остальных параметров. Значит, чтобы линейная модель хорошо справлялась с предсказанием, признаки имеет смысл преобразовать так, чтобы зависимость стала более похожей на линейную. Но как придумать такую зависимость?"],"metadata":{"id":"L-v73i1vhi_9"}},{"cell_type":"markdown","source":["Логично предположить, что масса рыбы должна каким-то гладким образом зависеть от остальных параметров, отвечающих так или иначе за размеры. Если впомнить, что масса — это произведение плотности на объем, то\n","\n","$$\n","m = \\rho \\cdot V.\n","$$\n","\n","Допустим, что средняя плотность у всех рыб одинаковая, и вспомним, что при гомотетии объем объекта зависит от линейных размеров как куб, тогда получим\n","\n","$$\n","m\\sim V\\sim d^3\n","$$\n","\n","Все признаки тренировочного и тестового наборов данных, отвечающие так или иначе за размеры (<code>Height, Width, Lengths</code>), возведите в третью степень, и проверьте, стала ли зависимость массы от этих признаков похожа на линейную."],"metadata":{"id":"sSIndS-yCP6P"}},{"cell_type":"code","source":["# < ENTER YOUR CODE HERE > "],"metadata":{"id":"DOK0OyLVEmrh"},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":["Введите выборочное среднее колонки <code>Width</code> тренировочного набора данных после возведения в куб."],"metadata":{"id":"Jm-1XAOpE_Dc"}},{"cell_type":"code","source":["# < ENTER YOUR CODE HERE > "],"metadata":{"id":"5-2zYkgWGCFN"},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":["Выберите изображения, соответствующие зависимости <code>Weight</code> от <code>Width</code> до преобразования и после."],"metadata":{"id":"PCKM-CcQjfH5"}},{"cell_type":"markdown","source":["Обучите базовую модель линейной регресси на полученных тренировочных данных, снова выбросив категориальные признаки. Выполните предсказания для тестовых данных, оцените при помощи `r2_score()`."],"metadata":{"id":"NZtrdFamiTl1"}},{"cell_type":"code","source":["# < ENTER YOUR CODE HERE > "],"metadata":{"id":"ybKUGN-GGUoH"},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":["Обратите внимание на то, как такая нехитрая работа с признаками помогла разительно улучшить точность модели!"],"metadata":{"id":"WReH7dKZirOt"}},{"cell_type":"markdown","source":["## Добавление категориальных признаков"],"metadata":{"id":"MesFzgX1Sqgz"}},{"cell_type":"markdown","source":["Произведите <code>one-hot</code> кодировние категориального признака `Species`, например, с помощью <code>pd.get_dummies()</code>.\n","\n","Обучите модель линейной регресси на полученных тренировочных данных. Выполните предсказания для тестовых данных, оцените модель при помощи <code>r2_score()</code>.\n","\n","<b>Примечание</b>: Мы специально использовали стратифицированное разделение, чтобы все значения категориального признака <code>Species</code> присутствовали во всех наборах данных. Но такое возможно не всегда. Про то, как с этим бороться можно почитать, [например, здесь](https://predictivehacks.com/?all-tips=how-to-deal-with-get_dummies-in-train-and-test-dataset)."],"metadata":{"id":"VThVRIkvSuV0"}},{"cell_type":"code","source":["# < ENTER YOUR CODE HERE > "],"metadata":{"id":"AcJDFGFLSpzT"},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":["И снова точность возрасла."],"metadata":{"id":"bBDsQEGvjMV2"}},{"cell_type":"markdown","source":["Как можно увидеть, после `one-hot` кодирования признаки стали коррелированы. От этого можно избавиться, например, при помощи параметра `drop_first=True`. Заново обучите модель после исправления этого недочета. Выполните предсказания для тестовых данных, оцените модель при помощи <code>r2_score()</code>."],"metadata":{"id":"vciWw1gajo5H"}},{"cell_type":"code","source":["# < ENTER YOUR CODE HERE > "],"metadata":{"id":"zV6u-Ha1kH3p"},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":["На таком сравнительно небольшом наборе данных, впрочем, разницы мы не видим."],"metadata":{"id":"ABS2Tw6tkSW5"}}]}