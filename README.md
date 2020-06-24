# Movies-Genre-Classification :notebook:
In this repo i have created a Movies Genre Classification project in machine learning using NLP, and i am using [nltk](https://pypi.org/project/nltk/) Library for NLP.

# Dependentias :warning:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
```

## ScreenShot :camera_flash:
![](https://github.com/yogeshnile/Movies-Genre-Classification/blob/master/Images/1.png) <br>
  - WordCloud
<br>

![](https://github.com/yogeshnile/Movies-Genre-Classification/blob/master/Images/2.png)       ![](https://github.com/yogeshnile/Movies-Genre-Classification/blob/master/Images/3.png)
![](https://github.com/yogeshnile/Movies-Genre-Classification/blob/master/Images/4.png)


## Bug / Feature Request :man_technologist:
If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here](https://github.com/yogeshnile/Movies-Genre-Classification/issues/new) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/yogeshnile/Movies-Genre-Classification/issues/new). Please include sample queries and their corresponding results.

## Follow on a Social Media :busts_in_silhouette:
- [LinkedIn](https://bit.ly/2Ky3ho6)
- [Instagram](https://bit.ly/3b9Qeo4)
- [Instagram Personal](https://bit.ly/32SXHV0)
- [Twitter](https://bit.ly/3dbLJLC)
