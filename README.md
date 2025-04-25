## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
 ```

import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
`![437547912-5ec31a5a-c83e-405b-816f-8f603db28de1](https://github.com/user-attachments/assets/0b6d35d8-4883-4b15-aa5d-d89563a7b5a3)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![437548034-3b07cb57-601c-413d-b932-acabe1a148ea](https://github.com/user-attachments/assets/073021f6-ac58-46ce-9a0f-cc99e1fe652c)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![437548164-56b5b965-539d-489e-b1f5-1d9e3aa53237](https://github.com/user-attachments/assets/1a8ea9de-f32b-49de-a9c1-0df37ff72ab4)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![437548303-c895d1fd-cb06-480e-bb98-4d2063c54526](https://github.com/user-attachments/assets/c4bb6e7a-1ae0-4454-b1e6-5094630cd390)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![437548525-c2277e6a-c543-4d22-8169-193b863adf0d](https://github.com/user-attachments/assets/45bcefb0-0f6f-411f-a68f-5de396df8f53)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![437548837-8eb08bce-55be-4e3f-8964-12c1eaf96127](https://github.com/user-attachments/assets/b57aee8f-e4dc-4e02-ba30-090562a91352)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![437549030-7dd3f373-add4-4b46-8a39-ce314686506c](https://github.com/user-attachments/assets/f4d81a2a-4dec-42d1-8129-c7a8a0096de8)

```
dfb=pd.concat([df,nd],axis=1)
dfb
```
![437549153-7af14088-d6de-4603-853b-9e3455797876](https://github.com/user-attachments/assets/a3dc6c64-3e8e-46bb-b6b7-61ef95c68362)
```

from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![437549298-2aaf0d41-b4e8-48e2-a2a0-d716b44a5563](https://github.com/user-attachments/assets/20dc012d-68b1-4072-99bb-45b33d2fd449)

```
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![437549374-57392653-6625-4773-adc4-df18e1133923](https://github.com/user-attachments/assets/80abf69c-e67e-40c2-8daa-7df2dcc01581)

```
df.skew()
```
![437549482-e6b2444f-a3cd-4321-9d65-9c637243b20d](https://github.com/user-attachments/assets/17a1d129-1de5-4d6b-9278-99a54aed73e7)
```
np.log(df["Highly Positive Skew"])
```
![437549599-7bae1473-e3ba-480e-9437-ffb0ea4db249](https://github.com/user-attachments/assets/591e85bc-84a3-4e4b-84a6-a4b2dfb10d57)
```
np.reciprocal(df["Moderate Positive Skew"])
```

![437549769-35d6d278-d941-4762-9ce9-612f189f66b0](https://github.com/user-attachments/assets/6e806b6d-9195-457c-ad2b-163c44e519f0)
```
np.sqrt(df["Highly Positive Skew"])
```

![437549873-f7b4ac03-ac0d-4c08-8c9b-25eb1691c354](https://github.com/user-attachments/assets/d959c2cf-c34a-4902-9689-00951455d019)
```
np.square(df["Highly Positive Skew"])
```

![437549996-13287e2f-81fd-460b-92a2-531dc72a473e](https://github.com/user-attachments/assets/75f21daf-7e63-4288-8401-d620b45b7945)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![437550107-a3cef940-8b2b-425e-8b2b-500e056cc879](https://github.com/user-attachments/assets/95c252d7-0b9b-4f47-bd73-4f702582e44a)
```
df.skew()
```
![437550216-d5504a6d-6963-4edc-bcf6-3f5f26dea404](https://github.com/user-attachments/assets/10f1bde3-8153-414b-ac90-e24025b2e632)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![437550312-da2b6581-0cec-4232-a39e-37383fd32541](https://github.com/user-attachments/assets/e1649da2-7a75-409f-b6d6-52124eefc67d)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![437550492-93f9be16-0805-4d9e-ab95-333f3c587625](https://github.com/user-attachments/assets/13392040-f87d-4dd3-9ea2-4e4eac327db9)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![437550604-78be949d-033f-4619-b7d9-fbec5595ec36](https://github.com/user-attachments/assets/c32ae466-771f-4139-b083-c3a68aae4772)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![437550696-35592df2-e252-45bc-9535-19ba2681b567](https://github.com/user-attachments/assets/ab679419-d16c-4268-b1ef-d566b06d2e54)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![437550824-9c8a92c6-e1bd-47e1-90a8-2cb944ecf316](https://github.com/user-attachments/assets/f75fc8a1-e506-4240-a1d1-bcbc4b4f6d93)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![437550911-c47720df-1c12-4cf0-8252-08fce5f5ccb5](https://github.com/user-attachments/assets/fce7251b-4753-4f95-a628-76aee9652be2)

```
dt=pd.read_csv("titanic_dataset.csv")
dt
```
![437551021-4aa92f3e-cb5a-4a48-a139-fcc8f2e4120d](https://github.com/user-attachments/assets/1cb1c810-b139-4b78-b247-83ff6a71263d)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
![437551108-3db6ef8d-c9b0-46db-802d-886e11e6c567](https://github.com/user-attachments/assets/63bbfa6a-b4d9-4001-959b-98bf7dab6dce)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![437551203-249280b8-aca1-45f6-a40e-da21dbc2d264](https://github.com/user-attachments/assets/ab617c71-b2fa-4cb7-b5ad-ce44b3221d53)








# RESULT:
      
         Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
