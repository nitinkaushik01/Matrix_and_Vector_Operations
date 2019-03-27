
# coding: utf-8

# In[2]:

import numpy as np


# ## Create the column vector

# In[4]:

col_vect = np.array([[0],[4],[7]])
col_vect


# ## Create the row vector

# In[5]:

row_vect = np.array([4,7,8])
row_vect


# ## Create the matrix

# In[6]:

mat = np.array([[7,2],[2,5]])
mat


# ## Matrix Shape, Size and Dimensions

# In[7]:

mat.ndim


# In[10]:

matx = np.array([[1,2,4],
                 [3,4,6],
                 [7,8,5],
                 [9,2,1]])
#Find no. of rows and columns
matx.shape


# In[11]:

#Find no. of elements
matx.size


# In[12]:

matx.ndim


# ## Matrix Addition and Substraction

# In[14]:

mat_1 = np.array([[0,1],
                  [4,7]])
mat_2 = np.array([[5,8],
                  [3,9]])

#Add two matrices
np.add(mat_1, mat_2)


# In[15]:

#Subtract one matrix from another two matrices
np.subtract(mat_1, mat_2)


# ## Transpose Matrix and Vector

# In[16]:

vec = np.array([4,7,6,8,0,3,9])
#Transpose the vector
vec.T


# In[17]:

mat_1.T


# ## Matrix Conversion - Dense to Sparse

# In[18]:

from scipy import sparse


# In[19]:

#Dense Matrix
dense_mat = np.array([[0,0],[0,1],[6,0]])


# In[21]:

#Convert Dense matrix to Sparse
sparse_mat = sparse.csr_matrix(dense_mat)
sparse_mat


# ## Select an element from Vector and Matrix

# In[7]:

vect = np.array([4,2,8,6,3,4])
vect[1]


# In[8]:

#Select an element from an array
mat_1 = np.array([[0,1],
                  [4,7]])
mat_1[1,1]


# ## Reshape a matrix

# In[10]:

matx = np.array([[1,2,4],
                 [3,4,6],
                 [7,8,5],
                 [9,2,1]])
matx.reshape(6,2)


# ## Invert an existing matrix

# In[14]:

mat_1 = np.array([[0,1],
                  [4,7]])
np.linalg.inv(mat_1)


# ## Get the diagonal of a matrix

# In[15]:

matx = np.array([[1,2,4],
                 [3,4,6],
                 [7,8,5]])
matx.diagonal()


# ## Retrieve max and min value of an element from the given matrix

# In[17]:

np.max(matx)


# In[18]:

np.min(matx)


# In[20]:

np.max(matx, axis=0)


# In[21]:

# matx = np.array([[1,2,4],
#                  [3,4,6],
#                  [7,8,5]])
np.max(matx, axis=1)


# ## Determinant of a Matrix

# In[22]:

matx = np.array([[1,2,4],
                 [3,4,6],
                 [7,8,5]])
np.linalg.det(matx)


# ## Faltten a Matrix

# In[23]:

matx = np.array([[1,2,4],
                 [3,4,6],
                 [7,8,5]])
matx.flatten()


# ## Trace of a Matrix

# In[25]:

matx = np.array([[1,2,4],
                 [3,4,6],
                 [7,8,5]])
matx.diagonal().sum()


# ## Rank of a Matrix

# In[27]:

matx = np.array([[1,2,4],
                 [3,4,6],
                 [7,8,5]])
np.linalg.matrix_rank(matx)


# ## Dot product of twp Vectors

# In[28]:

vect1 = np.array([4,6,8])
vect2 = np.array([2,3,5])
np.dot(vect1, vect2)


# ## Converting a Dictionary into Matrix

# In[33]:

from sklearn.feature_extraction import DictVectorizer
#Create a dictionary
dictionary_1 = [{'one':1, 'two':2},
               {'three':3, 'four':4},
               {'five':5, 'six':6}]

#Create a DictVectorizer object
DV = DictVectorizer(sparse=False)
#Convert dictionary to feature matrix
feat_mat = DV.fit_transform(dictionary_1)
feat_mat


# In[34]:

DV.get_feature_names()


# ## Mean, Variance and Standard Deviation of a Matrix

# In[35]:

#Mean
matx = np.array([[1,2,4],
                 [3,4,6],
                 [7,8,5]])
np.mean(matx)


# In[36]:

#Variance
np.var(matx)


# In[37]:

#Standard Deviation
np.std(matx)


# ## Element wise operations using inline function called Lambda

# In[39]:

matx = np.array([[1,2,4],
                 [3,4,6],
                 [7,8,5]])

addition = lambda i:i+5
add_5_vec = np.vectorize(addition)
add_5_vec(matx)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



