# COURSE 4: PYTHON FOR DATA SCIENCE

## WEEK 1: Python Basics
## WEEK 2: Python Data Structures

### 1. Tuples 

Tuples: Array which can contain different variable types

```python
tup = ('disco', 10, 1.2)
tup[0] = "disco"
tup[2] = 1.2
# In Python we can use negative index
# The last element has index -1
tup[2] = tup[-1] = 1.2
```
	
Typle operations:

Combine tuples:

```python
tup = ('disco', 10, 1.2)
tup2 = tup + ("jazz", 15)
# tup2 will be: ('disco', 10, 1.2, "jazz", 15)
```

Slicing tuples:

```python
tup = ('disco', 10, 1.2, "jazz", 15)
# tup[0:3] will take index 0-2: ('disco', 10, 1.2)
```

Get tuple length: use function `len()`
Tuples are *immutable*: we can't change them

```python
tup = ('disco', 10, 1.2, "jazz", 15)
tup[0] = "jazz" # Does not work
```

To manipulate tuples, we must create new tuples:

```python
tup = (10, 1.2)
tup2 = sorted(tup) # tup2 will be (1.2, 10)
```

*Nesting*: Tuple can contain other tuples:

```python
tup = (10, 1.2, ("pop", "rock"), (3,4), 5)
# -> tup[2][1] = "rock" 
```

### 2. Lists: Also ordered sequences like tuples

```python
ls = ['disco', 10, 1.2]	# See the difference with tuple?
```

Unlike tuple, lists are *mutable*:

```python
ls = ['disco', 10, 1.2]	
ls[0] = "rock" # Will work!
```

List can contain other list, and also tuple

```python
ls = ['disco', 10, 1.2, [1,2], ('A', 1)]	
```

We can also access negative index of list:

```python
ls = ['disco', 10, 1.2, [1,2], ('A', 1)]
# -> ls[-1] = ('A', 1) 	
```

Concatenate list:

```python
ls = ['disco', 10, 1.2]	
ls2 = ls + [[1,2], ('A', 1)] 
```
We can change directly the list (*mutability*):

```python
ls = [1, 2]	

# Change elements in list:
ls[0] = "rock" 		   # ls becomes ["rock", 2]
del(ls[0])			   # To delete elements in list	

# Add elements to list:	

ls.extend(["pop", 10]) # ls = [1, 2, "pop", 10]: Add all elements	
ls.append(["pop", 10]) # ls = [1, 2,  ["pop", 10]]: Add as only 1 element

# Split string -> create 1 list of two string elements
"hard rock".split() # becomes ["hard", "rock"]
"a,b".split(",")	# define delimiter to split
```

*Aliasing*: If list B is an alias of A, when change A, B will change as well
		
```python
A = [1, 2]	
B = A
A[0] = 0 # Therefore B becomes [0, 2] as a consequence
```

*Clone*: To ged rid of changement, we can use clone, any change in the original list will not affect the clone

```python
A = [1, 2]	
B = A[:]		# a clone
C = A.copy()	# also a clone
```


### 3. Dictionaries

* A dictionary has `key` (label) and `value`
* Keys have to be *immutable* and *unique*

```python
# A dictionary can contain various variable types
mydict = {"number": 1, "string": "harry", "list": [1,2,3], "tuple": (4,4,4)} 	

mydict["tuple"] = (4,4,4)
mydict["string"] = "harry"

# Add new entry to the exist dictionary:
mydict["new_key"] = [1,2]

# Delete an entry in the dictionary using del() 
del(mydict["new_key"])

# Check existence of entry in the dictionary with key
'string' in mydict 	# Check entry with key "string" -> return TRUE

# Get all keys or values in the dictionary:
mydict.key()		# Return ["number", "string", "list", "tuple"]
mydict.values()	# Idem
```


### 4. Sets
* Set are type of collections:
	* Like List/Tuples: Can contain different Python types
	* Unlike List/Tuples: 
		* Sets are unordered: 
		* Set only have unique elements

Example:
```python
# A dictionary can contain various variable types
set1 = {"pop", "rock", "pop"} # When set1 is create, duplicates will be removed
# set1 becomes {"pop", "rock"} 	
```

* *Type Casting*: Convert a list to a set: Using `set()`

Example:
```python
# Start with a list
ls = ["pop", "rock", "pop", 1990] # Duplicate is okay in List/Tuples

# Cast (convert) the list to a set:
ls = set(ls)  # ls is now {'pop', 'rock', 1990}
```

* Set Operations:

```python
myset = {"pop", "rock"}

# Add elements to set
myset.add("jazz")

# Remove elements from set
myset.remove("jazz")

# Verify existence of an element in set
"pop" in myset 	# TRUE

# Math operations between different sets:
set1 = {"pop", "rock"}
set2 = {"pop", "jazz"}

set1 & set2 			# Intersection of 2 sets: {"pop"}
set1.union(set2)		# Union of 2 sets: {"pop", "rock", "jazz"}
set1.issubset(set2)		# Check if set1 is a subset of set2: FALSE
```



## WEEK 3: Python Programming Fundamentals

### Video 1: Conditions and Branching
### Video 2: Loops

```python
range(10) # Get 0 to 9

for i, x in enumerate(['A', 'B', 'C']):
    print(i, x) # i returns index, x return value 

for i, x in enumerate(['A', 'B', 'C']):
    print(i, 2*x) 	# returns: 0 AA; 1 BB; 2 CC
```

### Video 3: Functions

```python
L = [1, 3, 2]
sorted(L) # sorted is a function and returns a new list, it does not change the list L
L.sort()  # this will change L	 

# When defining function, one can use "pass" to relax the function
def noWork():
	pass
# print(noWork()) -> return None
```

Collecting Arguments: All arguments can be combined into the function with * 
```python
def ArtisNames(*names):  # names = ("harry", "potter")
	for name in names:
		print (name)

ArtisNames("harry", "potter") # Arguments are collected automatically with *names
```

Local & Global Variables:
	* If a variable is not defined but used inside function, Python will look for it in the global variables

```python
def ACDC(y):  
	print (rating)		# rating not exist in local
	return (rating + y)

rating = 9 		# variable in global scope
z = ACDC(1)
print (rating)
```

Define Global Variable: One can also define global variable inside a function
```python
def ACDC(y):  
	global rating
	rating = 9		
	return (rating + y)
```


### Video 4: Objects and Classes

1. Object
	* Python has lots of data types:
		* int, float, string, list [], dictionary {"key": value}, bool: true/false
	* Each is an *Object*
	* Every object has:
		* A type
		* An internal data representation (a *blueprint*)
		* A set of procedures for interacting with the object (*methods*)
	* An object is an *instance* of a particular *type*
	* A method: changes or interacts with the object

Example:
```python
ls = [1,2,3]
ls.sort()		# a method
ls.reverse() 	# another method
```

2. Classes:

Example:
```python
class Circle(object):
	def __init__(self, radius, color):
		# __init__: constructor
		# self param refers to the newly created instance of the class
		self.radius = radius 
		self.color = color

# Create new object
c = Circle(10, "red")
print(c.radius)
print(c.color)

# We can change directly a paramenter of object
c.color = "blue"
print(c.color)
```

Define Method in Class:

```python
class Circle(object):
	def __init__(self, radius, color):
		# __init__: constructor
		# self param refers to the newly created instance of the class
		self.radius = radius 
		self.color = color

	def add_radius(self, r):
		# Forget about the constructor self when calling the method
		self.radius += r

	# One can also define directly the value for argument like this:
	def __init__(self, radius, color='red'):
		self.radius = radius 
		self.color = color

# Test
c = Circle(10, "red")
c.add_radius(2)
print(c.radius)
```

To get info about all attributes and methods associated with a class, use function `dir(ObjectName)`

```python
class Circle(object):
	def __init__(self, radius, color):
		# __init__: constructor
		# self param refers to the newly created instance of the class
		self.radius = radius 
		self.color = color

	def add_radius(self, r):
		# Forget about the constructor self when calling the method
		self.radius += r

# Test
c = Circle(10, "red")
c.add_radius(2)
print (dir(c))
```



## WEEK 4: Working with Data in Python

### Video 1: Reading Files with Open

```python
file1 = open("/resources/data/example2.txt", "w")
# Available modes: writing (w), reading (r), appending (a)

file1.name 		# get the name (string) of the file: '/resources/data/example2.txt'
file1.mode 		# get the mode: 'w'
file1.close()	# close the file

# Advantage of "with" statement: automatically close the file
with open("/resources/data/example2.txt", "r") as file1:
	file_stuff = file1.read()	# store the content as a string
	print (file_stuff)

print (file1.close())	# check if file1 is closed or not
print (file_stuff)		# print the content of file1

# Reading all lines of the file: 
with open("/resources/data/example2.txt", "r") as file1:
	file_stuff = file1.readlines() 	# read all lines, attention: readlines() <> readline()
	print (file_stuff)
	# file_stuff: ['this is line 1\n', 'this is line 2\n']	

# Reading line-by-line:
with open("/resources/data/example2.txt", "r") as file1:
	file_stuff = file1.readline() 	
	print (file_stuff)		# file_stuff: 'this is line 1\n'
	file_stuff = file1.readline() 	
	print (file_stuff)		# file_stuff: 'this is line 2\n'

# Alternative way to print all lines in file1:
with open("/resources/data/example2.txt", "r") as file1:
	for line in file1:
		print (line)

# Specify the number of characters to print:
with open("/resources/data/example2.txt", "r") as file1:
	file_stuff = file1.readlines(4) 	# file_stuff: 'this'

```

### Video 2: Writing Files with Open


```python
file1 = open("/resources/data/example2.txt", "w") # open the existed file or create new one

# Write each line to the file at once
with open("/resources/data/example2.txt", "w") as file1:
	file1.write("this is line 1")
	file1.write("this is line 2")
	file1.close()

# Write each element of a list to a file
lines = ["this is line 1\n", "this is line 2\n"]
with open("/resources/data/example2.txt", "w") as file1:
	for line in lines:
		file1.write(line)

# Mode "a" not to create new file, just to open exist file
with open("/resources/data/example2.txt", "a") as file1:
	file1.write("this is line 3")


# Copy content line-by-line from 1 file to another file:
with open("example1.txt", "r") as read_file: 		# mode "read"
	with open("example2.txt", "w") as write_file: 	# mode "write"
		for line in read_file:
			write_file.write(line)


```

### Video 2: Loading Data with Pandas


```python
import pandas as pd

# Open CSV file
csv_path = 'file1.csv' 
df = pd.read_csv(csv_path)	# df becomes a dataframe

# Similar to Excel file
xlsx_path = 'file1.xlsx' 
df = pd.read_excel(xlsx_path)	# df becomes a dataframe

# Read first 5 rows, like the head function in R
df.head()	

# Create a dataframe out of a dictionary:
songs = {'Album': ['Thriller', 'Back in Black'], \
		 'Release': [1982, 1980],
		 'Length': ['00:42:19', '00:42:11']}  # songs is a dictionary

df = pd.DataFrame(songs)	# The Keys of the dict become the column labels of DF

'''
df:
------------------------------------------
	Album			length 		Release
0	Thriller		00:42:19	1982
1	Back in Black	00:42:11	1980
------------------------------------------
'''

# Create sub DF from a DF:
x = df[['Length']]			# Get one column
y = df[['Album', 'Length']]	# Get multiple columns

'''
x:
------------------------------------------
	length 		
0	00:42:19	
1	00:42:11	
------------------------------------------
y:
------------------------------------------
	Album			length 		
0	Thriller		00:42:19	
1	Back in Black	00:42:11	
------------------------------------------
'''

# To access unique elements: ix method
df.ix[i,j]	# Access the (i+1)-th row, (j+1)-th column	
df.ix[0,'Album'] 	# Access the first row of column named 'Album'

# Slice data frame to smaller ones:
z = df.ix[0:2, 0:3]					# Access with indices
z = df.ix[0:2, 'Album':'Release']	# Access with both indices and col names
```

### Video 3: Pandas: Working with and Saving Data

```python
# List unique values in a column
df['Release'].unique()

# List values with condition
df['Release'] >= 1980	# Result will be array of boolean values (true/false)
df1 = df[df['Release'] >= 1980]	# Simple do this to get the values with condition

# Save as CSV:
df1.to_csv('file_name.csv')

```

### Video 4: One-Dimensional Numpy
* Basics and array creation
* Indexing and slicing
* Basic operations
* Universal functions

1. Basics and array creation

```python
a = ["0", 1, "two", "3", 4]		# a normal python list

# Using numpy: Much faster than regular python
import numpy as np

# Cast the list to a numpy array:
a = np.array([0, 1, 2, 3, 4])

type(a)		# obtain the type of a: numpy.ndarray
a.dtype		# obtain the data type of a: dtype('int64')
a.size		# 5
a.ndim 		# 1
a.shape 	# (5,)

# Another array
b = np.array([3.1, 11.2, 6.2, 4, 1.0])

type(b)		# numpy.ndarray
b.dtype 	# dtype('float64')
```


2. Indexing and slicing

- With numpy we can easy change the value of each element in an array

```python
a = np.array([0, 1, 2, 3, 4])

# Indexing
a[0] = 100 	# a:array([100, 1, 2, 3, 4])

# Slicing: Like python list, we don't count the element corr. to the last index
b = a[1:4]	# b:array([1, 2, 3])

# Change multiple values in array:
a[3:5] = 300,400	# a:array([100, 1, 2, 300, 400])
```

3. Basic operations

```python
# Vector operations:
u = [1,0]
v = [1,0]
z = []

for n,m in zip(u,v):
	z.append(n+m)	# z = [2,0]

# Using numpy: run much faster
u = np.array([1,0])
v = np.array([1,0])
z = u+v 	# z:array([2,0])

# Vector mutiplication:
z = 2*u  	# z = [2,0]
z = u*v 	# z = [1,0]

# Dot product: z = u'*v (like matrix multiplication)
u = np.array([1,2])
v = np.array([3,1])
z = np.dot(u,v)	# z = u' * v = 1*3+2*1 = 5

# Adding constant to a np array (called BROADCASTING)
u = np.array([1,2])
z = u + 1	# z:array([2,3])
```

4. Universal functions

```python
u = np.array([1,2])

u.mean()	# (1+2)/2 = 1.5
u.max()		# 2

# Numpy built-in function:
np.pi 		# pi
x = np.array([0, np.pi/2, np.pi])
y = np.sin(x) 	# y:array([0, 1, 1.2e-16])

# Numpy linspace:
np.linspace(-2, 2, num=5)	# array([-2, -1, 0, 1, 2])

# Plotting math functions
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x) 

import matplotlib.pylot as plt 
# matplotlib inline -> display the plot if using Jupyter notebook

plt.plot(x,y)	# plot y as function of x
```


### Video 5: Two-Dimensional Numpy
* Basics and array creation in 2D
* Indexing and slicing in 2D
* Basic operations in 2D


```python
a = [[11,12,13], [21,22,23], [31,32,33]]
A = np.array(a)

A.ndim 	# return 2
A.shape # return a tuple: (3,3) = (nrow, ncol)
A.size	# return total nb of elements 9

# Indexing
A[i][j]	# access row i, col j of A

# Slicing:
A[0, 0:2]	# get the first row and col from 0 to 1 -> array

# Operation:
X = np.array([[1,0],[0,1]])
Y = np.array([[2,1],[1,2]])
Z = X+Y 	# Z:array([[3,1],[1,3]])
Z = 2*Y 	# Z:array([[4,2],[2,4]])

# Matrix product
Z = X*Y 	# Z:array([[2,0],[0,2]]) 

# Matrix multiplication
X = np.array([[0,1,1],[1,0,1]])		# 2x3 matrix
Y = np.array([[1,1],[1,1],[-1,1]])	# 3x2 matrix
Z = np.dot(X,Y)	# 2x2 matrix: array([[0,0],[0,2]])
```


## WEEK 5: Fake Album Cover Game

### Video: Getting Started

### Peer-graded Assignment: Make a Fake Album Cover Game with Web Scraping

