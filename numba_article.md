In this article , we will present the package Numba that we can use to boost a python code .

When it is  too difficult or impossible to vectorize an  algorithm or a math-heavy computational process,  you often need to use Python loops. However, Python loops are slow. 

A common method for solving this speed problem is to re-write your code in something fast like C++ and then throw a Python wrapper on top. That’ll get you the speed of C++ while maintaining the ease of using Python in your main application.
The challenge with that of course is that you’ll have to re-write the code in C++ or Cython ; that’s a pretty time consuming process.
Fortunately, The Python library  **Numba** gives us an easy way around that challenge — free speed ups without having to write any code other than Python!


### 1. Numba overview

Numba is a package created by Continuum Analytics (http://www.continuum.io) . Numba takes pure Python code and translates it automatically (just-in-time) into optimized machine code. With the help of Numba you can speed up all of your calculation focused and computationally heavy python functions

In practice, this means that we can write a non-vectorized function in pure Python, using for loops, and have this function vectorized automatically by using a single decorator. Performance speedups when compared to pure Python code can reach several orders of magnitude and may even outmatch manually-vectorized NumPy code. 

### 2.  How does Numba work?

Numba allows the compilation of selected portions of pure Python code to native code, and generates optimized machine code using the  [LLVM](http://llvm.org/)  compiler infrastructure.
With a few simple annotations, array-oriented and math-heavy Python code can be just-in-time (JIT) optimized to achieve performance similar to C, C++ and Fortran, without having to switch languages or Python interpreters.


Here is how the code is compiled:

![](https://raw.githubusercontent.com/ContinuumIO/gtc2017-numba/6ddaeec9baecf07df1a22e3e685d5f6e3b4f33d9/img/numba_flowchart.png)

First, Python function is taken, optimized and is converted into numba’s intermediate representation, then after type inference which is like numpy’s type inference (so python float is a float64) it is converted into LLVM interpretable code. This code is then fed to LLVM’s just-in-time compiler to give out a machine code.
Numba works at the function level. From a function, Numba can generate native code for that function as well as the wrapper code needed to call it directly from Python. This compilation is done on-the-fly and in-memory.

### 3.  Example 

Let's try Numba on a simple program : 

We will create a function that computes the pairwise euclidian distance between all vectors in a matrix . Here is the naive functions that we can implement:
````

def  pure_euclidean_distance(x1, x2):
	x1 = np.asarray(x1)
	x2 = np.asarray(x2)
	return np.sqrt(np.sum((x1 - x2) **  2))

def  pairwise(X, metric ):

	X = np.asarray(X)
	n_samples, n_dim = X.shape
	D = np.empty((n_samples, n_samples))

	for i in  range(n_samples):
		for j in  range(n_samples):
			D[i, j] = metric(X[i], X[j])
	  
	return D
````


Let's run our pairwise function on a matrix of 5000 vectors and 100 features ( components )  :
````
X = np.random.random((5000, 100)) 

print("---- Pairwise distance with pure python ---")
result_pure_python = pairwise(X,metric=pure_euclidean_distance)
````

 - #####  Function pairwise time: -- Wall time :  4min 50s ####

No let's do our magic and make use of Numba .This can be done simply by writing the the @autojit decorator before the euclidian function definition and make sure to import numba with the relevant decorator.

````
from numba.decorators import autojit

@autojit
def  numba_euclidean_distance(x1, x2):
	x1 = np.asarray(x1)
	x2 = np.asarray(x2)
return np.sqrt(np.sum((x1 - x2) **  2))

print("---- Pairwise distance with numba ---")
result_with_numba = pairwise(X,metric=numba_euclidean_distance)
````
 - #####  Function pairwise time: -- Wall time :  40s ####

As we can see , we have already enhanced significantly the performance by using a simple decorator.
It's not all , there is another simple trick to boost loops. In fact, instead of using the native  range () function of python , we can use prange() of Numba  declaring that there are no dependencies between different loop iterations. In that situation, the compiler is free to break the range into chunks and execute them in different threads. This is a very simple, but powerful abstraction. Below the final code : 
````
from numba.decorators import autojit
import numba

@autojit
def  numba_euclidean_distance(x1, x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return np.sqrt(np.sum((x1 - x2) **  2))


@numba.jit(nopython=True, parallel=True)
def  numba_pairwise(X, metric ):

    X = np.asarray(X)
    n_samples, n_dim = X.shape
    D = np.empty((n_samples, n_samples))

    for i in  numba.prange(n_samples):
        for j in  numba.prange(n_samples):
            D[i, j] = metric(X[i], X[j])
 
    return D
````
 - #####  Function pairwise time: -- Wall time :  8s  ####

**Conclusion:** 

Pairwise distance computation takes 4min 50 seconds with the standard Python implementation and only 8 seconds with numba decorators . Not too bad for something that doesn't take almost ANY code changes.

###  4. Discussion

#####  - Is it always super fast?

Numba is going to be most effective when applied in either of these areas:

-   Places where Python code is slower than C code (typically loops)
-   Places where the same operation is applied to an area (i.e the same operation on many elements)

Outside of those areas, Numba probably won’t be giving you much speed. Since the advantage that comes with converting to the lower level code is gone in that case.. Pandas for example are not helped by numba, and using numba will actually slow panda code down a little ( consider using Pandarallel or Dask in this scenario ). So always test numba to see which functions it can speed up (and consider breaking larger functions down into smaller ones so that blocks that can use numba may be separated out).



##### -  Advantages of Numba:

-   Ease of use
-   Automatic parallelization
-   Support for numpy operations and objects
-   GPU support

##### - Disadvantages of Numba:

-   Many layers of abstraction make it very hard to debug and optimize
-   There is no way to interact with Python and its modules in  `nopython`  mode
-   Limited support for classes







Webography : 

- https://numba.pydata.org/
- https://christophdeil.com/download/2019-07-11_Christoph_Deil_Numba.pdf
- https://www.audiolabs-erlangen.de/resources/MIR/FMP/B/B_PythonNumba.html
- https://en.wikipedia.org/wiki/Numba
