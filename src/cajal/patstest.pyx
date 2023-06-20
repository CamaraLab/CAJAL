cdef extern from "EMD.h" :
    int myCMessage(int x,int y)
# Use myCMessage in a python function
def foo(int x,int y) :
    cdef int k
    print("Testing, testing")
    k = myCMessage(x,y)
    print("hello, world!")
    print(k)
