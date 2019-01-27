class Callable:
    def __init__(self, anycallable):
        self.__call__ = anycallable
class Static:
    static_object = 2
    def modify_static_object(x):
        Static.static_object +=x
        Static.another_object= x
    modify_static_object = Callable(modify_static_object)
    def createObject(x):
        Static.A = [1,2,3,4]
    createObject = Callable(createObject)
    #createObject = Callable(createObject)
##static = Static()
##static.createObject()

