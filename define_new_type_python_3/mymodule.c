/*
<keywords>
test, c, python3, python, define, new, class, type, from, c, extension
</keywords>
<description>
a demonstration on how to create a module that defines a new type in C in
 python 3.
</description>
<seealso>
 https://docs.python.org/3/extending/newtypes.html
</seealso>
*/


/*
  define a python module that defines a new type with pure C
 */
#include <Python.h>

typedef struct {

    /* this macro defines a refcount and a pointer to a type obj */
    PyObject_HEAD

    /* Type-specific fields go here */
} my_module_MyNewClassObject;


static PyTypeObject my_module_MyNewClassType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "my_module.MyNewClasss",   /* the name of the type */
        sizeof(my_module_MyNewClassObject),
        0,                         /* tp_itemsize */
        0,                         /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        0,                         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        0,                         /* tp_str */
        0,                         /* tp_getattro */
        0,                         /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,        /* tp_flags */
        "MyNewClass bla bla",      /* tp_doc (the docstring) */
};

// module info
static PyModuleDef my_module = {
        PyModuleDef_HEAD_INIT,
        "my_module",
        "this is the docstring of the module...",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

/*
 * initialize the module
 */
PyMODINIT_FUNC
PyInit_my_module(void)
{
    PyObject *m;

    // initialize MyNewClass type
    my_module_MyNewClassType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&my_module_MyNewClassType) < 0)
        return NULL;

    m = PyModule_Create(&my_module);
    if (m == NULL)
        return NULL;

    // add the type to the module dictionary
    Py_INCREF(&my_module_MyNewClassType);
    PyModule_AddObject(m, "MyNewClass", (PyObject *)&my_module_MyNewClassType);
    return m;
}
