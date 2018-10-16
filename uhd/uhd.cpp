#include <Python.h>

/** Since import_array() IS called here, include like this. **/
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL UHD_ARRAY_API
#include <numpy/arrayobject.h>

#include "uhd.hpp"
#include "uhd_object.hpp"
#include "uhd_timespec.hpp"

namespace uhd {

PyObject *UhdError;

static PyMethodDef module_methods[] = {{NULL}};

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "uhd",                                  /* m_name */
    "USRP hardware driver Python module.",  /* m_doc */
    -1,                                     /* m_size */
    module_methods,                         /* m_methods */
    NULL,                                   /* m_reload */
    NULL,                                   /* m_traverse */
    NULL,                                   /* m_clear */
    NULL,                                   /* m_free */
};

#ifdef __cplusplus
extern "C"
#endif
PyMODINIT_FUNC PyInit_uhd(void) {

    import_array();

    PyObject *module = nullptr;
    if ((module = PyModule_Create(&moduledef)) == nullptr)
        return nullptr;

    /** Uhd object **/
    if (Uhd_register_type(module) < 0)
        return nullptr;

    /** TimeSpec object **/
    if (TimeSpec_register_type(module) < 0)
        return nullptr;

    /** UhdError exception **/
    UhdError = PyErr_NewExceptionWithDoc((char *)"uhd.UhdError", (char *)"UHD exception.", NULL, NULL);
    Py_INCREF(UhdError);
    PyModule_AddObject(module, "UhdError", UhdError);

    return module;
}

}
