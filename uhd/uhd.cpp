#include <Python.h>

/** Since import_array() IS called here, include like this. **/
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL UHD_ARRAY_API
#include <numpy/arrayobject.h>

#include <uhd/version.hpp>

#include "uhd.hpp"
#include "uhd_object.hpp"
#include "uhd_timespec.hpp"

namespace uhd {

PyObject *UhdError;

static PyMethodDef module_methods[] = {{NULL}};

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "pyuhd",                                /* m_name */
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
PyMODINIT_FUNC PyInit_pyuhd(void) {

    import_array();

    PyObject *module = nullptr;
    if ((module = PyModule_Create(&moduledef)) == nullptr)
        return nullptr;

    /** Add Uhd object **/
    if (Uhd_register_type(module) < 0)
        return nullptr;

    /** Add TimeSpec object **/
    if (TimeSpec_register_type(module) < 0)
        return nullptr;

    /** Add UhdError exception **/
    UhdError = PyErr_NewExceptionWithDoc((char *)"pyuhd.UhdError", (char *)"UHD exception.", NULL, NULL);
    Py_INCREF(UhdError);
    PyModule_AddObject(module, "UhdError", UhdError);

    /** Add UHD_{VERSION, VERSION_ABI, VERSION_LONG} **/
    if (PyModule_AddIntConstant(module, "UHD_VERSION", UHD_VERSION))
        return PyErr_Format(PyExc_ValueError, "Failed to add int UHD_VERSION.");
    if (PyModule_AddStringConstant(module, "UHD_VERSION_ABI", get_abi_string().c_str()))
        return PyErr_Format(PyExc_ValueError, "Failed to add string UHD_VERSION_ABI.");
    if (PyModule_AddStringConstant(module, "UHD_VERSION_LONG", get_version_string().c_str()))
        return PyErr_Format(PyExc_ValueError, "Failed to add string UHD_VERSION_LONG.");

    /** Add __version__ **/
    if (PyModule_AddStringConstant(module, "__version__", PYUHD_VERSION))
        return PyErr_Format(PyExc_ValueError, "Failed to add string __version__.");

    return module;
}

}
