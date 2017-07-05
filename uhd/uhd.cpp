#include <vector>

#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
#include <uhd/version.hpp>

#include "uhd.hpp"
#include "uhd_types.hpp"
#include "uhd_expect.hpp"
#include "uhd_rx.hpp"

#if UHD_VERSION == 30906
  /** UHD tag release_003_009_006 **/
  #include "uhd_30906.hpp"
#elif UHD_VERSION == 3100199
  /** UHD tag release_003_010_001_001 **/
  #include "uhd_3100199.hpp"
#else
  #error Unsupported UHD version
#endif

namespace uhd {

PyObject *UhdError;

static void Uhd_dealloc(Uhd *self) {
    delete self->receiver;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

static PyObject *Uhd_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Uhd *self = reinterpret_cast<Uhd *>(type->tp_alloc(type, 0));
    return reinterpret_cast<PyObject *>(self);
}

static int Uhd_init(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);

    std::string dev_addr;
    if (nargs > 0) {
        Expect<std::string> _dev_addr;
        if (!(_dev_addr = to<std::string>(PyTuple_GetItem(args, 0)))) {
            PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", _dev_addr.what());
            return -1;
        }
        dev_addr = _dev_addr.get();
    }

    try {
        self->dev = uhd::usrp::multi_usrp::make(dev_addr);
    } catch(const uhd::exception &e) {
        PyErr_SetString(UhdError, e.what());
        return -1;
    } catch(...) {
        PyErr_SetString(UhdError, "Error: unknown exception has occurred.");
        return -1;
    }

    self->receiver = new ReceiveWorker(self->dev, std::ref(self->dev_lock));
    self->receiver->init();

    return 0;
}

static PyObject *_get_receive(Uhd *self) {

    Expect<ReceiveResult> _result;
    if (!(_result = self->receiver->read()))
        return PyErr_Format(PyExc_TypeError, "Failed to receive: %s", _result.what());
    ReceiveResult &result = _result.get();

    const size_t num_channels = result.bufs.size();

    PyObject *ret, *ele;
    if (!(ret = PyList_New(num_channels)))
        return PyErr_Format(PyExc_ValueError, "Failed to create list.");
    npy_intp dims = result.num_samps;
    for (unsigned int i = 0; i < num_channels; i++) {
        ele = PyArray_SimpleNewFromData(1, &dims, NPY_COMPLEX64, result.bufs[i]);
        PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject *>(ele), NPY_ARRAY_OWNDATA);
        PyList_SET_ITEM(ret, i, ele);
    }

    return ret;
}

static PyObject *Uhd_receive(Uhd *self, PyObject *args, PyObject *kwds) {

    const Py_ssize_t nargs = PyTuple_Size(args);

    if (nargs) {

        /** Required **/
        size_t num_samps = 0;
        PyObject *_channels = nullptr;
        std::vector<size_t> channels;
        /** Optional **/
        PyObject *_streaming = nullptr;
        bool streaming = false;
        double seconds_in_future = 1.0;
        double timeout = 5.0;

        static const char *kwds_list[] = {"num_samps", "channels", "streaming",
                                          "seconds_in_future", "timeout", nullptr};
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "IO|Odd", const_cast<char **>(kwds_list), &num_samps,
                                        &_channels, &_streaming, &seconds_in_future, &timeout)) {
            return nullptr;
        }

        if (PySequence_Check(_channels) && (_channels = PySequence_Fast(_channels, "Expected sequence.")) != nullptr
            && PySequence_Fast_GET_SIZE(_channels)) {
            channels.resize(PySequence_Fast_GET_SIZE(_channels));
            for (size_t it = 0; it < channels.size(); it++) {
                PyObject *elem = PySequence_Fast_GET_ITEM(_channels, it);
                if (PyLong_CheckExact(elem)) {
                    channels[it] = static_cast<size_t>(PyLong_AsUnsignedLongMask(elem));
                } else {
                    PyErr_SetString(PyExc_TypeError, "Invalid argument for argument # 2: channels must be list of integers.");
                    return nullptr;
                }
            }
        } else {
            return PyErr_Format(PyExc_TypeError, "Invalid argument for argument # 2: expected sequence of non-zero length.");
        }

        streaming = (_streaming) ? PyObject_IsTrue(_streaming) : streaming;

        std::future<void> accepted = self->receiver->request(streaming ? ReceiveRequestType::Continuous : ReceiveRequestType::Single,
                                                             num_samps, std::move(channels), seconds_in_future, timeout);
        accepted.wait();

        if (streaming) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        return _get_receive(self);
    } else {
        return _get_receive(self);
    }
}

static PyObject *Uhd_num_received(Uhd *self, void *closure) {
    return from(self->receiver->num_received());
}

static PyObject *Uhd_stop_receive(Uhd *self, PyObject *args) {

    std::future<void> accepted = self->receiver->request(ReceiveRequestType::Stop);
    accepted.wait();

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef module_methods[] = {{NULL}};
static PyMemberDef Uhd_members[] = {{NULL}};

static std::vector<PyMethodDef> Uhd_methods;
const static std::vector<PyMethodDef> Uhd_user_methods {
    {"receive", (PyCFunction)Uhd_receive, METH_VARARGS | METH_KEYWORDS, ""},
    {"stop_receive", (PyCFunction)Uhd_stop_receive, METH_NOARGS, ""},
};

static PyGetSetDef Uhd_getset[] = {
    {(char *)"num_received", (getter)Uhd_num_received, NULL, (char *)"Number of sample-blocks received.", NULL},
    {NULL},
};

static PyTypeObject UhdType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "uhd.Uhd",                                 /* tp_name */
    sizeof(Uhd),                               /* tp_basicsize */
    0,                                         /* tp_itemsize */
    (destructor)Uhd_dealloc,                   /* tp_dealloc */
    0,                                         /* tp_print */
    0,                                         /* tp_getattr */
    0,                                         /* tp_setattr */
    0,                                         /* tp_as_async */
    0,                                         /* tp_repr */
    0,                                         /* tp_as_number */
    0,                                         /* tp_as_sequence */
    0,                                         /* tp_as_mapping */
    0,                                         /* tp_hash  */
    0,                                         /* tp_call */
    0,                                         /* tp_str */
    0,                                         /* tp_getattro */
    0,                                         /* tp_setattro */
    0,                                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* tp_flags */
    "Uhd object",                              /* tp_doc */
    0,                                         /* tp_traverse */
    0,                                         /* tp_clear */
    0,                                         /* tp_richcompare */
    0,                                         /* tp_weaklistoffset */
    0,                                         /* tp_iter */
    0,                                         /* tp_iternext */
    0,                                         /* tp_methods (DEFERRED) */
    Uhd_members,                               /* tp_members */
    Uhd_getset,                                /* tp_getset */
    0,                                         /* tp_base */
    0,                                         /* tp_dict */
    0,                                         /* tp_descr_get */
    0,                                         /* tp_descr_set */
    0,                                         /* tp_dictoffset */
    (initproc)Uhd_init,                        /* tp_init */
    0,                                         /* tp_alloc */
    Uhd_new,                                   /* tp_new */
};

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

    /** Register generated & user methods **/
    Uhd_methods.clear();
    for (const auto &method: Uhd_gen_methods)
        Uhd_methods.push_back(method);
    for (const auto &method: Uhd_user_methods)
        Uhd_methods.push_back(method);
    Uhd_methods.push_back({NULL});
    UhdType.tp_methods = Uhd_methods.data();

    if (PyType_Ready(&UhdType) < 0)
        return nullptr;

    PyObject *m = nullptr;
    if ((m = PyModule_Create(&moduledef)) == nullptr)
        return nullptr;

    Py_INCREF(&UhdType);
    PyModule_AddObject(m, "Uhd", reinterpret_cast<PyObject *>(&UhdType));

    UhdError = PyErr_NewExceptionWithDoc((char *)"uhd.error", (char *)"UHD exception.", NULL, NULL);
    Py_INCREF(UhdError);
    PyModule_AddObject(m, "UhdError", UhdError);

    return m;
}

}
