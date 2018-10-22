#include <vector>

#include <Python.h>
#include "structmember.h"

/** Since import_array() is NOT called here, include like this. **/
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL UHD_ARRAY_API
#include <numpy/arrayobject.h>

#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
#include <uhd/version.hpp>

#include "uhd.hpp"
#include "uhd_object.hpp"
#include "uhd_types.hpp"
#include "uhd_expect.hpp"
#include "uhd_rx.hpp"

#include "uhd_30900.hpp"
#include "uhd_30901.hpp"
#include "uhd_30902.hpp"
#include "uhd_30903.hpp"
#include "uhd_30904.hpp"
#include "uhd_30905.hpp"
#include "uhd_30906.hpp"
#include "uhd_30907.hpp"
#include "uhd_3100099.hpp"
#include "uhd_3100199.hpp"
#include "uhd_3100299.hpp"
#include "uhd_3100399.hpp"
#include "uhd_3110099.hpp"
#include "uhd_3110199.hpp"
#include "uhd_3120099.hpp"
#include "uhd_3130099.hpp"

#ifndef __UHD_GEN_HPP__
  #error Unsupported UHD version
#endif

namespace uhd {

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
            PyErr_Format(PyExc_TypeError, "[0] dev_addr: %s", _dev_addr.what());
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
        return PyErr_Format(UhdError, "Error on receive: %s", _result.what());
    ReceiveResult &result = _result.get();

    const size_t num_channels = result.bufs.size();

    PyObject *ret, *ele;
    if (!(ret = PyList_New(num_channels)))
        return PyErr_Format(PyExc_ValueError, "Failed to create list.");
    npy_intp dims = result.num_samps;
    for (size_t it = 0; it < num_channels; it++) {
        ele = PyArray_SimpleNewFromData(1, &dims, NPY_COMPLEX64, result.bufs[it]);
        PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject *>(ele), NPY_ARRAY_OWNDATA);
        PyList_SET_ITEM(ret, it, ele);
    }

    return ret;
}

#define DOC_RECEIVE \
"Receive samples. Given at least, arguments 'num_samps' and 'channels', a new \n" \
"receive is started. If no arguments are provided, this call must follow a \n" \
"previous call to receive for which streaming = True." \
"\n" \
"Args:\n" \
"    num_samps (int): number of samples\n" \
"    channels (sequence): list of channels to receive\n" \
"    streaming (bool, optional): is receive streaming\n" \
"    seconds_in_future (float, optional): seconds in the future to receive\n" \
"    timeout (float, optional): timeout in seconds\n" \
"\n" \
"Returns:\n" \
"    list: list of ndarrays if not streaming else None\n"
static PyObject *Uhd_receive(Uhd *self, PyObject *args, PyObject *kwargs) {

    if (PyTuple_Size(args)) {

        /** Required **/
        PyObject *p_num_samps = nullptr;
        PyObject *p_channels = nullptr;
        /** Optional **/
        PyObject *p_streaming = nullptr;
        PyObject *p_seconds_in_future = nullptr;
        PyObject *p_timeout = nullptr;
        static const char *keywords[] = {"num_samps", "channels", "streaming",
                                         "seconds_in_future", "timeout", nullptr};
        if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OOO", const_cast<char **>(keywords), &p_num_samps,
                                        &p_channels, &p_streaming, &p_seconds_in_future, &p_timeout)) {
            return nullptr;
        }

        /** num_samps **/
        Expect<size_t> num_samps;
        if (!(num_samps = to<size_t>(p_num_samps)))
            return PyErr_Format(PyExc_TypeError, "[0] num_samps: %s", num_samps.what());

        /** channels **/
        std::vector<long unsigned int> channels;
        if (PySequence_Check(p_channels) && (p_channels = PySequence_Fast(p_channels, "[1] channels: Expected sequence.")) != nullptr
            && PySequence_Fast_GET_SIZE(p_channels)) {
            channels.resize(PySequence_Fast_GET_SIZE(p_channels));
            for (size_t it = 0; it < channels.size(); it++) {
                PyObject *elem = PySequence_Fast_GET_ITEM(p_channels, it);
                if (PyLong_CheckExact(elem)) {
                    channels[it] = static_cast<long unsigned int>(PyLong_AsUnsignedLongMask(elem));
                } else {
                    PyErr_SetString(PyExc_TypeError, "[1] channels: Expected sequence of integers.");
                    return nullptr;
                }
            }
        } else {
            return PyErr_Format(PyExc_TypeError, "[1] channels: Expected sequence of non-zero length.");
        }

        /** streaming (optional) **/
        bool streaming = false;
        if (p_streaming) {
            Expect<bool> _streaming;
            if (!(_streaming = to<bool>(p_streaming)))
                return PyErr_Format(PyExc_TypeError, "[2] streaming: %s", _streaming.what());
            streaming = _streaming.get();
        }

        /** seconds_in_future (optional) **/
        double seconds_in_future = 1.0;
        if (p_seconds_in_future) {
            Expect<double> _seconds_in_future;
            if (!(_seconds_in_future = to<double>(p_seconds_in_future)))
                return PyErr_Format(PyExc_TypeError, "[3] seconds_in_future: %s", _seconds_in_future.what());
            seconds_in_future = _seconds_in_future.get();
        }

        /** timeout (optional) **/
        double timeout = seconds_in_future + 0.1;
        if (p_timeout) {
            Expect<double> _timeout;
            if (!(_timeout = to<double>(p_timeout)))
                return PyErr_Format(PyExc_TypeError, "[4] timeout: %s", _timeout.what());
            timeout = _timeout.get();
        }

        std::future<void> accepted = self->receiver->request(
            streaming ? ReceiveRequestType::Continuous : ReceiveRequestType::Single,
            num_samps.get(),
            std::move(channels),
            seconds_in_future,
            timeout
        );
        accepted.wait();

        if (streaming) {
            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    return _get_receive(self);
}

#define DOC_NUM_RECEIVED \
"Number of sample-blocks received.\n" \
"\n" \
"Returns:\n" \
"    int: number of sample-blocks\n"
static PyObject *Uhd_num_received(Uhd *self, void *closure) {
    return from(self->receiver->num_received());
}

#define DOC_STOP_RECEIVE \
"Stop receiving."
static PyObject *Uhd_stop_receive(Uhd *self, PyObject *args) {
    std::future<void> accepted = self->receiver->request(ReceiveRequestType::Stop);
    accepted.wait();
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMemberDef Uhd_members[] = {{NULL}};

static std::vector<PyMethodDef> Uhd_methods;

const static std::vector<PyMethodDef> Uhd_user_methods {
    {"receive", (PyCFunction)Uhd_receive, METH_VARARGS | METH_KEYWORDS, DOC_RECEIVE},
    {"stop_receive", (PyCFunction)Uhd_stop_receive, METH_NOARGS, DOC_STOP_RECEIVE},
};

static PyGetSetDef Uhd_getset[] = {
    {(char *)"num_received", (getter)Uhd_num_received, NULL, (char *)DOC_NUM_RECEIVED, NULL},
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

int Uhd_register_type(PyObject *module) {
    /* Append generated & user methods */
    Uhd_methods.clear();
    for (const auto &method: Uhd_gen_methods)
        Uhd_methods.push_back(method);
    for (const auto &method: Uhd_user_methods)
        Uhd_methods.push_back(method);
    Uhd_methods.push_back({NULL});
    UhdType.tp_methods = Uhd_methods.data();
    if (PyType_Ready(&UhdType) < 0)
        return -1;
    Py_INCREF(&UhdType);
    PyModule_AddObject(module, "Uhd", reinterpret_cast<PyObject *>(&UhdType));
    return 0;
}

}
