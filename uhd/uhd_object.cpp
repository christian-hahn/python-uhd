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
#include "uhd_tx.hpp"

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
#include "uhd_3130199.hpp"
#include "uhd_3140099.hpp"
#include "uhd_3140199.hpp"
#include "uhd_3150099.hpp"
#include "uhd_4000099.hpp"

#ifndef __UHD_GEN_HPP__
  #error Unsupported UHD version
#endif

namespace uhd {

static void Usrp_dealloc(Usrp *self) {
    delete self->receiver;
    delete self->transmitter;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

static PyObject *Usrp_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Usrp *self = reinterpret_cast<Usrp *>(type->tp_alloc(type, 0));
    return reinterpret_cast<PyObject *>(self);
}

static int Usrp_init(Usrp *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);

    std::string dev_addr;
    if (nargs > 0) {
        Expect<std::string> _dev_addr;
        if (!(_dev_addr = to<std::string>(PyTuple_GetItem(args, 0)))) {
            PyErr_Format(PyExc_TypeError, "(0) dev_addr: %s", _dev_addr.what());
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

    self->transmitter = new TransmitWorker(self->dev, std::ref(self->dev_lock));
    self->transmitter->init();

    return 0;
}

static PyObject *_get_receive(Usrp *self, const bool fresh = false) {

    ReceiveResult *result = self->receiver->get_result(fresh);
    if (result->error) {
        std::string error(std::move(result->message));
        delete result;
        return PyErr_Format(UhdError, "Error on receive: %s", error.c_str());
    }

    const size_t num_channels = result->bufs.size();

    PyObject *ret, *ele;
    if (!(ret = PyList_New(num_channels))) {
        for (auto &ptr : result->bufs)
            free(ptr);
        delete result;
        return PyErr_Format(PyExc_ValueError, "Failed to create list.");
    }
    npy_intp dims = result->num_samps;
    for (size_t it = 0; it < num_channels; it++) {
        ele = PyArray_SimpleNewFromData(1, &dims, NPY_COMPLEX64, result->bufs[it]);
        PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject *>(ele), NPY_ARRAY_OWNDATA);
        PyList_SET_ITEM(ret, it, ele);
    }

    delete result;

    return ret;
}

#define DOC_RECEIVE \
"(1) Receive samples or start streaming.\n" \
"\n" \
"Args:\n" \
"    num_samps (int): number of samples\n" \
"    channels (sequence): sequence of channels to receive\n" \
"    streaming (bool, optional): is streaming receive, default is False\n" \
"    recycle (bool, optional): recycled un-claimed results, default is False\n" \
"    seconds_in_future (float, optional): seconds in the future to receive,\n" \
"                                         default is 1.0\n" \
"    timeout (float, optional): timeout in seconds, default is 0.5\n" \
"    otw_format (str, optional): over-the-wire format, default is 'sc16'\n" \
"\n" \
"Returns:\n" \
"    list: None if streaming else list of ndarrays\n" \
"\n" \
"(2) Receive streaming samples. Must follow a previous call to (1) receive() for\n" \
"    which streaming was True.\n" \
"\n" \
"Args:\n" \
"    fresh (bool, optional): block until fresh samples are available. Only valid\n" \
"                            if streaming and recycle are True. Default is False.\n" \
"\n" \
"Returns:\n" \
"    list: list of ndarrays\n"
static PyObject *Usrp_receive(Usrp *self, PyObject *args, PyObject *kwargs) {

    if (PyTuple_Size(args)) {

        /** Required **/
        PyObject *p_num_samps = nullptr;
        PyObject *p_channels = nullptr;
        /** Optional **/
        PyObject *p_streaming = nullptr;
        PyObject *p_recycle = nullptr;
        PyObject *p_seconds_in_future = nullptr;
        PyObject *p_timeout = nullptr;
        PyObject *p_otw_format = nullptr;
        static const char *keywords[] = {"num_samps", "channels", "streaming", "recycle",
                                         "seconds_in_future", "timeout", "otw_format", nullptr};
        if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OOOOO", const_cast<char **>(keywords), &p_num_samps,
                                        &p_channels, &p_streaming, &p_recycle, &p_seconds_in_future, &p_timeout, &p_otw_format)) {
            return nullptr;
        }

        /** num_samps **/
        Expect<size_t> num_samps;
        if (!(num_samps = to<size_t>(p_num_samps)))
            return PyErr_Format(PyExc_TypeError, "(0) num_samps: %s", num_samps.what());

        /** channels **/
        if ((p_channels = PySequence_Fast(p_channels, "(1) channels: Expected sequence.")) == nullptr)
            return nullptr;
        if (PySequence_Fast_GET_SIZE(p_channels) <= 0) {
            Py_DECREF(p_channels);
            return PyErr_Format(PyExc_TypeError, "(1) channels: Expected sequence of length > 0.");
        }
        std::vector<long unsigned int> channels(PySequence_Fast_GET_SIZE(p_channels));
        for (size_t it = 0; it < channels.size(); it++) {
            PyObject *elem = PySequence_Fast_GET_ITEM(p_channels, it);
            if (PyLong_CheckExact(elem)) {
                channels[it] = static_cast<long unsigned int>(PyLong_AsUnsignedLongMask(elem));
            } else {
                Py_DECREF(p_channels);
                return PyErr_Format(PyExc_TypeError, "(1) channels: Expected sequence of integers.");
            }
        }
        Py_DECREF(p_channels);

        /** streaming (optional) **/
        bool streaming = false;
        if (p_streaming) {
            Expect<bool> _streaming;
            if (!(_streaming = to<bool>(p_streaming)))
                return PyErr_Format(PyExc_TypeError, "streaming: %s", _streaming.what());
            streaming = _streaming.get();
        }

        /** recycle (optional) **/
        bool recycle = false;
        if (p_recycle) {
            Expect<bool> _recycle;
            if (!(_recycle = to<bool>(p_recycle)))
                return PyErr_Format(PyExc_TypeError, "recycle: %s", _recycle.what());
            recycle = _recycle.get();
        }

        /** seconds_in_future (optional) **/
        double seconds_in_future = 1.0;
        if (p_seconds_in_future) {
            Expect<double> _seconds_in_future;
            if (!(_seconds_in_future = to<double>(p_seconds_in_future)))
                return PyErr_Format(PyExc_TypeError, "seconds_in_future: %s", _seconds_in_future.what());
            seconds_in_future = _seconds_in_future.get();
        }

        /** timeout (optional) **/
        double timeout = 0.5;
        if (p_timeout) {
            Expect<double> _timeout;
            if (!(_timeout = to<double>(p_timeout)))
                return PyErr_Format(PyExc_TypeError, "timeout: %s", _timeout.what());
            timeout = _timeout.get();
        }

        /** otw_format (optional) **/
        std::string otw_format("sc16");
        if (p_otw_format) {
            Expect<std::string> _otw_format;
            if (!(_otw_format = to<std::string>(p_otw_format)))
                return PyErr_Format(PyExc_TypeError, "otw_format: %s", _otw_format.what());
            otw_format = _otw_format.get();
        }

        /** Classify request type. **/
        ReceiveRequestType req_type;
        if (streaming && recycle)
            req_type = ReceiveRequestType::Recycle;
        else if (streaming)
            req_type = ReceiveRequestType::Continuous;
        else
            req_type = ReceiveRequestType::Single;

        std::future<void> accepted = self->receiver->make_request(
            req_type,
            num_samps.get(),
            std::move(channels),
            seconds_in_future,
            timeout,
            otw_format
        );
        accepted.wait();

        if (streaming) {
            Py_INCREF(Py_None);
            return Py_None;
        } else {
            return _get_receive(self);
        }
    } else {
        /** Optional **/
        PyObject *p_fresh = nullptr;
        static const char *keywords[] = {"fresh", nullptr};
        if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", const_cast<char **>(keywords), &p_fresh)) {
            return nullptr;
        }

        /** fresh (optional) **/
        bool fresh = false;
        if (p_fresh) {
            Expect<bool> _fresh;
            if (!(_fresh = to<bool>(p_fresh)))
                return PyErr_Format(PyExc_TypeError, "fresh: %s", _fresh.what());
            fresh = _fresh.get();
        }

        return _get_receive(self, fresh);
    }
}

#define DOC_NUM_RECEIVED \
"Number of sample-blocks received.\n" \
"\n" \
"Returns:\n" \
"    int: number of sample-blocks\n"
static PyObject *Usrp_num_received(Usrp *self, void *closure) {
    return from(self->receiver->num_received());
}

#define DOC_STOP_RECEIVE \
"Stop receiving."
static PyObject *Usrp_stop_receive(Usrp *self, PyObject *args) {
    std::future<void> accepted = self->receiver->make_request(ReceiveRequestType::Stop);
    accepted.wait();
    Py_INCREF(Py_None);
    return Py_None;
}

#define DOC_TRANSMIT \
"Transmit samples.\n" \
"\n" \
"Args:\n" \
"    samples (sequence): sequence of ndarrays of samples of type complex64\n" \
"    channels (sequence): sequence of channels to receive of type int\n" \
"    continuous (bool, optional): is continuous transmit, default is False\n" \
"    seconds_in_future (float, optional): seconds in the future to transmit,\n" \
"                                         default is 1.0\n" \
"    timeout (float, optional): timeout in seconds, default is 0.5\n" \
"    otw_format (str, optional): over-the-wire format, default is 'sc16'\n"
static PyObject *Usrp_transmit(Usrp *self, PyObject *args, PyObject *kwargs) {

    /** Required **/
    PyObject *p_samples = nullptr;
    PyObject *p_channels = nullptr;
    /** Optional **/
    PyObject *p_continuous = nullptr;
    PyObject *p_seconds_in_future = nullptr;
    PyObject *p_timeout = nullptr;
    PyObject *p_otw_format = nullptr;
    static const char *keywords[] = {"samples", "channels", "continuous",
                                     "seconds_in_future", "timeout", "otw_format", nullptr};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OOOO", const_cast<char **>(keywords), &p_samples,
                                    &p_channels, &p_continuous, &p_seconds_in_future, &p_timeout, &p_otw_format)) {
        return nullptr;
    }

    /** samples **/
    if ((p_samples = PySequence_Fast(p_samples, "(0) samples: Expected sequence.")) == nullptr)
        return nullptr;
    if (PySequence_Fast_GET_SIZE(p_samples) <= 0) {
        Py_DECREF(p_samples);
        return PyErr_Format(PyExc_TypeError, "(0) samples: Expected sequence of length > 0.");
    }
    std::vector<std::complex<float> *> samples(PySequence_Fast_GET_SIZE(p_samples));
    size_t num_samps = 0;
    for (size_t it = 0; it < samples.size(); it++) {
        PyObject * const elem = PySequence_Fast_GET_ITEM(p_samples, it);
        if (!PyArray_CheckExact(elem)) {
            Py_DECREF(p_samples);
            return PyErr_Format(PyExc_TypeError, "(0) samples[i]: Expected ndarray.");
        }
        PyArrayObject * const array = reinterpret_cast<PyArrayObject *>(elem);
        const npy_intp size = PyArray_SIZE(array);
        if (!size) {
            Py_DECREF(p_samples);
            return PyErr_Format(PyExc_ValueError, "(0) samples[i]: Expected ndarray of size > 0.");
        }
        if (it && size != static_cast<npy_intp>(num_samps)) {
            Py_DECREF(p_samples);
            return PyErr_Format(PyExc_ValueError, "(0) samples[i]: Expected ndarrays of equal size.");
        }
        num_samps = static_cast<size_t>(size);
        if (PyArray_DESCR(array)->type_num != NPY_COMPLEX64) {
            Py_DECREF(p_samples);
            return PyErr_Format(PyExc_TypeError, "(0) samples[i]: Expected ndarrays of type complex64.");
        }
        if (!(PyArray_FLAGS(array) & NPY_ARRAY_OWNDATA)
            || PyArray_BASE(array) != nullptr
            || reinterpret_cast<PyArrayObject_fields *>(array)->weakreflist != nullptr
            || PyArray_REFCOUNT(array) > 2) {
            Py_DECREF(p_samples);
            return PyErr_Format(PyExc_ValueError, "(0) samples[i]: Bad ndarray: must own its data and "
                                                  "cannot be referenced-by or reference another array.");
        }
        samples[it] = reinterpret_cast<std::complex<float> *>(array);
    }
    Py_DECREF(p_samples);
    for (size_t it = 0; it < samples.size(); it++) {
        PyArrayObject * const array = reinterpret_cast<PyArrayObject *>(samples[it]);
        PyArrayObject_fields * const fields = reinterpret_cast<PyArrayObject_fields *>(array);
        samples[it] = reinterpret_cast<std::complex<float> *>(fields->data);
        void * const new_data = PyDataMem_NEW(PyArray_DESCR(array)->elsize);
        if (new_data == nullptr)
            return PyErr_Format(PyExc_MemoryError, "Failed to allocate memory for array.");
        fields->data = reinterpret_cast<char *>(new_data);
        PyDimMem_FREE(fields->dimensions);
        fields->nd = 0;
        fields->dimensions = nullptr;
        fields->strides = nullptr;
    }

    /** channels **/
    if ((p_channels = PySequence_Fast(p_channels, "(1) channels: Expected sequence.")) == nullptr)
        return nullptr;
    if (PySequence_Fast_GET_SIZE(p_channels) != static_cast<Py_ssize_t>(samples.size())) {
        Py_DECREF(p_channels);
        return PyErr_Format(PyExc_TypeError, "(1) channels: Expected sequence of length %d.", samples.size());
    }
    std::vector<long unsigned int> channels(samples.size());
    for (size_t it = 0; it < channels.size(); it++) {
        PyObject *elem = PySequence_Fast_GET_ITEM(p_channels, it);
        if (PyLong_CheckExact(elem)) {
            channels[it] = static_cast<long unsigned int>(PyLong_AsUnsignedLongMask(elem));
        } else {
            Py_DECREF(p_channels);
            return PyErr_Format(PyExc_TypeError, "(1) channels: Expected sequence of integers.");
        }
    }
    Py_DECREF(p_channels);

    /** continuous (optional) **/
    bool continuous = false;
    if (p_continuous) {
        Expect<bool> _continuous;
        if (!(_continuous = to<bool>(p_continuous)))
            return PyErr_Format(PyExc_TypeError, "continuous: %s", _continuous.what());
        continuous = _continuous.get();
    }

    /** seconds_in_future (optional) **/
    double seconds_in_future = 1.0;
    if (p_seconds_in_future) {
        Expect<double> _seconds_in_future;
        if (!(_seconds_in_future = to<double>(p_seconds_in_future)))
            return PyErr_Format(PyExc_TypeError, "seconds_in_future: %s", _seconds_in_future.what());
        seconds_in_future = _seconds_in_future.get();
    }

    /** timeout (optional) **/
    double timeout = 0.5;
    if (p_timeout) {
        Expect<double> _timeout;
        if (!(_timeout = to<double>(p_timeout)))
            return PyErr_Format(PyExc_TypeError, "timeout: %s", _timeout.what());
        timeout = _timeout.get();
    }

    /** otw_format (optional) **/
    std::string otw_format("sc16");
    if (p_otw_format) {
        Expect<std::string> _otw_format;
        if (!(_otw_format = to<std::string>(p_otw_format)))
            return PyErr_Format(PyExc_TypeError, "otw_format: %s", _otw_format.what());
        otw_format = _otw_format.get();
    }

    /** Classify request type. **/
    TransmitRequestType req_type;
    if (continuous)
        req_type = TransmitRequestType::Continuous;
    else
        req_type = TransmitRequestType::Single;

    std::future<std::string> accepted = self->transmitter->make_request(
        req_type,
        num_samps,
        std::move(samples),
        std::move(channels),
        seconds_in_future,
        timeout,
        otw_format
    );
    accepted.wait();

    const std::string &error = accepted.get();
    if (!error.empty())
        return PyErr_Format(UhdError, "Error on transmit(): %s", error.c_str());

    Py_INCREF(Py_None);
    return Py_None;
}

#define DOC_STOP_TRANSMIT \
"Stop transmit."
static PyObject *Usrp_stop_transmit(Usrp *self, PyObject *args) {
    std::future<std::string> accepted = self->transmitter->make_request(TransmitRequestType::Stop);
    accepted.wait();
    const std::string &error = accepted.get();
    if (!error.empty())
        return PyErr_Format(UhdError, "Error on stop_transmit(): %s", error.c_str());
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMemberDef Usrp_members[] = {{NULL}};

static std::vector<PyMethodDef> Usrp_methods;

const static std::vector<PyMethodDef> Usrp_user_methods {
    {"receive", (PyCFunction)Usrp_receive, METH_VARARGS | METH_KEYWORDS, DOC_RECEIVE},
    {"stop_receive", (PyCFunction)Usrp_stop_receive, METH_NOARGS, DOC_STOP_RECEIVE},
    {"transmit", (PyCFunction)Usrp_transmit, METH_VARARGS | METH_KEYWORDS, DOC_TRANSMIT},
    {"stop_transmit", (PyCFunction)Usrp_stop_transmit, METH_NOARGS, DOC_STOP_TRANSMIT},
};

static PyGetSetDef Usrp_getset[] = {
    {(char *)"num_received", (getter)Usrp_num_received, NULL, (char *)DOC_NUM_RECEIVED, NULL},
    {NULL},
};

static PyTypeObject UsrpType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "pyuhd.Usrp",                              /* tp_name */
    sizeof(Usrp),                              /* tp_basicsize */
    0,                                         /* tp_itemsize */
    (destructor)Usrp_dealloc,                  /* tp_dealloc */
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
    "Usrp object",                             /* tp_doc */
    0,                                         /* tp_traverse */
    0,                                         /* tp_clear */
    0,                                         /* tp_richcompare */
    0,                                         /* tp_weaklistoffset */
    0,                                         /* tp_iter */
    0,                                         /* tp_iternext */
    0,                                         /* tp_methods (DEFERRED) */
    Usrp_members,                              /* tp_members */
    Usrp_getset,                               /* tp_getset */
    0,                                         /* tp_base */
    0,                                         /* tp_dict */
    0,                                         /* tp_descr_get */
    0,                                         /* tp_descr_set */
    0,                                         /* tp_dictoffset */
    (initproc)Usrp_init,                       /* tp_init */
    0,                                         /* tp_alloc */
    Usrp_new,                                  /* tp_new */
};

int Usrp_register_type(PyObject *module) {
    /* Append generated & user methods */
    Usrp_methods.clear();
    for (const auto &method: Usrp_gen_methods)
        Usrp_methods.push_back(method);
    for (const auto &method: Usrp_user_methods)
        Usrp_methods.push_back(method);
    Usrp_methods.push_back({NULL});
    UsrpType.tp_methods = Usrp_methods.data();
    if (PyType_Ready(&UsrpType) < 0)
        return -1;
    Py_INCREF(&UsrpType);
    PyModule_AddObject(module, "Usrp", reinterpret_cast<PyObject *>(&UsrpType));
    return 0;
}

}
