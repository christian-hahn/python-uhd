#include <Python.h>
#include "structmember.h"

#include "uhd_timespec.hpp"
#include "uhd_types.hpp"
#include "uhd_expect.hpp"

namespace uhd {

#define DOC_TIMESPEC \
"A TimeSpec holds a seconds and a fractional seconds time value.\n" \
"Depending upon usage, the TimeSpec can represent absolute times,\n" \
"relative times, or time differences (between absolute times).\n" \
"\n" \
"The TimeSpec provides clock-domain independent time storage,\n" \
"but can convert fractional seconds to/from clock-domain specific units.\n" \
"\n" \
"The fractional seconds are stored as double precision floating point.\n" \
"This gives the fractional seconds enough precision to unambiguously\n" \
"specify a clock-tick/sample-count up to rates of several petahertz.\n"

static void TimeSpec_dealloc(TimeSpec *self) {
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

static PyObject *TimeSpec_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    TimeSpec *self = reinterpret_cast<TimeSpec *>(type->tp_alloc(type, 0));
    return reinterpret_cast<PyObject *>(self);
}

static int TimeSpec_init(TimeSpec *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);

    if (nargs == 1) {
        /** Create a time_spec_t from a real-valued seconds count.
            secs: the real-valued seconds count (default = 0) **/
        Expect<double> secs;
        if (!(secs = to<double>(PyTuple_GetItem(args, 0)))) {
            PyErr_Format(PyExc_TypeError, "[0] secs: %s", secs.what());
            return -1;
        }
        self->_time_spec = time_spec_t(secs.get());
    } else if (nargs == 2) {
        /** Create a time_spec_t from whole and fractional seconds.
            full_secs: the whole/integer seconds count
            frac_secs: the fractional seconds count (default = 0) **/
        Expect<long> full_secs;
        if (!(full_secs = to<long>(PyTuple_GetItem(args, 0)))) {
            PyErr_Format(PyExc_TypeError, "[0] full_secs: %s", full_secs.what());
            return -1;
        }
        Expect<double> frac_secs;
        if (!(frac_secs = to<double>(PyTuple_GetItem(args, 1)))) {
            PyErr_Format(PyExc_TypeError, "[1] frac_secs: %s", frac_secs.what());
            return -1;
        }
        self->_time_spec = time_spec_t(full_secs.get(), frac_secs.get());
    } else if (nargs == 3) {
        /** Create a time_spec_t from whole seconds and fractional ticks.
            Translation from clock-domain specific units.
            full_secs: the whole/integer seconds count
            tick_count: the fractional seconds tick count
            tick_rate: the number of ticks per second **/
        Expect<long> full_secs;
        if (!(full_secs = to<long>(PyTuple_GetItem(args, 0)))) {
            PyErr_Format(PyExc_TypeError, "[0] full_secs: %s", full_secs.what());
            return -1;
        }
        Expect<long> tick_count;
        if (!(tick_count = to<long>(PyTuple_GetItem(args, 1)))) {
            PyErr_Format(PyExc_TypeError, "[1] tick_count: %s", tick_count.what());
            return -1;
        }
        Expect<double> tick_rate;
        if (!(tick_rate = to<double>(PyTuple_GetItem(args, 2)))) {
            PyErr_Format(PyExc_TypeError, "[2] tick_rate: %s", tick_rate.what());
            return -1;
        }
        self->_time_spec = time_spec_t(full_secs.get(), tick_count.get(), tick_rate.get());
    } else {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected 1, 2 or 3.", nargs);
        return -1;
    }

    return 0;
}

static PyObject *TimeSpec_repr(TimeSpec *self)
{
    std::string str = "TimeSpec(full_secs: " + std::to_string(self->_time_spec.get_full_secs()) +
                      ", frac_secs: " + std::to_string(self->_time_spec.get_frac_secs()) + ")";
    return PyUnicode_FromString(str.c_str());
}

#define DOC_GET_TICK_COUNT \
"Convert the fractional seconds to clock ticks.\n" \
"Translation into clock-domain specific units.\n" \
"\n" \
"Args:\n" \
"    tick_rate (float): the number of ticks per second\n" \
"\n" \
"Returns:\n" \
"    int: the fractional seconds tick count\n"
PyObject *TimeSpec_get_tick_count(TimeSpec *self, PyObject *args) {
    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs != 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected 1.", nargs);
    Expect<double> tick_rate;
    if (!(tick_rate = to<double>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "[0] tick_rate: %s", tick_rate.what());
    return from(self->_time_spec.get_tick_count(tick_rate.get()));
}

#define DOC_TO_TICKS \
"Convert the time spec into a 64-bit tick count.\n" \
"Translation into clock-domain specific units.\n" \
"\n" \
"Args:\n" \
"    tick_rate (float): the number of ticks per second\n" \
"\n" \
"Returns:\n" \
"    int: an integer number of ticks\n"
PyObject *TimeSpec_to_ticks(TimeSpec *self, PyObject *args) {
    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs != 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected 1.", nargs);
    Expect<double> tick_rate;
    if (!(tick_rate = to<double>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "[0] tick_rate: %s", tick_rate.what());
    return from(static_cast<int64_t>(self->_time_spec.to_ticks(tick_rate.get())));
}

#define DOC_REAL_SECONDS \
"Get the time as a real-valued seconds count.\n" \
"Note: If this time_spec_t represents an absolute time,\n" \
"the precision of the fractional seconds may be lost.\n" \
"\n" \
"Returns:\n" \
"    float: the real-valued seconds\n"
static PyObject *TimeSpec_real_seconds(TimeSpec *self, void *closure) {
    return from(self->_time_spec.get_real_secs());
}

#define DOC_FULL_SECONDS \
"Get the whole/integer part of the time in seconds.\n" \
"\n" \
"Returns:\n" \
"    int: the whole/integer seconds\n"
static PyObject *TimeSpec_full_seconds(TimeSpec *self, void *closure) {
    return from(static_cast<int64_t>(self->_time_spec.get_full_secs()));
}

#define DOC_FRAC_SECONDS \
"Get the fractional part of the time in seconds..\n" \
"\n" \
"Returns:\n" \
"    float: the fractional seconds\n"
static PyObject *TimeSpec_frac_seconds(TimeSpec *self, void *closure) {
    return from(self->_time_spec.get_frac_secs());
}

static PyObject *TimeSpec_add(PyObject *left, PyObject *right) {
    if (PyObject_TypeCheck(left, &TimeSpecType)) {
        if (PyObject_TypeCheck(right, &TimeSpecType))
            return from(reinterpret_cast<TimeSpec *>(left)->_time_spec + reinterpret_cast<TimeSpec *>(right)->_time_spec);
        else if (PyFloat_CheckExact(right))
            return from(reinterpret_cast<TimeSpec *>(left)->_time_spec + PyFloat_AsDouble(right));
    } else if (PyFloat_CheckExact(left)) {
        if (PyObject_TypeCheck(right, &TimeSpecType))
            return from(PyFloat_AsDouble(left) + reinterpret_cast<TimeSpec *>(right)->_time_spec);
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyMemberDef TimeSpec_members[] = {{NULL}};
static PyMethodDef TimeSpec_methods[] = {
    {"get_tick_count", (PyCFunction)TimeSpec_get_tick_count, METH_VARARGS, DOC_GET_TICK_COUNT},
    {"to_ticks", (PyCFunction)TimeSpec_to_ticks, METH_VARARGS, DOC_TO_TICKS},
    {NULL},
};
static PyGetSetDef TimeSpec_getset[] = {
    {(char *)"real_seconds", (getter)TimeSpec_real_seconds, NULL, (char *)DOC_REAL_SECONDS, NULL},
    {(char *)"full_seconds", (getter)TimeSpec_full_seconds, NULL, (char *)DOC_FULL_SECONDS, NULL},
    {(char *)"frac_seconds", (getter)TimeSpec_frac_seconds, NULL, (char *)DOC_FRAC_SECONDS, NULL},
    {NULL},
};

static PyNumberMethods TimeSpec_as_number = {
    TimeSpec_add,    /* nb_add */
    0,               /* nb_subtract */
    0,               /* nb_multiply */
    0,               /* nb_remainder */
    0,               /* nb_divmod */
    0,               /* nb_power */
    0,               /* nb_negative */
    0,               /* nb_positive */
    0,               /* nb_absolute */
    0,               /* nb_bool */
    0,               /* nb_invert */
    0,               /* nb_lshift */
    0,               /* nb_rshift */
    0,               /* nb_and */
    0,               /* nb_xor */
    0,               /* nb_or */
    0,               /* nb_int */
    0,               /* nb_reserved */
    0,               /* nb_float */
};

PyTypeObject TimeSpecType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "uhd.TimeSpec",                            /* tp_name */
    sizeof(TimeSpec),                          /* tp_basicsize */
    0,                                         /* tp_itemsize */
    (destructor)TimeSpec_dealloc,              /* tp_dealloc */
    0,                                         /* tp_print */
    0,                                         /* tp_getattr */
    0,                                         /* tp_setattr */
    0,                                         /* tp_as_async */
    (reprfunc)TimeSpec_repr,                   /* tp_repr */
    &TimeSpec_as_number,                       /* tp_as_number */
    0,                                         /* tp_as_sequence */
    0,                                         /* tp_as_mapping */
    0,                                         /* tp_hash  */
    0,                                         /* tp_call */
    (reprfunc)TimeSpec_repr,                   /* tp_str */
    0,                                         /* tp_getattro */
    0,                                         /* tp_setattro */
    0,                                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* tp_flags */
    DOC_TIMESPEC,                              /* tp_doc */
    0,                                         /* tp_traverse */
    0,                                         /* tp_clear */
    0,                                         /* tp_richcompare */
    0,                                         /* tp_weaklistoffset */
    0,                                         /* tp_iter */
    0,                                         /* tp_iternext */
    TimeSpec_methods,                          /* tp_methods */
    TimeSpec_members,                          /* tp_members */
    TimeSpec_getset,                           /* tp_getset */
    0,                                         /* tp_base */
    0,                                         /* tp_dict */
    0,                                         /* tp_descr_get */
    0,                                         /* tp_descr_set */
    0,                                         /* tp_dictoffset */
    (initproc)TimeSpec_init,                   /* tp_init */
    0,                                         /* tp_alloc */
    TimeSpec_new,                              /* tp_new */
};

int TimeSpec_register_type(PyObject *module) {
    if (PyType_Ready(&TimeSpecType) < 0)
        return -1;
    Py_INCREF(&TimeSpecType);
    PyModule_AddObject(module, "TimeSpec", reinterpret_cast<PyObject *>(&TimeSpecType));
    return 0;
}

TimeSpec *TimeSpec_from_time_spec_t(const time_spec_t &value) {
    TimeSpec *ret = reinterpret_cast<TimeSpec *>(TimeSpecType.tp_new(&TimeSpecType, nullptr, nullptr));
    if (!ret)
        return nullptr;
    ret->_time_spec = value;
    return ret;
}

}
