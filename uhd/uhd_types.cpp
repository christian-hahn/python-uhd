#include <limits>

#include <Python.h>

#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
#include <uhd/types/dict.hpp>

#include "uhd.hpp"
#include "uhd_types.hpp"
#include "uhd_expect.hpp"

namespace uhd {

/******************************************************************************/
/** Functions to determine if PyObject is of type <T>. **/

template<>
bool is<bool>(PyObject *obj) {
    return static_cast<bool>(PyBool_Check(obj));
}

template<>
bool is<uint8_t>(PyObject *obj) {
    return static_cast<bool>(PyLong_CheckExact(obj));
}

template<>
bool is<uint32_t>(PyObject *obj) {
    return static_cast<bool>(PyLong_CheckExact(obj));
}

template<>
bool is<uint64_t>(PyObject *obj) {
    return static_cast<bool>(PyLong_CheckExact(obj));
}

template<>
bool is<double>(PyObject *obj) {
    return static_cast<bool>(PyFloat_CheckExact(obj));
}

template<>
bool is<std::string>(PyObject *obj) {
    return static_cast<bool>(PyUnicode_CheckExact(obj));
}

template<>
bool is<std::complex<double>>(PyObject *obj) {
    return static_cast<bool>(PyComplex_CheckExact(obj));
}

template<>
bool is<tune_request_t>(PyObject *obj) {
    if (PyFloat_CheckExact(obj)) {
        return true;
    } else if (PyDict_CheckExact(obj)) {
        PyObject *target_freq = PyDict_GetItemString(obj, "target_freq");
        if (target_freq && PyFloat_CheckExact(target_freq)) {
            PyObject *lo_off = PyDict_GetItemString(obj, "lo_off");
            if (lo_off && PyFloat_CheckExact(lo_off))
                return true;
        }
    }
    return false;
}

template<>
bool is<uhd::usrp::subdev_spec_t>(PyObject *obj) {
    return is<std::string>(obj);
}

/******************************************************************************/
/** Functions to translate to <T> from PyObject. **/

template<>
Expect<bool> to<bool>(PyObject *arg) {
    if (is<bool>(arg))
        return static_cast<bool>(PyObject_IsTrue(arg));
    return Error("Expected bool.");
}

template <typename T>
static Expect<T> toUnsignedInt(PyObject *arg) {
    if (is<T>(arg))
        return static_cast<T>(PyLong_AsUnsignedLongMask(arg));
    return Error("Expected integer.");
}

template<>
Expect<uint8_t> to<uint8_t>(PyObject *arg) {
    return toUnsignedInt<uint8_t>(arg);
}

template<>
Expect<uint32_t> to<uint32_t>(PyObject *arg) {
    return toUnsignedInt<uint32_t>(arg);
}

template<>
Expect<uint64_t> to<uint64_t>(PyObject *arg) {
    return toUnsignedInt<uint64_t>(arg);
}

template<>
Expect<double> to<double>(PyObject *arg) {
    if (is<double>(arg))
        return PyFloat_AsDouble(arg);
    return Error("Expected float.");
}

template<>
Expect<std::string> to<std::string>(PyObject *arg) {
    if (is<std::string>(arg)) {
        char *str = PyUnicode_AsUTF8(arg);
        if (str)
            return std::string(str);
        return Error("Failed to get UTF-8 string from object.");
    }
    return Error("Expected string.");
}

template<>
Expect<std::complex<double>> to<std::complex<double>>(PyObject *arg) {
    if (is<std::complex<double>>(arg))
        return std::complex<double>(PyComplex_RealAsDouble(arg), PyComplex_ImagAsDouble(arg));
    return Error("Expected complex.");
}

template<>
Expect<tune_request_t> to<tune_request_t>(PyObject *arg) {
    if (PyFloat_CheckExact(arg)) {
        return tune_request_t(PyFloat_AsDouble(arg));
    } else if (PyDict_CheckExact(arg)) {
        PyObject *target_freq = PyDict_GetItemString(arg, "target_freq");
        if (target_freq && PyFloat_CheckExact(target_freq)) {
            PyObject *lo_off = PyDict_GetItemString(arg, "lo_off");
            if (lo_off && PyFloat_CheckExact(lo_off))
                return tune_request_t(PyFloat_AsDouble(target_freq),
                                      PyFloat_AsDouble(lo_off));
        }
        return Error("Expected dict with {'target_freq': <float>, 'lo_off': <float>}.");
    }
    return Error("Expected float or dict with {'target_freq': <float>, 'lo_off': <float>}.");
}

template<>
Expect<uhd::usrp::subdev_spec_t> to<uhd::usrp::subdev_spec_t>(PyObject *arg) {
    if (is<uhd::usrp::subdev_spec_t>(arg)) {
        char *str = PyUnicode_AsUTF8(arg);
        if (str) {
            try {
                return uhd::usrp::subdev_spec_t(std::string(str));
            } catch(const uhd::exception &e) {
                return Error(e.what());
            }
        }
        return Error("Failed to get UTF-8 string from object.");
    }
    return Error("Expected string.");
}

/******************************************************************************/
/** Functions to translate from <T> to PyObject. **/

PyObject *from(const bool value) {
    if (value)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

PyObject *from(const uint32_t value) {
    return PyLong_FromLong(value);
}

PyObject *from(const uint64_t value) {
    return PyLong_FromLong(value);
}

PyObject *from(const double value) {
    return PyFloat_FromDouble(value);
}

PyObject *from(const std::string &value) {
    return PyUnicode_FromString(value.c_str());
}

inline static bool dict_insert_string_float(PyObject *dict, const char *key, const double &val) {
    PyObject *pval = PyFloat_FromDouble(val);
    if (pval) {
        if (PyDict_SetItemString(dict, key, pval)) {
            Py_DECREF(pval);
            return false;
        }
        return true;
    }
    return false;
}

PyObject *from(const tune_result_t &value) {
    PyObject *ret = PyDict_New();
    if (ret) {
        if (dict_insert_string_float(ret, "clipped_rf_freq", value.clipped_rf_freq)
            && dict_insert_string_float(ret, "target_rf_freq", value.target_rf_freq)
            && dict_insert_string_float(ret, "actual_rf_freq", value.actual_rf_freq)
            && dict_insert_string_float(ret, "target_dsp_freq", value.target_dsp_freq)
            && dict_insert_string_float(ret, "actual_dsp_freq", value.actual_dsp_freq))
            return ret;
        Py_DECREF(ret);
        return PyErr_Format(PyExc_ValueError, "Failed to create dict: error on insert.");
    }
    return PyErr_Format(PyExc_ValueError, "Failed to create dict.");
}

PyObject *from(const uhd::usrp::subdev_spec_t &value) {
    return PyUnicode_FromString(value.to_string().c_str());
}

PyObject *from(const std::vector<std::string> &value) {
    PyObject *ret = PyList_New(value.size());
    if (ret) {
        for (size_t i = 0; i < value.size(); i++) {
            PyObject *pval = PyUnicode_FromString(value[i].c_str());
            if (pval) {
                if (PyList_SetItem(ret, i, pval)) {
                    Py_DECREF(pval);
                    Py_DECREF(ret);
                    return PyErr_Format(PyExc_ValueError, "Failed to create list: error on insert.");
                }
            } else {
                Py_DECREF(ret);
                return PyErr_Format(PyExc_ValueError, "Failed to create list: failed to get string object.");
            }
        }
        return ret;
    }
    return PyErr_Format(PyExc_ValueError, "Failed to create list.");
}

PyObject *from(const dict<std::string, std::string> &value) {
    std::vector<std::string> keys = value.keys();
    PyObject *ret = PyDict_New();
    if (ret) {
        for (const auto &key : keys) {
            PyObject *pval = PyUnicode_FromString(value[key].c_str());
            if (pval) {
                if (PyDict_SetItemString(ret, key.c_str(), pval)) {
                    Py_DECREF(pval);
                    Py_DECREF(ret);
                    return PyErr_Format(PyExc_ValueError, "Failed to create dict: error on insert.");
                }
            } else {
                Py_DECREF(ret);
                return PyErr_Format(PyExc_ValueError, "Failed to create dict: failed to get string object.");
            }
        }
        return ret;
    }
    return PyErr_Format(PyExc_ValueError, "Failed to create dict.");
}

PyObject *from(const meta_range_t &value) {
    PyObject *ret = PyDict_New();
    if (ret) {
        if (dict_insert_string_float(ret, "start", value.start())
            && dict_insert_string_float(ret, "stop", value.stop())
            && dict_insert_string_float(ret, "step", value.step()))
            return ret;
        Py_DECREF(ret);
        return PyErr_Format(PyExc_ValueError, "Failed to create dict: error on insert.");
    }
    return PyErr_Format(PyExc_ValueError, "Failed to create dict.");
}

}
