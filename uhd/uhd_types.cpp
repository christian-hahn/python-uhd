#include <Python.h>

#include <uhd/exception.hpp>

#include "uhd_types.hpp"
#include "uhd_timespec.hpp"

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
bool is<uint16_t>(PyObject *obj) {
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
bool is<int8_t>(PyObject *obj) {
    return static_cast<bool>(PyLong_CheckExact(obj));
}

template<>
bool is<int16_t>(PyObject *obj) {
    return static_cast<bool>(PyLong_CheckExact(obj));
}

template<>
bool is<int32_t>(PyObject *obj) {
    return static_cast<bool>(PyLong_CheckExact(obj));
}

template<>
bool is<int64_t>(PyObject *obj) {
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

static const std::array<std::pair<std::string,tune_request_t::policy_t>,3> tune_request_policies = {{
    {"none", tune_request_t::policy_t::POLICY_NONE},
    {"auto", tune_request_t::policy_t::POLICY_AUTO},
    {"manual", tune_request_t::policy_t::POLICY_MANUAL},
}};

template<>
bool is<tune_request_t::policy_t>(PyObject *obj) {
    if (!is<std::string>(obj))
        return false;
    Expect<std::string> req;
    if (!(req = to<std::string>(obj)))
        return false;
    return (std::find_if(tune_request_policies.begin(),
            tune_request_policies.end(),
            [&req](const std::pair<std::string,tune_request_t::policy_t> &v) {
                return v.first == req.get();
            }) != tune_request_policies.end());
}

template<>
bool is<device_addr_t>(PyObject *obj) {
    return is<std::string>(obj);
}

template<>
bool is<tune_request_t>(PyObject *obj) {
    if (PyFloat_CheckExact(obj)) {
        return true;
    } else if (PyDict_CheckExact(obj)) {
        /** target_freq / float / required **/
        PyObject *target_freq = PyDict_GetItemString(obj, "target_freq");
        if (target_freq && PyFloat_CheckExact(target_freq)) {
            /** lo_off / float / optional **/
            PyObject *lo_off = PyDict_GetItemString(obj, "lo_off");
            if (!lo_off || !PyFloat_CheckExact(lo_off))
                return false;
            /** rf_freq_policy / str / optional **/
            PyObject *rf_freq_policy = PyDict_GetItemString(obj, "rf_freq_policy");
            if (!rf_freq_policy || !is<tune_request_t::policy_t>(rf_freq_policy))
                return false;
            /** rf_freq / double / optional **/
            PyObject *rf_freq = PyDict_GetItemString(obj, "rf_freq");
            if (!rf_freq || !PyFloat_CheckExact(rf_freq))
                return false;
            /** dsp_freq_policy / str / optional **/
            PyObject *dsp_freq_policy = PyDict_GetItemString(obj, "dsp_freq_policy");
            if (!dsp_freq_policy || !is<tune_request_t::policy_t>(dsp_freq_policy))
                return false;
            /** dsp_freq / double / optional **/
            PyObject *dsp_freq = PyDict_GetItemString(obj, "dsp_freq");
            if (!dsp_freq || !PyFloat_CheckExact(dsp_freq))
                return false;
            /** args / str / optional **/
            PyObject *args = PyDict_GetItemString(obj, "args");
            if (!args || !is<device_addr_t>(args))
                return false;
            return true;
        }
    }
    return false;
}

template<>
bool is<usrp::subdev_spec_t>(PyObject *obj) {
    return is<std::string>(obj);
}

template<>
bool is<time_spec_t>(PyObject *obj) {
    if (PyFloat_CheckExact(obj)) {
        return true;
    } else if (PyObject_TypeCheck(obj, &TimeSpecType)) {
        return true;
    }
    return false;
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
        return static_cast<T>(PyLong_AsUnsignedLongLongMask(arg));
    return Error("Expected integer.");
}

template <typename T>
static Expect<T> toSignedInt(PyObject *arg) {
    if (is<T>(arg)) {
        int overflow;
        T value = static_cast<T>(PyLong_AsLongLongAndOverflow(arg, &overflow));
        if (overflow)
            return Error("Integer out-of-range.");
        return value;
    }
    return Error("Expected integer.");
}

template<>
Expect<uint8_t> to<uint8_t>(PyObject *arg) {
    return toUnsignedInt<uint8_t>(arg);
}

template<>
Expect<uint16_t> to<uint16_t>(PyObject *arg) {
    return toUnsignedInt<uint16_t>(arg);
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
Expect<int8_t> to<int8_t>(PyObject *arg) {
    return toSignedInt<int8_t>(arg);
}

template<>
Expect<int16_t> to<int16_t>(PyObject *arg) {
    return toSignedInt<int16_t>(arg);
}

template<>
Expect<int32_t> to<int32_t>(PyObject *arg) {
    return toSignedInt<int32_t>(arg);
}

template<>
Expect<int64_t> to<int64_t>(PyObject *arg) {
    return toSignedInt<int64_t>(arg);
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
        const char *str = PyUnicode_AsUTF8(arg);
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
Expect<tune_request_t::policy_t> to<tune_request_t::policy_t>(PyObject *arg) {
    if (is<std::string>(arg)) {
        const char *c_str = PyUnicode_AsUTF8(arg);
        if (c_str) {
            const std::string str(c_str);
            const auto &ret = std::find_if(tune_request_policies.begin(), tune_request_policies.end(),
                                     [&str](const std::pair<std::string,tune_request_t::policy_t> &v) {
                                     return v.first == str;});
            if (ret != tune_request_policies.end())
                return ret->second;
        }
    }
    return Error("Expected string: {'none', 'auto', 'manual'}.");
}

template<>
Expect<device_addr_t> to<device_addr_t>(PyObject *arg) {
    if (is<device_addr_t>(arg)) {
        const char *str = PyUnicode_AsUTF8(arg);
        if (str) {
            try {
                return device_addr_t(std::string(str));
            } catch (const uhd::exception &e) {
                return Error(e.what());
            }
        }
        return Error("Failed to get UTF-8 string from object.");
    }
    return Error("Expected string.");
}

template<>
Expect<tune_request_t> to<tune_request_t>(PyObject *arg) {
    if (PyFloat_CheckExact(arg)) {
        return tune_request_t(PyFloat_AsDouble(arg));
    } else if (PyDict_CheckExact(arg)) {
        PyObject *target_freq_ = PyDict_GetItemString(arg, "target_freq");
        if (target_freq_) {
            Expect<double> target_freq = to<double>(target_freq_);
            if (!target_freq)
                return Error("target_freq: " + std::string(target_freq.what()));

            /** lo_off / str / optional **/
            tune_request_t ret;
            PyObject *lo_off_ = PyDict_GetItemString(arg, "lo_off");
            if (lo_off_) {
                Expect<double> lo_off = to<double>(lo_off_);
                if (!lo_off)
                    return Error("lo_off: " + std::string(lo_off.what()));
                ret = tune_request_t(target_freq.get(), lo_off.get());
            } else {
                ret = tune_request_t(target_freq.get());
            }

            PyObject *rf_freq_policy_ = PyDict_GetItemString(arg, "rf_freq_policy");
            PyObject *rf_freq_ = PyDict_GetItemString(arg, "rf_freq");
            PyObject *dsp_freq_policy_ = PyDict_GetItemString(arg, "dsp_freq_policy");
            PyObject *dsp_freq_ = PyDict_GetItemString(arg, "dsp_freq");
            PyObject *args_ = PyDict_GetItemString(arg, "args");

            /** rf_freq_policy / str / optional **/
            if (rf_freq_policy_) {
                Expect<tune_request_t::policy_t> rf_freq_policy = to<tune_request_t::policy_t>(rf_freq_policy_);
                if (!rf_freq_policy)
                    return Error("rf_freq_policy: " + std::string(rf_freq_policy.what()));
                ret.rf_freq_policy = rf_freq_policy.get();
            }
            /** rf_freq / double / optional **/
            if (rf_freq_) {
                Expect<double> rf_freq = to<double>(rf_freq_);
                if (!rf_freq)
                    return Error("rf_freq: " + std::string(rf_freq.what()));
                ret.rf_freq = rf_freq.get();
            }
            /** dsp_freq_policy / str / optional **/
            if (dsp_freq_policy_) {
                Expect<tune_request_t::policy_t> dsp_freq_policy = to<tune_request_t::policy_t>(dsp_freq_policy_);
                if (!dsp_freq_policy)
                    return Error("dsp_freq_policy: " + std::string(dsp_freq_policy.what()));
                ret.dsp_freq_policy = dsp_freq_policy.get();
            }
            /** dsp_freq / double / optional **/
            if (dsp_freq_) {
                Expect<double> dsp_freq = to<double>(dsp_freq_);
                if (!dsp_freq)
                    return Error("dsp_freq: " + std::string(dsp_freq.what()));
                ret.dsp_freq = dsp_freq.get();
            }
            /** args / str / optional **/
            if (args_) {
                Expect<device_addr_t> args = to<device_addr_t>(args_);
                if (!args)
                    return Error("args: " + std::string(args.what()));
                ret.args = args.get();
            }

            return ret;
        }
    }
    return Error("Expected float or dict with {'target_freq': <float>, 'lo_off': <float>}.");
}

template<>
Expect<usrp::subdev_spec_t> to<usrp::subdev_spec_t>(PyObject *arg) {
    if (is<usrp::subdev_spec_t>(arg)) {
        const char *str = PyUnicode_AsUTF8(arg);
        if (str) {
            try {
                return usrp::subdev_spec_t(std::string(str));
            } catch (const uhd::exception &e) {
                return Error(e.what());
            }
        }
        return Error("Failed to get UTF-8 string from object.");
    }
    return Error("Expected string.");
}

template<>
Expect<time_spec_t> to<time_spec_t>(PyObject *arg) {
    if (PyFloat_CheckExact(arg)) {
        return time_spec_t(PyFloat_AsDouble(arg));
    } else if (PyObject_TypeCheck(arg, &TimeSpecType)) {
        return reinterpret_cast<TimeSpec *>(arg)->_time_spec;
    }
    return Error("Expected float or dict with {'integer': <int>, 'fractional': <float>}.");
}

/******************************************************************************/
/** Functions to translate from <T> to PyObject. **/

PyObject *from(const bool value) {
    if (value)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

PyObject *from(const uint8_t value) {
    return PyLong_FromUnsignedLong(value);
}

PyObject *from(const uint16_t value) {
    return PyLong_FromUnsignedLong(value);
}

PyObject *from(const uint32_t value) {
    return PyLong_FromUnsignedLong(value);
}

PyObject *from(const uint64_t value) {
    return PyLong_FromUnsignedLongLong(value);
}

PyObject *from(const int8_t value) {
    return PyLong_FromLong(value);
}

PyObject *from(const int16_t value) {
    return PyLong_FromLong(value);
}

PyObject *from(const int32_t value) {
    return PyLong_FromLong(value);
}

PyObject *from(const int64_t value) {
    return PyLong_FromLongLong(value);
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
        Py_DECREF(pval);
        return true;
    }
    return false;
}

inline static bool dict_insert_string_long_long(PyObject *dict, const char *key, const long long &val) {
    PyObject *pval = PyLong_FromLongLong(val);
    if (pval) {
        if (PyDict_SetItemString(dict, key, pval)) {
            Py_DECREF(pval);
            return false;
        }
        Py_DECREF(pval);
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

PyObject *from(const usrp::subdev_spec_t &value) {
    return PyUnicode_FromString(value.to_string().c_str());
}

PyObject *from(const std::vector<std::string> &value) {
    PyObject *ret = PyList_New(value.size());
    if (ret) {
        for (size_t i = 0; i < value.size(); i++) {
            PyObject *pval = PyUnicode_FromString(value[i].c_str());
            if (pval) {
                PyList_SET_ITEM(ret, i, pval);
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
    const std::vector<std::string> &keys = value.keys();
    const std::vector<std::string> &values = value.vals();
    PyObject *ret = PyDict_New();
    if (ret) {
        for (size_t i = 0; i < keys.size(); i++) {
            PyObject *pval = PyUnicode_FromString(values[i].c_str());
            if (pval) {
                if (PyDict_SetItemString(ret, keys[i].c_str(), pval)) {
                    Py_DECREF(pval);
                    Py_DECREF(ret);
                    return PyErr_Format(PyExc_ValueError, "Failed to create dict: error on insert.");
                }
                Py_DECREF(pval);
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
    if (value.empty()) {
        /** meta-range is empty. **/
        Py_INCREF(Py_None);
        return Py_None;
    }
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

PyObject *from(const time_spec_t &value) {
    PyObject *ret = reinterpret_cast<PyObject *>(TimeSpec_from_time_spec_t(value));
    if (!ret)
        return PyErr_Format(PyExc_ValueError, "Failed to create TimeSpec.");
    return ret;
}

}
