#pragma once

#include <array>
#include <complex>
#include <string>
#include <type_traits>
#include <vector>

#include <Python.h>

#include <uhd/exception.hpp>
#include <uhd/types/dict.hpp>
#include <uhd/types/ranges.hpp>
#include <uhd/types/sensors.hpp>
#include <uhd/types/time_spec.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/types/tune_result.hpp>
#include <uhd/usrp/subdev_spec.hpp>

#include "uhd_expect.hpp"
#include "uhd_timespec.hpp"

namespace uhd {

template<typename T, typename Enable = void>
struct type_t;

/**
 * Specialization for unsigned integral types
 */
template<typename T>
struct type_t<T, typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value>::type>
{
    static bool check(PyObject *obj)
    {
        return PyLong_CheckExact(obj);
    }

    static Expect<T> to(PyObject *obj)
    {
        if (check(obj))
        {
            unsigned long long value = PyLong_AsUnsignedLongLong(obj);
            if (PyErr_Occurred())
            {
                if (PyErr_ExceptionMatches(PyExc_OverflowError))
                {
                    return Error("Integer overflow exception.");
                }
                else if (PyErr_ExceptionMatches(PyExc_TypeError))
                {
                    return Error("Integer type error.");
                }
                else
                {
                    return Error("Unexpected error occurred.");
                }
            }

            T ret = static_cast<T>(value);
            if (static_cast<unsigned long long>(ret) != value)
            {
                return Error("Integer value overflow.");
            }

            return ret;
        }
        else
        {
            return Error("Expected integer.");
        }
    }

    static PyObject *from(const T value)
    {
        return PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(value));
    }
};

/**
 * Specialization for signed integral types
 */
template<typename T>
struct type_t<T, typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value>::type>
{
    static bool check(PyObject *obj)
    {
        return PyLong_CheckExact(obj);
    }

    static Expect<T> to(PyObject *obj)
    {
        if (check(obj))
        {
            int overflow = 0;
            long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
            if (overflow != 0)
            {
                return Error("Integer overflow occurred.");
            }
            else if (PyErr_Occurred())
            {
                if (PyErr_ExceptionMatches(PyExc_OverflowError))
                {
                    return Error("Integer overflow exception.");
                }
                else if (PyErr_ExceptionMatches(PyExc_TypeError))
                {
                    return Error("Integer type error.");
                }
                else
                {
                    return Error("Unexpected error occurred.");
                }
            }

            T ret = static_cast<T>(value);
            if (static_cast<long long>(ret) != value)
            {
                return Error("Integer value overflow.");
            }

            return ret;
        }
        else
        {
            return Error("Expected integer.");
        }
    }

    static PyObject *from(const T value)
    {
        return PyLong_FromLongLong(static_cast<long long>(value));
    }
};

/**
 * Specialization for bool
 */
template<>
struct type_t<bool>
{
    static bool check(PyObject *obj)
    {
        return PyBool_Check(obj);
    }

    static Expect<bool> to(PyObject *obj)
    {
        if (check(obj))
        {
            return static_cast<bool>(PyObject_IsTrue(obj));
        }
        return Error("Expected bool.");
    }

    static PyObject *from(const bool value)
    {
        if (value)
        {
            Py_RETURN_TRUE;
        }
        else
        {
            Py_RETURN_FALSE;
        }
    }
};

/**
 * Specialization for double
 */
template<>
struct type_t<double>
{
    static bool check(PyObject *obj)
    {
        return PyFloat_CheckExact(obj);
    }

    static Expect<double> to(PyObject *obj)
    {
        if (check(obj))
        {
            return PyFloat_AsDouble(obj);
        }
        return Error("Expected float.");
    }

    static PyObject *from(const double value)
    {
        return PyFloat_FromDouble(value);
    }
};

/**
 * Specialization for std::string
 */
template<>
struct type_t<std::string>
{
    static bool check(PyObject *obj)
    {
        return static_cast<bool>(PyUnicode_CheckExact(obj));
    }

    static Expect<std::string> to(PyObject *obj)
    {
        if (check(obj))
        {
            const char *str = PyUnicode_AsUTF8(obj);
            if (str)
            {
                return std::string(str);
            }
            return Error("Failed to get UTF-8 string from object.");
        }
        return Error("Expected string.");
    }

    static PyObject *from(const std::string &value)
    {
        return PyUnicode_FromString(value.c_str());
    }
};

/**
 * Specialization for std::complex<double>
 */
template<>
struct type_t<std::complex<double>>
{
    static bool check(PyObject *obj)
    {
        return static_cast<bool>(PyComplex_CheckExact(obj));
    }

    static Expect<std::complex<double>> to(PyObject *obj)
    {
        if (check(obj))
        {
            return std::complex<double>(PyComplex_RealAsDouble(obj),
                                        PyComplex_ImagAsDouble(obj));
        }
        return Error("Expected complex.");
    }
};

/**
 * Specialization for uhd::tune_request_t::policy_t
 */
template<>
struct type_t<tune_request_t::policy_t>
{
    struct StringPolicyPair
    {
        const char *str;
        tune_request_t::policy_t policy;
    };

    static const std::array<StringPolicyPair,3>& tune_request_policies()
    {
        static const std::array<StringPolicyPair,3> policies = {{
            {"none", tune_request_t::policy_t::POLICY_NONE},
            {"auto", tune_request_t::policy_t::POLICY_AUTO},
            {"manual", tune_request_t::policy_t::POLICY_MANUAL},
        }};
        return policies;
    }

    static bool check(PyObject *obj)
    {
        if (!PyUnicode_CheckExact(obj))
        {
            return false;
        }

        const char *c_str = PyUnicode_AsUTF8(obj);
        if (c_str == nullptr)
        {
            return false;
        }

        for (const auto &pair : tune_request_policies())
        {
            if (strcmp(pair.str, c_str) == 0)
            {
                // Match found: is valid policy.
                return true;
            }
        }
        // No match found: is not valid policy.
        return false;
    }

    static Expect<tune_request_t::policy_t> to(PyObject *obj)
    {
        if (PyUnicode_CheckExact(obj))
        {
            const char *c_str = PyUnicode_AsUTF8(obj);
            if (c_str)
            {
                for (const auto &pair : tune_request_policies())
                {
                    if (strcmp(pair.str, c_str) == 0)
                    {
                        return pair.policy;
                    }
                }
            }
        }
        return Error("Expected string: {'none', 'auto', 'manual'}.");
    }
};

/**
 * Specialization for uhd::device_addr_t
 */
template<>
struct type_t<device_addr_t>
{
    static bool check(PyObject *obj)
    {
        return static_cast<bool>(PyUnicode_CheckExact(obj));
    }

    static Expect<device_addr_t> to(PyObject *obj)
    {
        if (check(obj)) {
            const char *str = PyUnicode_AsUTF8(obj);
            if (str) {
                try
                {
                    return device_addr_t(std::string(str));
                }
                catch (const exception &e)
                {
                    return Error(e.what());
                }
            }
            return Error("Failed to get UTF-8 string from object.");
        }
        return Error("Expected string.");
    }
};

/**
 * Specialization for uhd::tune_request_t
 */
template<>
struct type_t<tune_request_t>
{
    static bool check(PyObject *obj)
    {
        if (PyFloat_CheckExact(obj))
        {
            return true;
        }
        else if (PyDict_CheckExact(obj))
        {
            /** target_freq / float / required **/
            PyObject *target_freq = PyDict_GetItemString(obj, "target_freq");
            if (target_freq && PyFloat_CheckExact(target_freq)) {
                /** lo_off / float / optional **/
                PyObject *lo_off = PyDict_GetItemString(obj, "lo_off");
                if (!lo_off || !PyFloat_CheckExact(lo_off))
                    return false;
                /** rf_freq_policy / str / optional **/
                PyObject *rf_freq_policy = PyDict_GetItemString(obj, "rf_freq_policy");
                if (!rf_freq_policy || !type_t<tune_request_t::policy_t>::check(rf_freq_policy))
                    return false;
                /** rf_freq / double / optional **/
                PyObject *rf_freq = PyDict_GetItemString(obj, "rf_freq");
                if (!rf_freq || !PyFloat_CheckExact(rf_freq))
                    return false;
                /** dsp_freq_policy / str / optional **/
                PyObject *dsp_freq_policy = PyDict_GetItemString(obj, "dsp_freq_policy");
                if (!dsp_freq_policy || !type_t<tune_request_t::policy_t>::check(dsp_freq_policy))
                    return false;
                /** dsp_freq / double / optional **/
                PyObject *dsp_freq = PyDict_GetItemString(obj, "dsp_freq");
                if (!dsp_freq || !PyFloat_CheckExact(dsp_freq))
                    return false;
                /** args / str / optional **/
                PyObject *args = PyDict_GetItemString(obj, "args");
                if (!args || !type_t<device_addr_t>::check(args))
                    return false;
                return true;
            }
        }
        return false;
    }

    static Expect<tune_request_t> to(PyObject *obj)
    {
        if (PyFloat_CheckExact(obj))
        {
            return tune_request_t(PyFloat_AsDouble(obj));
        }
        else if (PyDict_CheckExact(obj))
        {
            PyObject *target_freq_ = PyDict_GetItemString(obj, "target_freq");
            if (target_freq_) {
                Expect<double> target_freq = type_t<double>::to(target_freq_);
                if (!target_freq)
                    return Error("target_freq: " + std::string(target_freq.what()));

                /** lo_off / str / optional **/
                tune_request_t ret;
                PyObject *lo_off_ = PyDict_GetItemString(obj, "lo_off");
                if (lo_off_) {
                    Expect<double> lo_off = type_t<double>::to(lo_off_);
                    if (!lo_off)
                        return Error("lo_off: " + std::string(lo_off.what()));
                    ret = tune_request_t(target_freq.get(), lo_off.get());
                } else {
                    ret = tune_request_t(target_freq.get());
                }

                PyObject *rf_freq_policy_ = PyDict_GetItemString(obj, "rf_freq_policy");
                PyObject *rf_freq_ = PyDict_GetItemString(obj, "rf_freq");
                PyObject *dsp_freq_policy_ = PyDict_GetItemString(obj, "dsp_freq_policy");
                PyObject *dsp_freq_ = PyDict_GetItemString(obj, "dsp_freq");
                PyObject *args_ = PyDict_GetItemString(obj, "args");

                /** rf_freq_policy / str / optional **/
                if (rf_freq_policy_) {
                    Expect<tune_request_t::policy_t> rf_freq_policy = type_t<tune_request_t::policy_t>::to(rf_freq_policy_);
                    if (!rf_freq_policy)
                        return Error("rf_freq_policy: " + std::string(rf_freq_policy.what()));
                    ret.rf_freq_policy = rf_freq_policy.get();
                }
                /** rf_freq / double / optional **/
                if (rf_freq_) {
                    Expect<double> rf_freq = type_t<double>::to(rf_freq_);
                    if (!rf_freq)
                        return Error("rf_freq: " + std::string(rf_freq.what()));
                    ret.rf_freq = rf_freq.get();
                }
                /** dsp_freq_policy / str / optional **/
                if (dsp_freq_policy_) {
                    Expect<tune_request_t::policy_t> dsp_freq_policy = type_t<tune_request_t::policy_t>::to(dsp_freq_policy_);
                    if (!dsp_freq_policy)
                        return Error("dsp_freq_policy: " + std::string(dsp_freq_policy.what()));
                    ret.dsp_freq_policy = dsp_freq_policy.get();
                }
                /** dsp_freq / double / optional **/
                if (dsp_freq_) {
                    Expect<double> dsp_freq = type_t<double>::to(dsp_freq_);
                    if (!dsp_freq)
                        return Error("dsp_freq: " + std::string(dsp_freq.what()));
                    ret.dsp_freq = dsp_freq.get();
                }
                /** args / str / optional **/
                if (args_) {
                    Expect<device_addr_t> args = type_t<device_addr_t>::to(args_);
                    if (!args)
                        return Error("args: " + std::string(args.what()));
                    ret.args = args.get();
                }

                return ret;
            }
        }
        return Error("Expected float or dict with {'target_freq': <float>, 'lo_off': <float>}.");
    }
};

/**
 * Specialization for uhd::uspr::subdev_spec_t
 */
template<>
struct type_t<usrp::subdev_spec_t>
{
    static bool check(PyObject *obj)
    {
        return static_cast<bool>(PyUnicode_CheckExact(obj));
    }

    static Expect<usrp::subdev_spec_t> to(PyObject *obj)
    {
        if (check(obj)) {
            const char *str = PyUnicode_AsUTF8(obj);
            if (str) {
                try
                {
                    return usrp::subdev_spec_t(std::string(str));
                }
                catch (const exception &e)
                {
                    return Error(e.what());
                }
            }
            return Error("Failed to get UTF-8 string from object.");
        }
        return Error("Expected string.");
    }

    static PyObject *from(const usrp::subdev_spec_t &value)
    {
        return PyUnicode_FromString(value.to_string().c_str());
    }
};

/**
 * Specialization for uhd::time_spec_t
 */
template<>
struct type_t<time_spec_t>
{
    static bool check(PyObject *obj)
    {
        return (PyFloat_CheckExact(obj) ||
                PyObject_TypeCheck(obj, &TimeSpecType));
    }

    static Expect<time_spec_t> to(PyObject *arg)
    {
        if (PyFloat_CheckExact(arg)) {
            return time_spec_t(PyFloat_AsDouble(arg));
        } else if (PyObject_TypeCheck(arg, &TimeSpecType)) {
            return reinterpret_cast<TimeSpec *>(arg)->_time_spec;
        }
        return Error("Expected float or dict with {'integer': <int>, 'fractional': <float>}.");
    }

    static PyObject *from(const time_spec_t &value)
    {
        PyObject *ret = reinterpret_cast<PyObject *>(TimeSpec_from_time_spec_t(value));
        if (!ret)
            return PyErr_Format(PyExc_ValueError, "Failed to create TimeSpec.");
        return ret;
    }
};

static bool dict_insert(PyObject *dict, const char *key, const double val)
{
    PyObject *pval = PyFloat_FromDouble(val);
    if (pval)
    {
        if (PyDict_SetItemString(dict, key, pval))
        {
            Py_DECREF(pval);
            return false;
        }
        Py_DECREF(pval);
        return true;
    }
    return false;
}

static bool dict_insert(PyObject *dict, const char *key, const std::string &val)
{
    PyObject *pval = PyUnicode_FromString(val.c_str());
    if (pval)
    {
        if (PyDict_SetItemString(dict, key, pval))
        {
            Py_DECREF(pval);
            return false;
        }
        Py_DECREF(pval);
        return true;
    }
    return false;
}

/**
 * Specialization for uhd::tune_result_t
 */
template<>
struct type_t<tune_result_t>
{
    static PyObject *from(const tune_result_t &value)
    {
        PyObject *ret = PyDict_New();
        if (ret)
        {
            if (dict_insert(ret, "clipped_rf_freq", value.clipped_rf_freq)
                && dict_insert(ret, "target_rf_freq", value.target_rf_freq)
                && dict_insert(ret, "actual_rf_freq", value.actual_rf_freq)
                && dict_insert(ret, "target_dsp_freq", value.target_dsp_freq)
                && dict_insert(ret, "actual_dsp_freq", value.actual_dsp_freq))
            {
                return ret;
            }
            Py_DECREF(ret);
            return PyErr_Format(PyExc_ValueError, "Failed to create dict: error on insert.");
        }
        return PyErr_Format(PyExc_ValueError, "Failed to create dict.");
    }
};

/**
 * Specialization for std::vector<std::string>
 */
template<>
struct type_t<std::vector<std::string>>
{
    static PyObject *from(const std::vector<std::string> &value)
    {
        PyObject *ret = PyList_New(value.size());
        if (ret)
        {
            for (size_t i = 0; i < value.size(); i++)
            {
                PyObject *pval = PyUnicode_FromString(value[i].c_str());
                if (pval)
                {
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
};

/**
 * Specialization for uhd::dict<std::string, std::string>
 */
template<>
struct type_t<dict<std::string, std::string>>
{
    static PyObject *from(const dict<std::string, std::string> &value)
    {
        const std::vector<std::string> &keys = value.keys();
        const std::vector<std::string> &values = value.vals();
        PyObject *ret = PyDict_New();
        if (ret)
        {
            for (size_t i = 0; i < keys.size(); i++)
            {
                PyObject *pval = PyUnicode_FromString(values[i].c_str());
                if (pval)
                {
                    if (PyDict_SetItemString(ret, keys[i].c_str(), pval))
                    {
                        Py_DECREF(pval);
                        Py_DECREF(ret);
                        return PyErr_Format(PyExc_ValueError, "Failed to create dict: error on insert.");
                    }
                    Py_DECREF(pval);
                }
                else
                {
                    Py_DECREF(ret);
                    return PyErr_Format(PyExc_ValueError, "Failed to create dict: failed to get string object.");
                }
            }
            return ret;
        }
        return PyErr_Format(PyExc_ValueError, "Failed to create dict.");
    }
};

/**
 * Specialization for uhd::meta_range_t
 */
template<>
struct type_t<meta_range_t>
{
    static PyObject *from(const meta_range_t &value)
    {
        if (value.empty())
        {
            /** meta-range is empty. **/
            Py_INCREF(Py_None);
            return Py_None;
        }
        PyObject *ret = PyDict_New();
        if (ret)
        {
            if (dict_insert(ret, "start", value.start())
                && dict_insert(ret, "stop", value.stop())
                && dict_insert(ret, "step", value.step()))
                return ret;
            Py_DECREF(ret);
            return PyErr_Format(PyExc_ValueError, "Failed to create dict: error on insert.");
        }
        return PyErr_Format(PyExc_ValueError, "Failed to create dict.");
    }
};

/**
 * Specialization for uhd::sensor_value_t
 */
template<>
struct type_t<sensor_value_t>
{
    static PyObject *from(const sensor_value_t &value)
    {
        const sensor_value_t::sensor_map_t &map = value.to_map();
        PyObject *ret = PyDict_New();
        if (!ret)
        {
            return PyErr_Format(PyExc_ValueError, "Failed to create dict.");
        }
        for (const auto &kv : map)
        {
            if (!dict_insert(ret, kv.first.c_str(), kv.second))
            {
                Py_DECREF(ret);
                return PyErr_Format(PyExc_ValueError, "Failed to create dict: error on insert.");
            }
        }
        return ret;
    }
};

/******************************************************************************/
/*  Function to determine if PyObject is of type <T>.                         */
/******************************************************************************/

template<typename T>
bool is(PyObject *obj)
{
    return type_t<T>::check(obj);
}

/******************************************************************************/
/*  Function to translate to <T> from PyObject.                               */
/******************************************************************************/

template<typename T>
Expect<T> to(PyObject *obj)
{
    return type_t<T>::to(obj);
}

/******************************************************************************/
/*  Function to translate from <T> to PyObject.                               */
/******************************************************************************/

template<typename T>
PyObject *from(const T &value)
{
    return type_t<T>::from(value);
}

}
