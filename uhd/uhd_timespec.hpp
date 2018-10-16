#ifndef __UHD_TIMESPEC_HPP__
#define __UHD_TIMESPEC_HPP__

#include <Python.h>

#include <uhd/types/time_spec.hpp>

namespace uhd {

    typedef struct {
        PyObject_HEAD
        time_spec_t _time_spec;
    } TimeSpec;

    int TimeSpec_register_type(PyObject *module);
    TimeSpec *TimeSpec_from_time_spec_t(const time_spec_t &value);

    extern PyTypeObject TimeSpecType;

}

#endif  /** __UHD_TIMESPEC_HPP__ **/
