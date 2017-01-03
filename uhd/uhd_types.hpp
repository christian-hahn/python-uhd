#ifndef __UHD_TYPES_HPP__
#define __UHD_TYPES_HPP__

#include <string>
#include <vector>
#include <complex>

#include <Python.h>

#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
#include <uhd/types/dict.hpp>

#include "uhd_expect.hpp"

namespace uhd {

/******************************************************************************/
/** Functions to determine if PyObject is of type <T>. **/

template<class T>
bool is(PyObject *obj);
template<>
bool is<bool>(PyObject *obj);
template<>
bool is<uint8_t>(PyObject *obj);
template<>
bool is<uint32_t>(PyObject *obj);
template<>
bool is<uint64_t>(PyObject *obj);
template<>
bool is<double>(PyObject *obj);
template<>
bool is<std::string>(PyObject *obj);
template<>
bool is<std::complex<double>>(PyObject *obj);
template<>
bool is<tune_request_t>(PyObject *obj);
template<>
bool is<uhd::usrp::subdev_spec_t>(PyObject *obj);

/******************************************************************************/
/** Functions to translate to <T> from PyObject. **/

template<class T>
Expect<T> to(PyObject *arg);
template<>
Expect<bool> to<bool>(PyObject *arg);
template<>
Expect<uint8_t> to<uint8_t>(PyObject *arg);
template<>
Expect<uint32_t> to<uint32_t>(PyObject *arg);
template<>
Expect<uint64_t> to<uint64_t>(PyObject *arg);
template<>
Expect<double> to<double>(PyObject *arg);
template<>
Expect<std::string> to<std::string>(PyObject *arg);
template<>
Expect<std::complex<double>> to<std::complex<double>>(PyObject *arg);
template<>
Expect<tune_request_t> to<tune_request_t>(PyObject *arg);
template<>
Expect<uhd::usrp::subdev_spec_t> to<uhd::usrp::subdev_spec_t>(PyObject *arg);

/******************************************************************************/
/** Functions to translate from <T> to PyObject. **/

PyObject *from(const bool value);
PyObject *from(const uint32_t value);
PyObject *from(const uint64_t value);
PyObject *from(const double value);
PyObject *from(const std::string &value);
PyObject *from(const tune_result_t &value);
PyObject *from(const uhd::usrp::subdev_spec_t &value);
PyObject *from(const std::vector<std::string> &value);
PyObject *from(const dict<std::string, std::string> &value);
PyObject *from(const meta_range_t &value);

}

#endif  /** __UHD_TYPES_HPP__ **/
