#ifndef __UHD_OBJECT_HPP__
#define __UHD_OBJECT_HPP__

#include <mutex>

#include <Python.h>

#include <uhd/usrp/multi_usrp.hpp>

#include "uhd_rx.hpp"

namespace uhd {

    typedef struct {
        PyObject_HEAD
        uhd::usrp::multi_usrp::sptr dev;
        std::mutex dev_lock;
        ReceiveWorker *receiver;
    } Uhd;

    int Uhd_register_type(PyObject *module);

}

#endif  /** __UHD_OBJECT_HPP__ **/
