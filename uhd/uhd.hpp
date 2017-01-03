#ifndef __UHD_HPP__
#define __UHD_HPP__

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

    extern PyObject *UhdError;

}

#endif  /** __UHD_HPP__ **/
