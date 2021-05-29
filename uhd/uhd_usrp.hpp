#ifndef __UHD_USRP_HPP__
#define __UHD_USRP_HPP__

#include <mutex>

#include <Python.h>

#include <uhd/usrp/multi_usrp.hpp>

#include "uhd_rx.hpp"
#include "uhd_tx.hpp"

namespace uhd {

    typedef struct {
        PyObject_HEAD
        uhd::usrp::multi_usrp::sptr dev;
        std::mutex dev_lock;
        ReceiveWorker *receiver;
        TransmitWorker *transmitter;
    } Usrp;

    int Usrp_register_type(PyObject *module);

}

#endif  /** __UHD_USRP_HPP__ **/
