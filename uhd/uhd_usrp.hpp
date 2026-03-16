/**
 * @file   uhd_usrp.hpp
 * @brief  Defines Usrp object.
 * @author Christian Hahn
 */

#pragma once

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

} // namespace uhd
