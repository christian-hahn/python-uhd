#include <Python.h>

#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
#include <uhd/types/dict.hpp>

#include "uhd.hpp"
#include "uhd_types.hpp"
#include "uhd_gen.hpp"

namespace uhd {

PyObject *Uhd_set_rx_lo_export_enabled(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 3)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...3].", nargs);

    Expect<bool> enabled;
    if (!(enabled = to<bool>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", enabled.what());

    Expect<std::string> name;
    Expect<size_t> chan;

    if (nargs > 1 && !(name = to<std::string>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", name.what());
    if (nargs > 2 && !(chan = to<size_t>(PyTuple_GetItem(args, 2))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 3: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 3)
            self->dev->set_rx_lo_export_enabled(enabled.get(), name.get(), chan.get());
        else if (nargs == 2)
            self->dev->set_rx_lo_export_enabled(enabled.get(), name.get());
        else
            self->dev->set_rx_lo_export_enabled(enabled.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_get_usrp_tx_info(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    dict<std::string, std::string> ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_usrp_tx_info(chan.get());
        else
            ret = self->dev->get_usrp_tx_info();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

static PyObject *_get_rx_gain_0(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<std::string> name;
    if (!(name = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", name.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    double ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            ret = self->dev->get_rx_gain(name.get(), chan.get());
        else
            ret = self->dev->get_rx_gain(name.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

static PyObject *_get_rx_gain_1(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    double ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_rx_gain(chan.get());
        else
            ret = self->dev->get_rx_gain();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_rx_gain(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs >= 1 && nargs <= 2
        && is<std::string>(PyTuple_GetItem(args, 0))
        && (nargs <= 1 || is<size_t>(PyTuple_GetItem(args, 1)))) {
        return _get_rx_gain_0(self, args);
    } else if (nargs >= 0 && nargs <= 1
        && (nargs <= 0 || is<size_t>(PyTuple_GetItem(args, 0)))) {
        return _get_rx_gain_1(self, args);
    }
    return _get_rx_gain_0(self, args);
}

PyObject *Uhd_get_tx_antenna(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    std::string ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_tx_antenna(chan.get());
        else
            ret = self->dev->get_tx_antenna();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_gpio_attr(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 2 || nargs > 3)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [2...3].", nargs);

    Expect<std::string> bank;
    if (!(bank = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", bank.what());
    Expect<std::string> attr;
    if (!(attr = to<std::string>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", attr.what());

    Expect<size_t> mboard;
    if (nargs > 2 && !(mboard = to<size_t>(PyTuple_GetItem(args, 2))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 3: %s", mboard.what());

    uint32_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 3)
            ret = self->dev->get_gpio_attr(bank.get(), attr.get(), mboard.get());
        else
            ret = self->dev->get_gpio_attr(bank.get(), attr.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_set_rx_rate(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<double> rate;
    if (!(rate = to<double>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", rate.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_rx_rate(rate.get(), chan.get());
        else
            self->dev->set_rx_rate(rate.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_get_num_mboards(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 0)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected None.", nargs);

    size_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        ret = self->dev->get_num_mboards();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_rx_antenna(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    std::string ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_rx_antenna(chan.get());
        else
            ret = self->dev->get_rx_antenna();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_filter_names(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<std::string> search_mask;
    if (nargs > 0 && !(search_mask = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", search_mask.what());

    std::vector<std::string> ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_filter_names(search_mask.get());
        else
            ret = self->dev->get_filter_names();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_fe_tx_freq_range(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    freq_range_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_fe_tx_freq_range(chan.get());
        else
            ret = self->dev->get_fe_tx_freq_range();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_set_tx_bandwidth(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<double> bandwidth;
    if (!(bandwidth = to<double>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", bandwidth.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_tx_bandwidth(bandwidth.get(), chan.get());
        else
            self->dev->set_tx_bandwidth(bandwidth.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_get_time_sources(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected 1.", nargs);

    Expect<size_t> mboard;
    if (!(mboard = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", mboard.what());

    std::vector<std::string> ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        ret = self->dev->get_time_sources(mboard.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_tx_freq(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    double ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_tx_freq(chan.get());
        else
            ret = self->dev->get_tx_freq();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_time_synchronized(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 0)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected None.", nargs);

    bool ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        ret = self->dev->get_time_synchronized();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_set_tx_freq(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<tune_request_t> tune_request;
    if (!(tune_request = to<tune_request_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", tune_request.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    tune_result_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            ret = self->dev->set_tx_freq(tune_request.get(), chan.get());
        else
            ret = self->dev->set_tx_freq(tune_request.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_set_time_source(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<std::string> source;
    if (!(source = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", source.what());

    Expect<size_t> mboard;
    if (nargs > 1 && !(mboard = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", mboard.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_time_source(source.get(), mboard.get());
        else
            self->dev->set_time_source(source.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_set_rx_freq(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<tune_request_t> tune_request;
    if (!(tune_request = to<tune_request_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", tune_request.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    tune_result_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            ret = self->dev->set_rx_freq(tune_request.get(), chan.get());
        else
            ret = self->dev->set_rx_freq(tune_request.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_tx_subdev_spec(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> mboard;
    if (nargs > 0 && !(mboard = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", mboard.what());

    uhd::usrp::subdev_spec_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_tx_subdev_spec(mboard.get());
        else
            ret = self->dev->get_tx_subdev_spec();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_read_register(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 2 || nargs > 3)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [2...3].", nargs);

    Expect<std::string> path;
    if (!(path = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", path.what());
    Expect<uint32_t> field;
    if (!(field = to<uint32_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", field.what());

    Expect<size_t> mboard;
    if (nargs > 2 && !(mboard = to<size_t>(PyTuple_GetItem(args, 2))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 3: %s", mboard.what());

    uint64_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 3)
            ret = self->dev->read_register(path.get(), field.get(), mboard.get());
        else
            ret = self->dev->read_register(path.get(), field.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_tx_num_channels(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 0)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected None.", nargs);

    size_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        ret = self->dev->get_tx_num_channels();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_set_tx_rate(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<double> rate;
    if (!(rate = to<double>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", rate.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_tx_rate(rate.get(), chan.get());
        else
            self->dev->set_tx_rate(rate.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_set_tx_antenna(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<std::string> ant;
    if (!(ant = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", ant.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_tx_antenna(ant.get(), chan.get());
        else
            self->dev->set_tx_antenna(ant.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_get_gpio_banks(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected 1.", nargs);

    Expect<size_t> mboard;
    if (!(mboard = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", mboard.what());

    std::vector<std::string> ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        ret = self->dev->get_gpio_banks(mboard.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_set_tx_iq_balance(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<std::complex<double>> correction;
    if (!(correction = to<std::complex<double>>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", correction.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_tx_iq_balance(correction.get(), chan.get());
        else
            self->dev->set_tx_iq_balance(correction.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_get_rx_gain_names(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    std::vector<std::string> ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_rx_gain_names(chan.get());
        else
            ret = self->dev->get_rx_gain_names();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_rx_lo_names(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    std::vector<std::string> ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_rx_lo_names(chan.get());
        else
            ret = self->dev->get_rx_lo_names();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

static PyObject *_set_rx_dc_offset_0(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<bool> enb;
    if (!(enb = to<bool>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", enb.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_rx_dc_offset(enb.get(), chan.get());
        else
            self->dev->set_rx_dc_offset(enb.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *_set_rx_dc_offset_1(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<std::complex<double>> offset;
    if (!(offset = to<std::complex<double>>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", offset.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_rx_dc_offset(offset.get(), chan.get());
        else
            self->dev->set_rx_dc_offset(offset.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_set_rx_dc_offset(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs >= 1 && nargs <= 2
        && is<bool>(PyTuple_GetItem(args, 0))
        && (nargs <= 1 || is<size_t>(PyTuple_GetItem(args, 1)))) {
        return _set_rx_dc_offset_0(self, args);
    } else if (nargs >= 1 && nargs <= 2
        && is<std::complex<double>>(PyTuple_GetItem(args, 0))
        && (nargs <= 1 || is<size_t>(PyTuple_GetItem(args, 1)))) {
        return _set_rx_dc_offset_1(self, args);
    }
    return _set_rx_dc_offset_0(self, args);
}

PyObject *Uhd_set_rx_bandwidth(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<double> bandwidth;
    if (!(bandwidth = to<double>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", bandwidth.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_rx_bandwidth(bandwidth.get(), chan.get());
        else
            self->dev->set_rx_bandwidth(bandwidth.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_clear_command_time(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> mboard;
    if (nargs > 0 && !(mboard = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", mboard.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            self->dev->clear_command_time(mboard.get());
        else
            self->dev->clear_command_time();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_set_rx_lo_freq(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 2 || nargs > 3)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [2...3].", nargs);

    Expect<double> freq;
    if (!(freq = to<double>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", freq.what());
    Expect<std::string> name;
    if (!(name = to<std::string>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", name.what());

    Expect<size_t> chan;
    if (nargs > 2 && !(chan = to<size_t>(PyTuple_GetItem(args, 2))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 3: %s", chan.what());

    double ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 3)
            ret = self->dev->set_rx_lo_freq(freq.get(), name.get(), chan.get());
        else
            ret = self->dev->set_rx_lo_freq(freq.get(), name.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_rx_rates(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    meta_range_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_rx_rates(chan.get());
        else
            ret = self->dev->get_rx_rates();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

static PyObject *_get_tx_gain_0(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<std::string> name;
    if (!(name = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", name.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    double ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            ret = self->dev->get_tx_gain(name.get(), chan.get());
        else
            ret = self->dev->get_tx_gain(name.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

static PyObject *_get_tx_gain_1(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    double ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_tx_gain(chan.get());
        else
            ret = self->dev->get_tx_gain();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_tx_gain(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs >= 1 && nargs <= 2
        && is<std::string>(PyTuple_GetItem(args, 0))
        && (nargs <= 1 || is<size_t>(PyTuple_GetItem(args, 1)))) {
        return _get_tx_gain_0(self, args);
    } else if (nargs >= 0 && nargs <= 1
        && (nargs <= 0 || is<size_t>(PyTuple_GetItem(args, 0)))) {
        return _get_tx_gain_1(self, args);
    }
    return _get_tx_gain_0(self, args);
}

PyObject *Uhd_set_tx_subdev_spec(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<uhd::usrp::subdev_spec_t> spec;
    if (!(spec = to<uhd::usrp::subdev_spec_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", spec.what());

    Expect<size_t> mboard;
    if (nargs > 1 && !(mboard = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", mboard.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_tx_subdev_spec(spec.get(), mboard.get());
        else
            self->dev->set_tx_subdev_spec(spec.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_get_tx_rates(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    meta_range_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_tx_rates(chan.get());
        else
            ret = self->dev->get_tx_rates();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_rx_lo_export_enabled(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...2].", nargs);

    Expect<std::string> name;
    Expect<size_t> chan;

    if (nargs > 0 && !(name = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", name.what());
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    bool ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            ret = self->dev->get_rx_lo_export_enabled(name.get(), chan.get());
        else if (nargs == 1)
            ret = self->dev->get_rx_lo_export_enabled(name.get());
        else
            ret = self->dev->get_rx_lo_export_enabled();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_normalized_rx_gain(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    double ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_normalized_rx_gain(chan.get());
        else
            ret = self->dev->get_normalized_rx_gain();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_rx_lo_sources(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...2].", nargs);

    Expect<std::string> name;
    Expect<size_t> chan;

    if (nargs > 0 && !(name = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", name.what());
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    std::vector<std::string> ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            ret = self->dev->get_rx_lo_sources(name.get(), chan.get());
        else if (nargs == 1)
            ret = self->dev->get_rx_lo_sources(name.get());
        else
            ret = self->dev->get_rx_lo_sources();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

static PyObject *_get_rx_gain_range_0(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<std::string> name;
    if (!(name = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", name.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    gain_range_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            ret = self->dev->get_rx_gain_range(name.get(), chan.get());
        else
            ret = self->dev->get_rx_gain_range(name.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

static PyObject *_get_rx_gain_range_1(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    gain_range_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_rx_gain_range(chan.get());
        else
            ret = self->dev->get_rx_gain_range();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_rx_gain_range(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs >= 1 && nargs <= 2
        && is<std::string>(PyTuple_GetItem(args, 0))
        && (nargs <= 1 || is<size_t>(PyTuple_GetItem(args, 1)))) {
        return _get_rx_gain_range_0(self, args);
    } else if (nargs >= 0 && nargs <= 1
        && (nargs <= 0 || is<size_t>(PyTuple_GetItem(args, 0)))) {
        return _get_rx_gain_range_1(self, args);
    }
    return _get_rx_gain_range_0(self, args);
}

static PyObject *_set_rx_iq_balance_0(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 2 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected 2.", nargs);

    Expect<bool> enb;
    if (!(enb = to<bool>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", enb.what());
    Expect<size_t> chan;
    if (!(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        self->dev->set_rx_iq_balance(enb.get(), chan.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *_set_rx_iq_balance_1(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<std::complex<double>> correction;
    if (!(correction = to<std::complex<double>>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", correction.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_rx_iq_balance(correction.get(), chan.get());
        else
            self->dev->set_rx_iq_balance(correction.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_set_rx_iq_balance(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs >= 2 && nargs <= 2
        && is<bool>(PyTuple_GetItem(args, 0))
        && is<size_t>(PyTuple_GetItem(args, 1))) {
        return _set_rx_iq_balance_0(self, args);
    } else if (nargs >= 1 && nargs <= 2
        && is<std::complex<double>>(PyTuple_GetItem(args, 0))
        && (nargs <= 1 || is<size_t>(PyTuple_GetItem(args, 1)))) {
        return _set_rx_iq_balance_1(self, args);
    }
    return _set_rx_iq_balance_0(self, args);
}

PyObject *Uhd_get_rx_bandwidth(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    double ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_rx_bandwidth(chan.get());
        else
            ret = self->dev->get_rx_bandwidth();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_write_register(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 3 || nargs > 4)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [3...4].", nargs);

    Expect<std::string> path;
    if (!(path = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", path.what());
    Expect<uint32_t> field;
    if (!(field = to<uint32_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", field.what());
    Expect<uint64_t> value;
    if (!(value = to<uint64_t>(PyTuple_GetItem(args, 2))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 3: %s", value.what());

    Expect<size_t> mboard;
    if (nargs > 3 && !(mboard = to<size_t>(PyTuple_GetItem(args, 3))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 4: %s", mboard.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 4)
            self->dev->write_register(path.get(), field.get(), value.get(), mboard.get());
        else
            self->dev->write_register(path.get(), field.get(), value.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_set_rx_lo_source(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 3)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...3].", nargs);

    Expect<std::string> src;
    if (!(src = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", src.what());

    Expect<std::string> name;
    Expect<size_t> chan;

    if (nargs > 1 && !(name = to<std::string>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", name.what());
    if (nargs > 2 && !(chan = to<size_t>(PyTuple_GetItem(args, 2))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 3: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 3)
            self->dev->set_rx_lo_source(src.get(), name.get(), chan.get());
        else if (nargs == 2)
            self->dev->set_rx_lo_source(src.get(), name.get());
        else
            self->dev->set_rx_lo_source(src.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_get_time_source(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected 1.", nargs);

    Expect<size_t> mboard;
    if (!(mboard = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", mboard.what());

    std::string ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        ret = self->dev->get_time_source(mboard.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_rx_lo_freq_range(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<std::string> name;
    if (!(name = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", name.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    freq_range_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            ret = self->dev->get_rx_lo_freq_range(name.get(), chan.get());
        else
            ret = self->dev->get_rx_lo_freq_range(name.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_rx_lo_freq(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<std::string> name;
    if (!(name = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", name.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    double ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            ret = self->dev->get_rx_lo_freq(name.get(), chan.get());
        else
            ret = self->dev->get_rx_lo_freq(name.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_mboard_name(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> mboard;
    if (nargs > 0 && !(mboard = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", mboard.what());

    std::string ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_mboard_name(mboard.get());
        else
            ret = self->dev->get_mboard_name();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_tx_bandwidth_range(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    meta_range_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_tx_bandwidth_range(chan.get());
        else
            ret = self->dev->get_tx_bandwidth_range();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_rx_subdev_name(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    std::string ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_rx_subdev_name(chan.get());
        else
            ret = self->dev->get_rx_subdev_name();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_set_clock_source(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<std::string> source;
    if (!(source = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", source.what());

    Expect<size_t> mboard;
    if (nargs > 1 && !(mboard = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", mboard.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_clock_source(source.get(), mboard.get());
        else
            self->dev->set_clock_source(source.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_get_master_clock_rate(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> mboard;
    if (nargs > 0 && !(mboard = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", mboard.what());

    double ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_master_clock_rate(mboard.get());
        else
            ret = self->dev->get_master_clock_rate();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_pp_string(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 0)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected None.", nargs);

    std::string ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        ret = self->dev->get_pp_string();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_tx_subdev_name(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    std::string ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_tx_subdev_name(chan.get());
        else
            ret = self->dev->get_tx_subdev_name();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_enumerate_registers(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> mboard;
    if (nargs > 0 && !(mboard = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", mboard.what());

    std::vector<std::string> ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->enumerate_registers(mboard.get());
        else
            ret = self->dev->enumerate_registers();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_rx_sensor_names(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    std::vector<std::string> ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_rx_sensor_names(chan.get());
        else
            ret = self->dev->get_rx_sensor_names();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_tx_freq_range(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    freq_range_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_tx_freq_range(chan.get());
        else
            ret = self->dev->get_tx_freq_range();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_rx_freq_range(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    freq_range_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_rx_freq_range(chan.get());
        else
            ret = self->dev->get_rx_freq_range();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

static PyObject *_set_rx_gain_0(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 2 || nargs > 3)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [2...3].", nargs);

    Expect<double> gain;
    if (!(gain = to<double>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", gain.what());
    Expect<std::string> name;
    if (!(name = to<std::string>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", name.what());

    Expect<size_t> chan;
    if (nargs > 2 && !(chan = to<size_t>(PyTuple_GetItem(args, 2))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 3: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 3)
            self->dev->set_rx_gain(gain.get(), name.get(), chan.get());
        else
            self->dev->set_rx_gain(gain.get(), name.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *_set_rx_gain_1(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<double> gain;
    if (!(gain = to<double>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", gain.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_rx_gain(gain.get(), chan.get());
        else
            self->dev->set_rx_gain(gain.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_set_rx_gain(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs >= 2 && nargs <= 3
        && is<double>(PyTuple_GetItem(args, 0))
        && is<std::string>(PyTuple_GetItem(args, 1))
        && (nargs <= 2 || is<size_t>(PyTuple_GetItem(args, 2)))) {
        return _set_rx_gain_0(self, args);
    } else if (nargs >= 1 && nargs <= 2
        && is<double>(PyTuple_GetItem(args, 0))
        && (nargs <= 1 || is<size_t>(PyTuple_GetItem(args, 1)))) {
        return _set_rx_gain_1(self, args);
    }
    return _set_rx_gain_0(self, args);
}

PyObject *Uhd_get_fe_rx_freq_range(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    freq_range_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_fe_rx_freq_range(chan.get());
        else
            ret = self->dev->get_fe_rx_freq_range();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_clock_source(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected 1.", nargs);

    Expect<size_t> mboard;
    if (!(mboard = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", mboard.what());

    std::string ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        ret = self->dev->get_clock_source(mboard.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_rx_lo_source(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...2].", nargs);

    Expect<std::string> name;
    Expect<size_t> chan;

    if (nargs > 0 && !(name = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", name.what());
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    std::string ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            ret = self->dev->get_rx_lo_source(name.get(), chan.get());
        else if (nargs == 1)
            ret = self->dev->get_rx_lo_source(name.get());
        else
            ret = self->dev->get_rx_lo_source();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_tx_gain_names(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    std::vector<std::string> ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_tx_gain_names(chan.get());
        else
            ret = self->dev->get_tx_gain_names();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_tx_antennas(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    std::vector<std::string> ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_tx_antennas(chan.get());
        else
            ret = self->dev->get_tx_antennas();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_set_normalized_rx_gain(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<double> gain;
    if (!(gain = to<double>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", gain.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_normalized_rx_gain(gain.get(), chan.get());
        else
            self->dev->set_normalized_rx_gain(gain.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_get_rx_num_channels(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 0)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected None.", nargs);

    size_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        ret = self->dev->get_rx_num_channels();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_rx_subdev_spec(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> mboard;
    if (nargs > 0 && !(mboard = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", mboard.what());

    uhd::usrp::subdev_spec_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_rx_subdev_spec(mboard.get());
        else
            ret = self->dev->get_rx_subdev_spec();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_rx_antennas(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    std::vector<std::string> ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_rx_antennas(chan.get());
        else
            ret = self->dev->get_rx_antennas();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_set_tx_dc_offset(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<std::complex<double>> offset;
    if (!(offset = to<std::complex<double>>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", offset.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_tx_dc_offset(offset.get(), chan.get());
        else
            self->dev->set_tx_dc_offset(offset.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_set_clock_source_out(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<bool> enb;
    if (!(enb = to<bool>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", enb.what());

    Expect<size_t> mboard;
    if (nargs > 1 && !(mboard = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", mboard.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_clock_source_out(enb.get(), mboard.get());
        else
            self->dev->set_clock_source_out(enb.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_set_master_clock_rate(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<double> rate;
    if (!(rate = to<double>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", rate.what());

    Expect<size_t> mboard;
    if (nargs > 1 && !(mboard = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", mboard.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_master_clock_rate(rate.get(), mboard.get());
        else
            self->dev->set_master_clock_rate(rate.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_get_rx_bandwidth_range(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    meta_range_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_rx_bandwidth_range(chan.get());
        else
            ret = self->dev->get_rx_bandwidth_range();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_set_rx_antenna(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<std::string> ant;
    if (!(ant = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", ant.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_rx_antenna(ant.get(), chan.get());
        else
            self->dev->set_rx_antenna(ant.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_set_user_register(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 2 || nargs > 3)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [2...3].", nargs);

    Expect<uint8_t> addr;
    if (!(addr = to<uint8_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", addr.what());
    Expect<uint32_t> data;
    if (!(data = to<uint32_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", data.what());

    Expect<size_t> mboard;
    if (nargs > 2 && !(mboard = to<size_t>(PyTuple_GetItem(args, 2))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 3: %s", mboard.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 3)
            self->dev->set_user_register(addr.get(), data.get(), mboard.get());
        else
            self->dev->set_user_register(addr.get(), data.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_get_rx_freq(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    double ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_rx_freq(chan.get());
        else
            ret = self->dev->get_rx_freq();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_tx_sensor_names(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    std::vector<std::string> ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_tx_sensor_names(chan.get());
        else
            ret = self->dev->get_tx_sensor_names();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_rx_rate(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    double ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_rx_rate(chan.get());
        else
            ret = self->dev->get_rx_rate();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_set_rx_subdev_spec(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<uhd::usrp::subdev_spec_t> spec;
    if (!(spec = to<uhd::usrp::subdev_spec_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", spec.what());

    Expect<size_t> mboard;
    if (nargs > 1 && !(mboard = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", mboard.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_rx_subdev_spec(spec.get(), mboard.get());
        else
            self->dev->set_rx_subdev_spec(spec.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_get_tx_rate(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    double ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_tx_rate(chan.get());
        else
            ret = self->dev->get_tx_rate();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_tx_bandwidth(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    double ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_tx_bandwidth(chan.get());
        else
            ret = self->dev->get_tx_bandwidth();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_set_normalized_tx_gain(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<double> gain;
    if (!(gain = to<double>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", gain.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_normalized_tx_gain(gain.get(), chan.get());
        else
            self->dev->set_normalized_tx_gain(gain.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *_get_tx_gain_range_0(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<std::string> name;
    if (!(name = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", name.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    gain_range_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            ret = self->dev->get_tx_gain_range(name.get(), chan.get());
        else
            ret = self->dev->get_tx_gain_range(name.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

static PyObject *_get_tx_gain_range_1(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    gain_range_t ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_tx_gain_range(chan.get());
        else
            ret = self->dev->get_tx_gain_range();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_tx_gain_range(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs >= 1 && nargs <= 2
        && is<std::string>(PyTuple_GetItem(args, 0))
        && (nargs <= 1 || is<size_t>(PyTuple_GetItem(args, 1)))) {
        return _get_tx_gain_range_0(self, args);
    } else if (nargs >= 0 && nargs <= 1
        && (nargs <= 0 || is<size_t>(PyTuple_GetItem(args, 0)))) {
        return _get_tx_gain_range_1(self, args);
    }
    return _get_tx_gain_range_0(self, args);
}

PyObject *Uhd_set_rx_agc(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<bool> enable;
    if (!(enable = to<bool>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", enable.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_rx_agc(enable.get(), chan.get());
        else
            self->dev->set_rx_agc(enable.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_get_mboard_sensor_names(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> mboard;
    if (nargs > 0 && !(mboard = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", mboard.what());

    std::vector<std::string> ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_mboard_sensor_names(mboard.get());
        else
            ret = self->dev->get_mboard_sensor_names();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

static PyObject *_set_tx_gain_0(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 2 || nargs > 3)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [2...3].", nargs);

    Expect<double> gain;
    if (!(gain = to<double>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", gain.what());
    Expect<std::string> name;
    if (!(name = to<std::string>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", name.what());

    Expect<size_t> chan;
    if (nargs > 2 && !(chan = to<size_t>(PyTuple_GetItem(args, 2))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 3: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 3)
            self->dev->set_tx_gain(gain.get(), name.get(), chan.get());
        else
            self->dev->set_tx_gain(gain.get(), name.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *_set_tx_gain_1(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<double> gain;
    if (!(gain = to<double>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", gain.what());

    Expect<size_t> chan;
    if (nargs > 1 && !(chan = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", chan.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_tx_gain(gain.get(), chan.get());
        else
            self->dev->set_tx_gain(gain.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_set_tx_gain(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs >= 2 && nargs <= 3
        && is<double>(PyTuple_GetItem(args, 0))
        && is<std::string>(PyTuple_GetItem(args, 1))
        && (nargs <= 2 || is<size_t>(PyTuple_GetItem(args, 2)))) {
        return _set_tx_gain_0(self, args);
    } else if (nargs >= 1 && nargs <= 2
        && is<double>(PyTuple_GetItem(args, 0))
        && (nargs <= 1 || is<size_t>(PyTuple_GetItem(args, 1)))) {
        return _set_tx_gain_1(self, args);
    }
    return _set_tx_gain_0(self, args);
}

PyObject *Uhd_get_clock_sources(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected 1.", nargs);

    Expect<size_t> mboard;
    if (!(mboard = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", mboard.what());

    std::vector<std::string> ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        ret = self->dev->get_clock_sources(mboard.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_get_usrp_rx_info(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    dict<std::string, std::string> ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_usrp_rx_info(chan.get());
        else
            ret = self->dev->get_usrp_rx_info();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

PyObject *Uhd_set_time_source_out(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 1 || nargs > 2)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [1...2].", nargs);

    Expect<bool> enb;
    if (!(enb = to<bool>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", enb.what());

    Expect<size_t> mboard;
    if (nargs > 1 && !(mboard = to<size_t>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", mboard.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 2)
            self->dev->set_time_source_out(enb.get(), mboard.get());
        else
            self->dev->set_time_source_out(enb.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_set_gpio_attr(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 3 || nargs > 5)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [3...5].", nargs);

    Expect<std::string> bank;
    if (!(bank = to<std::string>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", bank.what());
    Expect<std::string> attr;
    if (!(attr = to<std::string>(PyTuple_GetItem(args, 1))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 2: %s", attr.what());
    Expect<uint32_t> value;
    if (!(value = to<uint32_t>(PyTuple_GetItem(args, 2))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 3: %s", value.what());

    Expect<uint32_t> mask;
    Expect<size_t> mboard;

    if (nargs > 3 && !(mask = to<uint32_t>(PyTuple_GetItem(args, 3))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 4: %s", mask.what());
    if (nargs > 4 && !(mboard = to<size_t>(PyTuple_GetItem(args, 4))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 5: %s", mboard.what());

    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 5)
            self->dev->set_gpio_attr(bank.get(), attr.get(), value.get(), mask.get(), mboard.get());
        else if (nargs == 4)
            self->dev->set_gpio_attr(bank.get(), attr.get(), value.get(), mask.get());
        else
            self->dev->set_gpio_attr(bank.get(), attr.get(), value.get());
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *Uhd_get_normalized_tx_gain(Uhd *self, PyObject *args) {

    const Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < 0 || nargs > 1)
        return PyErr_Format(PyExc_TypeError, "Invalid number of arguments: got %ld, expected [0...1].", nargs);

    Expect<size_t> chan;
    if (nargs > 0 && !(chan = to<size_t>(PyTuple_GetItem(args, 0))))
        return PyErr_Format(PyExc_TypeError, "Invalid type for argument # 1: %s", chan.what());

    double ret;
    try {
        std::lock_guard<std::mutex> lg(self->dev_lock);
        if (nargs == 1)
            ret = self->dev->get_normalized_tx_gain(chan.get());
        else
            ret = self->dev->get_normalized_tx_gain();
    } catch(const uhd::exception &e) {
        return PyErr_Format(UhdError, "%s", e.what());
    }

    return from(ret);
}

const std::vector<PyMethodDef> Uhd_gen_methods = {{
    {"set_rx_lo_export_enabled", (PyCFunction)Uhd_set_rx_lo_export_enabled, METH_VARARGS, ""},
    {"get_usrp_tx_info", (PyCFunction)Uhd_get_usrp_tx_info, METH_VARARGS, ""},
    {"get_rx_gain", (PyCFunction)Uhd_get_rx_gain, METH_VARARGS, ""},
    {"get_tx_antenna", (PyCFunction)Uhd_get_tx_antenna, METH_VARARGS, ""},
    {"get_gpio_attr", (PyCFunction)Uhd_get_gpio_attr, METH_VARARGS, ""},
    {"set_rx_rate", (PyCFunction)Uhd_set_rx_rate, METH_VARARGS, ""},
    {"get_num_mboards", (PyCFunction)Uhd_get_num_mboards, METH_VARARGS, ""},
    {"get_rx_antenna", (PyCFunction)Uhd_get_rx_antenna, METH_VARARGS, ""},
    {"get_filter_names", (PyCFunction)Uhd_get_filter_names, METH_VARARGS, ""},
    {"get_fe_tx_freq_range", (PyCFunction)Uhd_get_fe_tx_freq_range, METH_VARARGS, ""},
    {"set_tx_bandwidth", (PyCFunction)Uhd_set_tx_bandwidth, METH_VARARGS, ""},
    {"get_time_sources", (PyCFunction)Uhd_get_time_sources, METH_VARARGS, ""},
    {"get_tx_freq", (PyCFunction)Uhd_get_tx_freq, METH_VARARGS, ""},
    {"get_time_synchronized", (PyCFunction)Uhd_get_time_synchronized, METH_VARARGS, ""},
    {"set_tx_freq", (PyCFunction)Uhd_set_tx_freq, METH_VARARGS, ""},
    {"set_time_source", (PyCFunction)Uhd_set_time_source, METH_VARARGS, ""},
    {"set_rx_freq", (PyCFunction)Uhd_set_rx_freq, METH_VARARGS, ""},
    {"get_tx_subdev_spec", (PyCFunction)Uhd_get_tx_subdev_spec, METH_VARARGS, ""},
    {"read_register", (PyCFunction)Uhd_read_register, METH_VARARGS, ""},
    {"get_tx_num_channels", (PyCFunction)Uhd_get_tx_num_channels, METH_VARARGS, ""},
    {"set_tx_rate", (PyCFunction)Uhd_set_tx_rate, METH_VARARGS, ""},
    {"set_tx_antenna", (PyCFunction)Uhd_set_tx_antenna, METH_VARARGS, ""},
    {"get_gpio_banks", (PyCFunction)Uhd_get_gpio_banks, METH_VARARGS, ""},
    {"set_tx_iq_balance", (PyCFunction)Uhd_set_tx_iq_balance, METH_VARARGS, ""},
    {"get_rx_gain_names", (PyCFunction)Uhd_get_rx_gain_names, METH_VARARGS, ""},
    {"get_rx_lo_names", (PyCFunction)Uhd_get_rx_lo_names, METH_VARARGS, ""},
    {"set_rx_dc_offset", (PyCFunction)Uhd_set_rx_dc_offset, METH_VARARGS, ""},
    {"set_rx_bandwidth", (PyCFunction)Uhd_set_rx_bandwidth, METH_VARARGS, ""},
    {"clear_command_time", (PyCFunction)Uhd_clear_command_time, METH_VARARGS, ""},
    {"set_rx_lo_freq", (PyCFunction)Uhd_set_rx_lo_freq, METH_VARARGS, ""},
    {"get_rx_rates", (PyCFunction)Uhd_get_rx_rates, METH_VARARGS, ""},
    {"get_tx_gain", (PyCFunction)Uhd_get_tx_gain, METH_VARARGS, ""},
    {"set_tx_subdev_spec", (PyCFunction)Uhd_set_tx_subdev_spec, METH_VARARGS, ""},
    {"get_tx_rates", (PyCFunction)Uhd_get_tx_rates, METH_VARARGS, ""},
    {"get_rx_lo_export_enabled", (PyCFunction)Uhd_get_rx_lo_export_enabled, METH_VARARGS, ""},
    {"get_normalized_rx_gain", (PyCFunction)Uhd_get_normalized_rx_gain, METH_VARARGS, ""},
    {"get_rx_lo_sources", (PyCFunction)Uhd_get_rx_lo_sources, METH_VARARGS, ""},
    {"get_rx_gain_range", (PyCFunction)Uhd_get_rx_gain_range, METH_VARARGS, ""},
    {"set_rx_iq_balance", (PyCFunction)Uhd_set_rx_iq_balance, METH_VARARGS, ""},
    {"get_rx_bandwidth", (PyCFunction)Uhd_get_rx_bandwidth, METH_VARARGS, ""},
    {"write_register", (PyCFunction)Uhd_write_register, METH_VARARGS, ""},
    {"set_rx_lo_source", (PyCFunction)Uhd_set_rx_lo_source, METH_VARARGS, ""},
    {"get_time_source", (PyCFunction)Uhd_get_time_source, METH_VARARGS, ""},
    {"get_rx_lo_freq_range", (PyCFunction)Uhd_get_rx_lo_freq_range, METH_VARARGS, ""},
    {"get_rx_lo_freq", (PyCFunction)Uhd_get_rx_lo_freq, METH_VARARGS, ""},
    {"get_mboard_name", (PyCFunction)Uhd_get_mboard_name, METH_VARARGS, ""},
    {"get_tx_bandwidth_range", (PyCFunction)Uhd_get_tx_bandwidth_range, METH_VARARGS, ""},
    {"get_rx_subdev_name", (PyCFunction)Uhd_get_rx_subdev_name, METH_VARARGS, ""},
    {"set_clock_source", (PyCFunction)Uhd_set_clock_source, METH_VARARGS, ""},
    {"get_master_clock_rate", (PyCFunction)Uhd_get_master_clock_rate, METH_VARARGS, ""},
    {"get_pp_string", (PyCFunction)Uhd_get_pp_string, METH_VARARGS, ""},
    {"get_tx_subdev_name", (PyCFunction)Uhd_get_tx_subdev_name, METH_VARARGS, ""},
    {"enumerate_registers", (PyCFunction)Uhd_enumerate_registers, METH_VARARGS, ""},
    {"get_rx_sensor_names", (PyCFunction)Uhd_get_rx_sensor_names, METH_VARARGS, ""},
    {"get_tx_freq_range", (PyCFunction)Uhd_get_tx_freq_range, METH_VARARGS, ""},
    {"get_rx_freq_range", (PyCFunction)Uhd_get_rx_freq_range, METH_VARARGS, ""},
    {"set_rx_gain", (PyCFunction)Uhd_set_rx_gain, METH_VARARGS, ""},
    {"get_fe_rx_freq_range", (PyCFunction)Uhd_get_fe_rx_freq_range, METH_VARARGS, ""},
    {"get_clock_source", (PyCFunction)Uhd_get_clock_source, METH_VARARGS, ""},
    {"get_rx_lo_source", (PyCFunction)Uhd_get_rx_lo_source, METH_VARARGS, ""},
    {"get_tx_gain_names", (PyCFunction)Uhd_get_tx_gain_names, METH_VARARGS, ""},
    {"get_tx_antennas", (PyCFunction)Uhd_get_tx_antennas, METH_VARARGS, ""},
    {"set_normalized_rx_gain", (PyCFunction)Uhd_set_normalized_rx_gain, METH_VARARGS, ""},
    {"get_rx_num_channels", (PyCFunction)Uhd_get_rx_num_channels, METH_VARARGS, ""},
    {"get_rx_subdev_spec", (PyCFunction)Uhd_get_rx_subdev_spec, METH_VARARGS, ""},
    {"get_rx_antennas", (PyCFunction)Uhd_get_rx_antennas, METH_VARARGS, ""},
    {"set_tx_dc_offset", (PyCFunction)Uhd_set_tx_dc_offset, METH_VARARGS, ""},
    {"set_clock_source_out", (PyCFunction)Uhd_set_clock_source_out, METH_VARARGS, ""},
    {"set_master_clock_rate", (PyCFunction)Uhd_set_master_clock_rate, METH_VARARGS, ""},
    {"get_rx_bandwidth_range", (PyCFunction)Uhd_get_rx_bandwidth_range, METH_VARARGS, ""},
    {"set_rx_antenna", (PyCFunction)Uhd_set_rx_antenna, METH_VARARGS, ""},
    {"set_user_register", (PyCFunction)Uhd_set_user_register, METH_VARARGS, ""},
    {"get_rx_freq", (PyCFunction)Uhd_get_rx_freq, METH_VARARGS, ""},
    {"get_tx_sensor_names", (PyCFunction)Uhd_get_tx_sensor_names, METH_VARARGS, ""},
    {"get_rx_rate", (PyCFunction)Uhd_get_rx_rate, METH_VARARGS, ""},
    {"set_rx_subdev_spec", (PyCFunction)Uhd_set_rx_subdev_spec, METH_VARARGS, ""},
    {"get_tx_rate", (PyCFunction)Uhd_get_tx_rate, METH_VARARGS, ""},
    {"get_tx_bandwidth", (PyCFunction)Uhd_get_tx_bandwidth, METH_VARARGS, ""},
    {"set_normalized_tx_gain", (PyCFunction)Uhd_set_normalized_tx_gain, METH_VARARGS, ""},
    {"get_tx_gain_range", (PyCFunction)Uhd_get_tx_gain_range, METH_VARARGS, ""},
    {"set_rx_agc", (PyCFunction)Uhd_set_rx_agc, METH_VARARGS, ""},
    {"get_mboard_sensor_names", (PyCFunction)Uhd_get_mboard_sensor_names, METH_VARARGS, ""},
    {"set_tx_gain", (PyCFunction)Uhd_set_tx_gain, METH_VARARGS, ""},
    {"get_clock_sources", (PyCFunction)Uhd_get_clock_sources, METH_VARARGS, ""},
    {"get_usrp_rx_info", (PyCFunction)Uhd_get_usrp_rx_info, METH_VARARGS, ""},
    {"set_time_source_out", (PyCFunction)Uhd_set_time_source_out, METH_VARARGS, ""},
    {"set_gpio_attr", (PyCFunction)Uhd_set_gpio_attr, METH_VARARGS, ""},
    {"get_normalized_tx_gain", (PyCFunction)Uhd_get_normalized_tx_gain, METH_VARARGS, ""},
}};

}
