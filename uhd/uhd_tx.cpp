#include <Python.h>

#include <mutex>
#include <vector>
#include <memory>
#include <thread>

#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
#include <uhd/utils/thread_priority.hpp>

#include "uhd_types.hpp"
#include "uhd_tx.hpp"

namespace uhd {

TransmitWorker::TransmitWorker(uhd::usrp::multi_usrp::sptr dev, std::mutex &dev_lock) : _dev(dev), _dev_lock(dev_lock), _transmitting(false) {}

TransmitWorker::~TransmitWorker() {
    std::future<std::string> accepted = make_request(TransmitRequestType::Exit);
    accepted.wait();
    _thread.join();
}

std::future<std::string> TransmitWorker::make_request(const TransmitRequestType type) {
    TransmitRequest *request = new TransmitRequest(type);
    std::future<std::string> accepted = request->accepted.get_future();
    /** Push into requests queue **/
    {
        std::lock_guard<std::mutex> lg(_requests_lock);
        _requests_queue.push(request);
        _requests_notify.notify_one();
    }
    return accepted;
}

std::future<std::string> TransmitWorker::make_request(const TransmitRequestType type, const size_t num_samps,
                                                      std::vector<std::complex<float> *> &&samps, std::vector<long unsigned int> &&channels,
                                                      const double seconds_in_future, const double timeout, const std::string &otw_format) {
    TransmitRequest *request = new TransmitRequest(type, num_samps, std::move(samps), std::move(channels),
                                                   seconds_in_future, timeout, otw_format);
    std::future<std::string> accepted = request->accepted.get_future();
    /** Push into requests queue **/
    {
        std::lock_guard<std::mutex> lg(_requests_lock);
        _requests_queue.push(request);
        _requests_notify.notify_one();
    }
    return accepted;
}

void TransmitWorker::init() {
    _thread = std::thread(&TransmitWorker::_worker, this);
}

void TransmitWorker::_worker() {

    uhd::set_thread_priority_safe(1.0, true);
    _transmitting = false;

    while (true) {

        /** Get new request **/
        TransmitRequest *req;
        {
            std::unique_lock<std::mutex> lg(_requests_lock);
            while (_requests_queue.empty())
                _requests_notify.wait(lg);
            req = _requests_queue.front();
            _requests_queue.pop();
        }

        const TransmitRequestType req_type = req->type;

        if (req_type == TransmitRequestType::Stop) {
            req->accepted.set_value("");
            delete req;
            continue;
        } else if (req_type == TransmitRequestType::Exit) {
            req->accepted.set_value("");
            delete req;
            break;
        }

        const size_t num_samps = req->num_samps;
        const std::vector<std::complex<float> *> samps(std::move(req->samps));
        const double seconds_in_future = req->seconds_in_future;
        const double timeout = req->timeout;
        /** Extend the first timeout by 'seconds_in_future' **/
        double next_timeout = timeout + seconds_in_future;
        const bool streaming = req_type == TransmitRequestType::Continuous;
        uhd::stream_args_t stream_args("fc32", req->otw_format);
        stream_args.channels = std::move(req->channels);

        uhd::tx_streamer::sptr tx_stream;
        uhd::tx_metadata_t md;
        std::string error;

        try {
            /** Lock device **/
            std::lock_guard<std::mutex> lg(_dev_lock);
            tx_stream = _dev->get_tx_stream(stream_args);
            md.start_of_burst = true;
            md.end_of_burst = false;
            md.has_time_spec = true;
            md.time_spec = _dev->get_time_now() + uhd::time_spec_t(seconds_in_future);
        } catch (const uhd::exception &e) {
            error = "UHD exception occurred: " + std::string(e.what());
        } catch (...) {
            error = "Unkown exception occurred.";
        }

        if (!error.empty()) {
            /** An error occurred **/
            req->accepted.set_value(std::move(error));
            delete req;
            /** Free buffers **/
            for (auto &s : samps)
                free(s);
            continue;
        }

        /** All done with request. **/
        req->accepted.set_value("");
        delete req;

        do {
            tx_stream->send(samps, num_samps, md, next_timeout);
            next_timeout = timeout;
            md.start_of_burst = false;
            md.has_time_spec = false;
            /** Check for new requests **/
            {
                std::lock_guard<std::mutex> lg(_requests_lock);
                if (!_requests_queue.empty())
                    break;
            }
        } while(streaming);

        /** Shutdown transmitter **/
        md.end_of_burst = true;
        tx_stream->send("", 0, md);
        /** Free buffers **/
        for (auto &s : samps)
            free(s);
    }
}

}
