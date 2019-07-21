#include <Python.h>

#include <mutex>
#include <vector>
#include <memory>
#include <thread>

#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
#include <uhd/utils/thread_priority.hpp>

#include "uhd_types.hpp"
#include "uhd_rx.hpp"

namespace uhd {

ReceiveWorker::ReceiveWorker(uhd::usrp::multi_usrp::sptr dev, std::mutex &dev_lock) : _dev(dev), _dev_lock(dev_lock), _receiving(false) {}

ReceiveWorker::~ReceiveWorker() {
    std::future<void> accepted = make_request(ReceiveRequestType::Exit);
    accepted.wait();
    _thread.join();

    /** Drain results queue **/
    {
        std::lock_guard<std::mutex> lg(_results_lock);
        while (!_results_queue.empty()) {
            ReceiveResult *result = _results_queue.front();
            _results_queue.pop();
            for (auto &ptr : result->bufs)
                free(ptr);
            delete result;
        }
    }
}

std::future<void> ReceiveWorker::make_request(const ReceiveRequestType type) {
    ReceiveRequest *request = new ReceiveRequest(type);
    std::future<void> accepted = request->accepted.get_future();
    /** Push into requests queue **/
    {
        std::lock_guard<std::mutex> lg(_requests_lock);
        _requests_queue.push(request);
        _requests_notify.notify_one();
    }
    return accepted;
}

std::future<void> ReceiveWorker::make_request(const ReceiveRequestType type, const size_t num_samps,
                                              std::vector<long unsigned int> &&channels,
                                              const double seconds_in_future, const double timeout,
                                              const std::string &otw_format) {
    ReceiveRequest *request = new ReceiveRequest(type, num_samps, std::move(channels),
                                                 seconds_in_future, timeout, otw_format);
    std::future<void> accepted = request->accepted.get_future();
    /** Push into requests queue **/
    {
        std::lock_guard<std::mutex> lg(_requests_lock);
        _requests_queue.push(request);
        _requests_notify.notify_one();
    }
    return accepted;
}

ReceiveResult *ReceiveWorker::get_result(const bool fresh) {

    time_spec_t now;
    if (fresh) {
        /** Lock device **/
        std::lock_guard<std::mutex> lg(_dev_lock);
        /** Get the time now **/
        now = _dev->get_time_now();
    }

    std::unique_lock<std::mutex> lg(_results_lock);
    while (true) {
        if (!_results_queue.empty()) {
            ReceiveResult *result = _results_queue.front();
            /** If we don't care if fresh, or if timestamp of result is greater than now.
                Since, when recycling, there is never more than 1 result in queue, if the
                timestamp of the result at the front of the queue is not greater than now,
                then wait until this result is re-claimed, and a new result is pushed. **/
            if (!fresh || result->error || result->start > now) {
                _results_queue.pop();
                return result;
            }
        }
        if (!_receiving)
            return new ReceiveResult("Not streaming, or no data available.");
        _results_notify.wait(lg);
    }
}

void ReceiveWorker::init() {
    _thread = std::thread(&ReceiveWorker::_worker, this);
}

void ReceiveWorker::_worker() {

    uhd::set_thread_priority_safe(1.0, true);

    _receiving = false;

    while (true) {

        /** Get new request **/
        ReceiveRequest *req;
        {
            std::unique_lock<std::mutex> lg(_requests_lock);
            while (_requests_queue.empty())
                _requests_notify.wait(lg);
            req = _requests_queue.front();
            _requests_queue.pop();
        }

        /** Drain results queue **/
        {
            std::lock_guard<std::mutex> lg(_results_lock);
            while (!_results_queue.empty()) {
                ReceiveResult *result = _results_queue.front();
                _results_queue.pop();
                for (auto &ptr : result->bufs)
                    free(ptr);
                delete result;
            }
        }

        const ReceiveRequestType req_type = req->type;

        if (req_type == ReceiveRequestType::Single ||
            req_type == ReceiveRequestType::Continuous ||
            req_type == ReceiveRequestType::Recycle) {
            _receiving = true;
            req->accepted.set_value();
        } else if (req_type == ReceiveRequestType::Stop) {
            req->accepted.set_value();
            delete req;
            continue;
        } else if (req_type == ReceiveRequestType::Exit) {
            req->accepted.set_value();
            delete req;
            break;
        }

        const size_t num_channels = req->channels.size();
        const size_t num_samps = req->num_samps;
        const double seconds_in_future = req->seconds_in_future;
        const double timeout = req->timeout;
        /** Extend the first timeout by 'seconds_in_future' **/
        double next_timeout = timeout + seconds_in_future;
        bool streaming = req_type == ReceiveRequestType::Continuous || req_type == ReceiveRequestType::Recycle;
        uhd::stream_args_t stream_args("fc32", req->otw_format);
        stream_args.channels = std::move(req->channels);
        /** All done with request. **/
        delete req;
        uhd::stream_cmd_t stream_cmd(streaming ? uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS
                                     : uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE);
        uhd::rx_streamer::sptr rx_stream;
        std::string error;

        try {
            /** Lock device **/
            std::lock_guard<std::mutex> lg(_dev_lock);
            rx_stream = _dev->get_rx_stream(stream_args);
            stream_cmd.num_samps = num_samps;
            stream_cmd.stream_now = (seconds_in_future < 0.0001) ? true : false;
            stream_cmd.time_spec = _dev->get_time_now() + uhd::time_spec_t(seconds_in_future);
            rx_stream->issue_stream_cmd(stream_cmd);
        } catch(const uhd::exception &e) {
            error = "UHD exception occurred: " + std::string(e.what());
        } catch(...) {
            error = "Unkown exception occurred.";
        }

        if (!error.empty()) {
            /** An error occurred **/
            std::lock_guard<std::mutex> lg(_results_lock);
            _results_queue.push(new ReceiveResult(std::move(error)));
            _receiving = false;
            _results_notify.notify_one();
            continue;
        }

        ReceiveResult *result = nullptr;

        do {
            if (!result)
                result = new ReceiveResult(num_channels, num_samps);

            try {
                uhd::rx_metadata_t md;
                size_t num_samps_recvd = rx_stream->recv(result->bufs, num_samps, md, next_timeout, false);
                next_timeout = timeout;
                (void)num_samps_recvd;
                if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE
                    && md.error_code != uhd::rx_metadata_t::ERROR_CODE_LATE_COMMAND) {
                    switch (md.error_code) {
                        case uhd::rx_metadata_t::ERROR_CODE_TIMEOUT:
                            error = "Timeout error: No packet received, implementation timed-out.";
                            break;
                        case uhd::rx_metadata_t::ERROR_CODE_LATE_COMMAND:
                            error = "Late command error: A stream command was issued in the past.";
                            break;
                        case uhd::rx_metadata_t::ERROR_CODE_BROKEN_CHAIN:
                            error = "Broken chain error: Expected another stream command.";
                            break;
                        case uhd::rx_metadata_t::ERROR_CODE_OVERFLOW:
                            error = "Overflow error: An internal receive buffer has filled or a sequence error has been detected.";
                            break;
                        case uhd::rx_metadata_t::ERROR_CODE_ALIGNMENT:
                            error = "Code alignment error: Multi-channel alignment failed.";
                            break;
                        case uhd::rx_metadata_t::ERROR_CODE_BAD_PACKET:
                            error = "Bad packet error: The packet could not be parsed.";
                            break;
                        default:
                            error = "Unknown error: " + md.strerror();
                            break;
                    }
                } else {
                    result->start = md.time_spec;
                }
            } catch(const uhd::exception &e) {
                error = "UHD exception occurred: " + std::string(e.what());
            } catch(...) {
                error = "An unknown exception occurred.";
            }

            if (!error.empty()) {
                /** An error occurred **/
                for (auto &ptr : result->bufs)
                    free(ptr);
                result->message = std::move(error);
                result->error = true;
                streaming = false;
            }

            /** Push result into queue **/
            {
                std::lock_guard<std::mutex> lg(_results_lock);
                _results_queue.push(result);
                _receiving = streaming;
                if (req_type == ReceiveRequestType::Recycle && _results_queue.size() > 1) {
                    /** Recycle old un-claimed result **/
                    result = _results_queue.front();
                    _results_queue.pop();
                } else {
                    result = nullptr;
                }
                _results_notify.notify_one();
            }

            /** Check for new requests **/
            {
                std::lock_guard<std::mutex> lg(_requests_lock);
                if (!_requests_queue.empty())
                    break;
            }
        } while(streaming);

        /** Shutdown receiver **/
        stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS;
        stream_cmd.stream_now = true;
        rx_stream->issue_stream_cmd(stream_cmd);

        /** Cleanup result **/
        if (result) {
            for (auto &ptr : result->bufs)
                free(ptr);
            delete result;
            result = nullptr;
        }
    }
}

size_t ReceiveWorker::num_received() {
    std::lock_guard<std::mutex> lg(_results_lock);
    return _results_queue.size();
}

}
