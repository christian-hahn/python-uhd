#include <Python.h>

#include <mutex>
#include <vector>
#include <memory>
#include <thread>

#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
#include <uhd/utils/thread_priority.hpp>

#include "uhd_types.hpp"
#include "uhd_expect.hpp"
#include "uhd_rx.hpp"

namespace uhd {

ReceiveWorker::ReceiveWorker(uhd::usrp::multi_usrp::sptr dev, std::mutex &dev_lock) : dev(dev), dev_lock(dev_lock) {
    _streaming = false;
    _receiving = false;
}

ReceiveWorker::~ReceiveWorker() {
    std::future<void> accepted = request(ReceiveRequestType::Exit, 0, {});
    accepted.wait();
    thread.join();

    /** Reclaim results **/
    while (!results.empty()) {
        Expect<ReceiveResult> result;
        if ((result = results.pop())) {
            for (auto &b : result.get().bufs)
                free(b);
        }
    }
}

std::future<void> ReceiveWorker::request(ReceiveRequestType type, size_t num_samps, std::vector<size_t> &&channels,
                                         double seconds_in_future, double timeout) {

    std::unique_ptr<ReceiveRequest> request = std::unique_ptr<ReceiveRequest>(new ReceiveRequest(type, num_samps, std::move(channels),
                                                                              seconds_in_future, timeout));
    std::future<void> accepted = request->accepted.get_future();
    requests.push(std::move(request));
    return accepted;
}

Expect<ReceiveResult> ReceiveWorker::read() {

    if (results.empty() && _receiving == false)
        return Error("Not streaming, or no data available.");

    return results.pop();
}

void ReceiveWorker::init() {
    thread = std::thread(&ReceiveWorker::worker, this);
}

void ReceiveWorker::worker() {

    uhd::set_thread_priority_safe(1.0, true);

    _streaming = false;
    _receiving = false;

    while (true) {

        _receiving = false;

        /** Get request **/
        std::unique_ptr<ReceiveRequest> req = requests.pop();

        /** Reclaim results **/
        while (!results.empty()) {
            Expect<ReceiveResult> result;
            if ((result = results.pop())) {
                for (auto &b : result.get().bufs)
                    free(b);
            }
        }

        if (req->type == ReceiveRequestType::Single ||
            req->type == ReceiveRequestType::Continuous)
            _receiving = true;

        /** Accept request **/
        req->accepted.set_value();

        if (req->type == ReceiveRequestType::Stop)
            continue;
        else if (req->type == ReceiveRequestType::Exit)
            break;

        const size_t num_channels = req->channels.size();
        const size_t num_samps = req->num_samps;

        _streaming = (req->type == ReceiveRequestType::Continuous) ? true : false;

        uhd::stream_args_t stream_args("fc32", "sc16");
        stream_args.channels = std::move(req->channels);

        uhd::stream_cmd_t stream_cmd(_streaming ? uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS
                                     : uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE);
        uhd::rx_streamer::sptr rx_stream;

        try {
            /** Lock device **/
            std::lock_guard<std::mutex> lg(dev_lock);

            dev->set_time_now(uhd::time_spec_t(0.0));
            rx_stream = dev->get_rx_stream(stream_args);

            stream_cmd.num_samps = num_samps;

            if (num_channels == 1) {
                stream_cmd.stream_now = true;
                stream_cmd.time_spec = uhd::time_spec_t(0.0);
            } else {
                stream_cmd.stream_now = false;
                stream_cmd.time_spec = uhd::time_spec_t(req->seconds_in_future);
            }

            rx_stream->issue_stream_cmd(stream_cmd);

        } catch(const uhd::exception &e) {
            results.push(Error("UHD exception occurred: " + std::string(e.what())));
            continue;
        } catch(...) {
            results.push(Error("Unkown exception occurred."));
            continue;
        }

        do {

            ReceiveResult result;

            result.num_samps = num_samps;
            result.bufs = std::vector<float *>(num_channels);
            for (auto &b : result.bufs)
                b = reinterpret_cast<float *>(malloc(sizeof(float) * 2  * result.num_samps));

            uhd::rx_metadata_t md;

            try {
                size_t num_samps_recvd = rx_stream->recv(result.bufs, result.num_samps, md, req->timeout, false);
                (void)num_samps_recvd;
            } catch(const uhd::exception &e) {
                results.push(Error("UHD exception occurred: " + std::string(e.what())));
                continue;
            } catch(...) {
                results.push(Error("Unkown exception occurred. "));
                continue;
            }

            if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE
                && md.error_code != uhd::rx_metadata_t::ERROR_CODE_LATE_COMMAND) {
                switch (md.error_code) {
                    case uhd::rx_metadata_t::ERROR_CODE_TIMEOUT:
                        results.push(Error("Timeout error: No packet received, implementation timed-out."));
                        break;
                    case uhd::rx_metadata_t::ERROR_CODE_LATE_COMMAND:
                        results.push(Error("Late command error: A stream command was issued in the past."));
                        break;
                    case uhd::rx_metadata_t::ERROR_CODE_BROKEN_CHAIN:
                        results.push(Error("Broken chain error: Expected another stream command."));
                        break;
                    case uhd::rx_metadata_t::ERROR_CODE_OVERFLOW:
                        results.push(Error("Overflow error: An internal receive buffer has filled or a sequence error has been detected."));
                        break;
                    case uhd::rx_metadata_t::ERROR_CODE_ALIGNMENT:
                        results.push(Error("Code alignment error: Multi-channel alignment failed."));
                        break;
                    case uhd::rx_metadata_t::ERROR_CODE_BAD_PACKET:
                        results.push(Error("Bad packet error: The packet could not be parsed."));
                        break;
                    default:
                        results.push(Error("Unknown error: " + md.strerror()));
                        break;
                }
                /** An error occurred **/
                for (auto &b : result.bufs)
                    free(b);
                _streaming = false;
            } else {
                /** No error **/
                results.push(std::move(result));
            }
        } while (requests.empty() && _streaming);

        /** Shutdown receiver **/
        stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS;
        stream_cmd.stream_now = true;
        rx_stream->issue_stream_cmd(stream_cmd);
    }
}

bool ReceiveWorker::stream_in_progress() {
    return _streaming;
}

size_t ReceiveWorker::num_received() {
    return results.size();
}

}
