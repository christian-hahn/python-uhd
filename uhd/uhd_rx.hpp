#ifndef __UHD_RX_HPP__
#define __UHD_RX_HPP__

#include <mutex>
#include <vector>
#include <future>
#include <queue>
#include <condition_variable>

#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>

namespace uhd {

enum class ReceiveRequestType {
    Single,
    Continuous,
    Recycle,
    Stop,
    Exit
};

struct ReceiveRequest {
    ReceiveRequest(const ReceiveRequestType type) : type(type) {}
    ReceiveRequest(const ReceiveRequestType type, const size_t num_samps, std::vector<long unsigned int> &&channels,
                   const double seconds_in_future, const double timeout, const std::string &otw_format)
        : type(type), num_samps(num_samps), channels(std::move(channels)), seconds_in_future(seconds_in_future),
          timeout(timeout), otw_format(otw_format) {}
    ReceiveRequestType type;
    size_t num_samps;
    std::vector<long unsigned int> channels;
    double seconds_in_future, timeout;
    std::string otw_format;
    std::promise<void> accepted;
};

struct ReceiveResult {
    ReceiveResult(std::string &&error) : num_samps(0), error(true), message(std::move(error)) {}
    ReceiveResult(const size_t num_channels, const size_t num_samps) : num_samps(num_samps), bufs(std::vector<float *>(num_channels)), error(false) {
        for (auto &ptr : bufs)
            ptr = reinterpret_cast<float *>(malloc(sizeof(float) * 2  * num_samps));
    }
    size_t num_samps;
    std::vector<float *> bufs;
    time_spec_t start;
    bool error;
    std::string message;
};

class ReceiveWorker {

  public:
    ReceiveWorker(uhd::usrp::multi_usrp::sptr dev, std::mutex &dev_lock);
    ~ReceiveWorker();
    void init();
    std::future<void> make_request(const ReceiveRequestType type);
    std::future<void> make_request(const ReceiveRequestType type, const size_t num_samps,
                                   std::vector<long unsigned int> &&channels,
                                   const double seconds_in_future, const double timeout,
                                   const std::string &otw_format);
    ReceiveResult *get_result(const bool fresh = false);
    size_t num_received();

  private:
    void _worker();

    uhd::usrp::multi_usrp::sptr _dev;
    std::mutex &_dev_lock;
    std::thread _thread;
    std::atomic<bool> _receiving;

    /** Requests queue **/
    std::mutex _requests_lock;
    std::condition_variable _requests_notify;
    std::queue<ReceiveRequest *> _requests_queue;

    /** Results queue **/
    std::mutex _results_lock;
    std::condition_variable _results_notify;
    std::queue<ReceiveResult *> _results_queue;
};

}

#endif  /** __UHD_RX_HPP__ **/
