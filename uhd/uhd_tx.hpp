#ifndef __UHD_TX_HPP__
#define __UHD_TX_HPP__

#include <mutex>
#include <vector>
#include <future>
#include <queue>
#include <condition_variable>

#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>

namespace uhd {

enum class TransmitRequestType {
    Single,
    Continuous,
    Stop,
    Exit
};

struct TransmitRequest {
    TransmitRequest(const TransmitRequestType type) : type(type) {}
    TransmitRequest(const TransmitRequestType type, const size_t num_samps, std::vector<std::complex<float> *> &&samps,
                    std::vector<long unsigned int> &&channels, const double seconds_in_future, const double timeout,
                    const std::string &otw_format)
        : type(type), num_samps(num_samps), samps(std::move(samps)), channels(std::move(channels)),
          seconds_in_future(seconds_in_future), timeout(timeout), otw_format(otw_format) {}
    TransmitRequestType type;
    size_t num_samps;
    std::vector<std::complex<float> *> samps;
    std::vector<long unsigned int> channels;
    double seconds_in_future, timeout;
    std::string otw_format;
    std::promise<std::string> accepted;
};

class TransmitWorker {

  public:
    TransmitWorker(uhd::usrp::multi_usrp::sptr dev, std::mutex &dev_lock);
    ~TransmitWorker();
    void init();
    std::future<std::string> make_request(const TransmitRequestType type);
    std::future<std::string> make_request(const TransmitRequestType type, const size_t num_samps,
                                          std::vector<std::complex<float> *> &&samps,
                                          std::vector<long unsigned int> &&channels,
                                          const double seconds_in_future, const double timeout,
                                          const std::string &otw_format);

  private:
    void _worker();

    uhd::usrp::multi_usrp::sptr _dev;
    std::mutex &_dev_lock;
    std::thread _thread;
    std::atomic<bool> _transmitting;

    /** Requests queue **/
    std::mutex _requests_lock;
    std::condition_variable _requests_notify;
    std::queue<TransmitRequest *> _requests_queue;
};

}

#endif  /** __UHD_TX_HPP__ **/
