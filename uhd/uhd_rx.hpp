#ifndef __UHD_RX_HPP__
#define __UHD_RX_HPP__

#include <mutex>
#include <vector>
#include <future>

#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>

#include "uhd_expect.hpp"
#include "uhd_queue.hpp"

namespace uhd {

enum class ReceiveRequestType {
    Single,
    Continuous,
    Stop,
    Exit
};

struct ReceiveRequest {
    ReceiveRequest(ReceiveRequestType type, size_t num_samps, std::vector<size_t> &&channels,
                   double seconds_in_future, double timeout)
    : type(type), num_samps(num_samps), channels(std::move(channels)),
      seconds_in_future(seconds_in_future), timeout(timeout) {}
    ReceiveRequestType type;
    size_t num_samps;
    std::vector<size_t> channels;
    double seconds_in_future, timeout;
    std::promise<void> accepted;
};

struct ReceiveResult {
    std::vector<float *> bufs;
    size_t num_samps;
};

class ReceiveWorker {

  public:
    ReceiveWorker(uhd::usrp::multi_usrp::sptr dev, std::mutex &dev_lock);
    ~ReceiveWorker();
    void init();
    std::future<void> request(ReceiveRequestType type, size_t num_samps = 0, std::vector<size_t> &&channels = {},
                              double seconds_in_future = 0.0, double timeout = 0.0);
    Expect<ReceiveResult> read();
    bool stream_in_progress();

  private:
    void worker();

    uhd::usrp::multi_usrp::sptr dev;
    std::mutex &dev_lock;
    std::thread thread;
    Queue<std::unique_ptr<ReceiveRequest>> requests;
    Queue<Expect<ReceiveResult>> results;
    std::atomic<bool> _streaming, _receiving;
};

}

#endif  /** __UHD_RX_HPP__ **/
