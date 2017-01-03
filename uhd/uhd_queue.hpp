#ifndef __UHD_QUEUE_HPP__
#define __UHD_QUEUE_HPP__

#include <queue>
#include <mutex>
#include <condition_variable>

namespace uhd {

template <typename T>
class Queue {
  public:
    Queue() {}
    void push(T &&value);
    void push(const T &value);
    T pop();
    void clear();
    bool empty();

  private:
    std::mutex _lock;
    std::condition_variable _notify;
    std::queue<T> _queue;
};

template <typename T>
void Queue<T>::push(T &&value) {
    std::lock_guard<std::mutex> lg(_lock);
    _queue.push(std::move(value));
    _notify.notify_one();
}

template <typename T>
T Queue<T>::pop() {
    std::unique_lock<std::mutex> lg(_lock);
    while (_queue.empty())
        _notify.wait(lg);
    T value(std::move(_queue.front()));
    _queue.pop();
    return value;
}

template <typename T>
void Queue<T>::clear() {
    std::lock_guard<std::mutex> lg(_lock);
    while (!_queue.empty())
        _queue.pop();
}

template <typename T>
bool Queue<T>::empty() {
    std::lock_guard<std::mutex> lg(_lock);
    return _queue.empty();
}

}

#endif  /** __UHD_QUEUE_HPP__ **/
