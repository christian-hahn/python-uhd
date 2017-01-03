#ifndef __UHD_EXPECT_HPP__
#define __UHD_EXPECT_HPP__

namespace uhd {

class Error {
  private:
    const std::string _what;

  public:
    Error(std::string what) : _what(what) {}
    std::string what() const { return _what; }
};

template <typename T>
class Expect {
  private:
    T _value;
    std::string _what;
    bool _success;

  public:
    Expect() : _success(false) {}
    Expect(T &&value) : _value(std::move(value)), _success(true) {}
    Expect(const T &value) : _value(value), _success(true) {}
    Expect(Error &&error) : _what(std::move(error.what())), _success(false) {}
    Expect(const Error &error) : _what(error.what()), _success(false) {}
    Expect(const Expect &other) : _value(other._value), _what(other._what), _success(other._success) {}
    Expect(Expect &&other) : _value(std::move(other._value)), _what(std::move(other._what)), _success(other._success) {}

    Expect &operator=(Expect &&other) {
        _value = std::move(other._value);
        _what = std::move(other._what);
        _success = other._success;
        return *this;
    }
    Expect &operator=(const Expect &other) = delete;

    operator bool() const { return _success; }
    T &get() { return _value; }
    const char *what() { return _what.c_str(); }

};

template <>
class Expect<void> {
  private:
    std::string _what;
    bool _success;

  public:
    Expect() : _success(true) {}
    Expect(Error &&error) : _what(std::move(error.what())), _success(false) {}
    Expect(const Error &error) : _what(error.what()), _success(false) {}
    Expect(const Expect &other) : _what(other._what), _success(other._success) {}
    Expect(Expect &&other) : _what(std::move(other._what)), _success(other._success) {}

    Expect &operator=(Expect &&other) {
        _what = std::move(other._what);
        _success = other._success;
        return *this;
    }
    Expect &operator=(const Expect &other) = delete;

    operator bool() const { return _success; }
    const char *what() { return _what.c_str(); }
};

}

#endif  /** __UHD_EXPECT_HPP__ **/
