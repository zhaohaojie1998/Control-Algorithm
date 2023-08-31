#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <deque>
#include <list>
#include <set>
#include <map>
#include <type_traits> // from typing import ...

/*
    str
    len
    int_
    bool_ 有bug
    hex
    oct
    bin
    ord
    chr
    id
    print
    range

    小目标:
    list_
    dict_
    deque_
    set_
*/


namespace cpython {
    // str
    using str = std::string;
    

    // len
    template<typename T> // NOTE: 这里定义了模板, 在分解成h和cpp中报错 LNK2019, 需要将cpp和h结合成hpp
    size_t len(const T& container) {
        return container.size();
    }

    template<typename T, size_t N>
    size_t len(const T(&arr)[N]) {
        return N;
    }


    // 进制转换函数
    int _BaseToDec(str num, int base = 0) {
        if (base == 0) {
            base = 10;
        }
        size_t start = 0;
        if (num[1] < '0' || num[1] > '9') {
            start = 2; // 0x等开头
        }
        int result = 0;
        int power = 1;
        for (size_t i = num.size() - 1; i >= start; i--) {
            int digit;
            if (num[i] >= '0' && num[i] <= '9') {
                digit = num[i] - '0';
            }
            else {
                digit = num[i] - 'A' + 10;
            }
            result += digit * power;
            power *= base;
        }
        return result;
    }

    str _DecToBase(int num, int base, bool with_start) {
        str result = "";
        while (num > 0) {
            int remainder = num % base;
            if (remainder < 10) {
                result = std::to_string(remainder) + result;
            }
            else {
                result = static_cast<char>('A' + remainder - 10) + result;
            }
            num /= base;
        }
        if (with_start) {
            switch (base) {
            case 2:
                return "0b" + result; // return了就不用写break了
            case 8:
                // return "0" + result;
                return "0o" + result;
            case 16:
                return "0x" + result;
            default:
                return result;
            }
        }
        return result;
    }


    // str -> 10进制
    int int_(str s, int base = 10) {
        return _BaseToDec(s, base);
    }


    // dec -> 16/8/2进制
    str hex(int x, bool with_start = true) {
        return _DecToBase(x, 16, with_start);
    }

    str oct(int x, bool with_start = true) {
        return _DecToBase(x, 8, with_start);
    }

    str bin(int x, bool with_start = true) {
        return _DecToBase(x, 2, with_start);
    }


    // ASCII转换
    int ord(char c) {
        return static_cast<int>(c);
    }

    char chr(int ascii) {
        return static_cast<char>(ascii);
    }


    // id获取
    template <typename T>
    long id(T* p) {
        return reinterpret_cast<long>(p);
    }


    // 打印函数的构造函数
    namespace _PyPrint {
        // 打印基本类型
        template <typename T>
        struct printValue {
            static void print(const T& value) {  //模板结构体只能在用的时候根据传入的数据类型实例化，无法定义非静态成员函数
                std::cout << value;
            }
        };

        template <size_t N>
        struct printValue<char[N]> {
            static void print(const char(&value)[N]) {
                std::cout << value;
            }
        };

        // 打印C语言数组
        template <typename T, size_t N>
        struct printValue<T[N]> {
            static void print(const T(&value)[N]) {
                std::cout << "[";
                for (size_t i = 0; i < N; ++i) {
                    if (i > 0) {
                        std::cout << ", ";
                    }
                    printValue<T>::print(value[i]);
                }
                std::cout << "]";
            }
        };

        // 打印vector,list,deque
        template <typename T>
        struct printValue<std::vector<T>> {
            static void print(const std::vector<T>& value) {
                std::cout << "[";
                bool first = true;
                for (const auto& element : value) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    printValue<T>::print(element);
                    first = false;
                }
                std::cout << "]";
            }
        };
        template <typename T>
        struct printValue<std::list<T>> {
            static void print(const std::list<T>& value) {
                std::cout << "[";
                bool first = true;
                for (const auto& element : value) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    printValue<T>::print(element);
                    first = false;
                }
                std::cout << "]";
            }
        };
        template <typename T>
        struct printValue<std::deque<T>> {
            static void print(const std::deque<T>& value) {
                std::cout << "[";
                bool first = true;
                for (const auto& element : value) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    printValue<T>::print(element);
                    first = false;
                }
                std::cout << "]";
            }
        };

        // 打印array
        template <typename T, size_t N>
        struct printValue<std::array<T, N>> {
            static void print(const std::array<T, N>& value) {
                std::cout << "[";
                bool first = true;
                for (const auto& element : value) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    printValue<T>::print(element);
                    first = false;
                }
                std::cout << "]";
            }
        };

        // 打印set
        template <typename T>
        struct printValue<std::set<T>> {
            static void print(const std::set<T>& value) {
                std::cout << "{";
                bool first = true;
                for (const auto& element : value) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    printValue<T>::print(element);
                    first = false;
                }
                std::cout << "}";
            }
        };

        // 打印map
        template <typename K, typename V>
        struct printValue<std::map<K, V>> {
            static void print(const std::map<K, V>& value) {
                std::cout << "{";
                bool first = true;
                for (const auto& element : value) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    std::cout << element.first << ": "; // key一般是int和str，直接cout
                    printValue<V>::print(element.second);
                    first = false;
                }
                std::cout << "}";
            }
        };

        // 打印pair
        template<typename L, typename R>
        struct printValue<std::pair<L, R>> {
            static void print(const std::pair<L, R>& value) {
                std::cout << "<";
                printValue<L>::print(value.first);
                std::cout << ", ";
                printValue<R>::print(value.second);
                std::cout << ">";
            }
        };

        // 打印tuple
        template <typename... Args>
        struct printValue<std::tuple<Args...>> {
            static void print(const std::tuple<Args...>& value) {
                std::cout << "(";
                bool first = true;
                std::apply([&first](const auto&... args) {
                    ((std::cout << (first ? "" : ", "), printValue<typename std::decay<decltype(args)>::type>::print(args), first = false), ...);
                    }, value);
                std::cout << ")";
            }
        };

        // 打印输出 print(value, end='\n')
        template <typename T>
        void pyprint(const T& value, std::string end = "\n") {
            printValue<T>::print(value);
            std::cout << end;
        }
    }
    

    // python打印print
    template <typename... Ts>
    void print(const Ts&... values) {
        ((_PyPrint::pyprint(values, " ")), ...);
        std::cout << "\n";
    }

    void print() {
        std::cout << std::endl;
    }


    // bool_可能有BUG
    //template<typename T>
    //typename std::enable_if<std::is_arithmetic<T>::value || std::is_same<T, bool>::value, bool>::type
    //bool_(T value) {
    //    return value != 0; // Number
    //}
    //template<typename T>
    //typename std::enable_if<!std::is_arithmetic<T>::value && !std::is_same<T, bool>::value && !std::is_pointer<T>::value && !std::is_array<T>::value, bool>::type
    //bool_(const T& container) {
    //    return !container.empty(); // container
    //}
    //template<typename T>
    //bool bool_(T* ptr) {
    //    return ptr != nullptr; // ptr
    //}
    //template<typename T, size_t N>
    //bool bool_(T(&arr)[N]) {
    //    return N != 0; // array
    //}
    //bool bool_(const char* s) {
    //    return s != nullptr && std::strlen(s) != 0; // cstring
    //}


    // range迭代器
    class _Range {
    private:
        int start, stop, step;
    public:
        _Range(int stop) : start(0), stop(stop), step(1) {}
        _Range(int start, int stop, int step = 1) : start(start), stop(stop), step(step) {}

        class Iterator {
        private:
            int value;
            int step;
            int stop;
        public:
            Iterator(int value, int step, int stop) : value(value), step(step), stop(stop) {}

            bool operator != (const Iterator& other) const {
                return value != other.value;
            }

            int operator * () const {
                return value;
            }

            Iterator& operator ++ () {
                value += step;
                if ((step > 0 && value >= stop) || (step < 0 && value <= stop)) {
                    value = stop;
                }
                return *this;
            }

            Iterator& operator -- () {
                value -= step;
                if ((step > 0 && value <= stop) || (step < 0 && value >= stop)) {
                    value = stop;
                }
                return *this;
            }
        };

        Iterator begin() const {
            return Iterator(start, step, stop);
        }

        Iterator end() const {
            return Iterator(stop, step, stop);
        }
    };


    // python的range函数
    _Range range(int stop) {
        return _Range(stop);
    }

    _Range range(int start, int stop, int step = 1) {
        return _Range(start, stop, step);
    }

}
