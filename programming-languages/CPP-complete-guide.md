# C++ 완벽 학습 가이드

## 목차
1. [C++의 기초와 C와의 차이점](#1-c의-기초와-c와의-차이점)
2. [입출력 스트림](#2-입출력-스트림)
3. [참조자와 포인터](#3-참조자와-포인터)
4. [함수 오버로딩과 기본 매개변수](#4-함수-오버로딩과-기본-매개변수)
5. [클래스와 객체](#5-클래스와-객체)
6. [생성자와 소멸자](#6-생성자와-소멸자)
7. [연산자 오버로딩](#7-연산자-오버로딩)
8. [상속](#8-상속)
9. [다형성과 가상 함수](#9-다형성과-가상-함수)
10. [템플릿](#10-템플릿)
11. [STL (표준 템플릿 라이브러리)](#11-stl-표준-템플릿-라이브러리)
12. [예외 처리](#12-예외-처리)
13. [네임스페이스](#13-네임스페이스)
14. [형변환 연산자](#14-형변환-연산자)
15. [스마트 포인터](#15-스마트-포인터)
16. [이동 의미론과 rvalue 참조](#16-이동-의미론과-rvalue-참조)
17. [람다 표현식](#17-람다-표현식)
18. [모던 C++ (C++11/14/17/20/23)](#18-모던-c-c111417202현대적-c)
19. [동시성과 멀티스레딩](#19-동시성과-멀티스레딩)
20. [메타프로그래밍과 고급 기법](#20-메타프로그래밍과-고급-기법)

---

## 1. C++의 기초와 C와의 차이점

### 1.1 첫 C++ 프로그램

```cpp
#include <iostream>

int main() {
    std::cout << "Hello, C++!" << std::endl;
    return 0;
}

// 컴파일: g++ -std=c++20 program.cpp -o program
```

### 1.2 C와의 주요 차이점

```cpp
// 1. 주석
// 한 줄 주석 (C++스타일, C99부터 C에서도 가능)
/* 여러 줄 주석 */

// 2. bool 타입 (키워드)
bool flag = true;  // C++에서는 키워드
// C에서는 <stdbool.h> 필요

// 3. 변수 선언 위치
for (int i = 0; i < 10; i++) {  // C++에서 가능
    int x = i * 2;  // 어디서든 선언 가능
}

// 4. const의 의미
const int SIZE = 100;
int arr[SIZE];  // C++에서는 가능, C에서는 불가 (VLA 제외)

// 5. 구조체
struct Point {
    int x, y;
};
Point p;  // struct 키워드 생략 가능

// 6. 함수 오버로딩
int add(int a, int b);
double add(double a, double b);  // C++에서 가능

// 7. 기본 매개변수
void func(int x = 10, int y = 20);

// 8. 네임스페이스
namespace MyNamespace {
    int value;
}

// 9. 참조자
int x = 10;
int &ref = x;  // 별칭

// 10. new/delete
int *ptr = new int(10);
delete ptr;
// vs C의 malloc/free
```

### 1.3 입력과 출력

```cpp
#include <iostream>
#include <string>

int main() {
    // 출력
    std::cout << "Hello, World!" << std::endl;

    int num = 42;
    std::cout << "Number: " << num << std::endl;

    // 입력
    int age;
    std::cout << "Enter age: ";
    std::cin >> age;

    // 문자열 입력
    std::string name;
    std::cout << "Enter name: ";
    std::cin >> name;  // 공백 전까지

    // 한 줄 입력
    std::string line;
    std::getline(std::cin, line);

    return 0;
}
```

### 1.4 동적 메모리 할당

```cpp
// new/delete
int *ptr = new int;      // 단일 객체
*ptr = 10;
delete ptr;

int *arr = new int[10];  // 배열
delete[] arr;            // 배열 삭제

// 초기화
int *p1 = new int(42);        // 42로 초기화
int *p2 = new int{42};        // C++11 유니폼 초기화
int *arr2 = new int[5]{1, 2, 3, 4, 5};

// 배치 new (placement new)
#include <new>
char buffer[sizeof(int)];
int *p = new (buffer) int(42);  // buffer 위치에 생성
p->~int();  // 명시적 소멸자 호출 (delete 호출 안함)
```

---

## 2. 입출력 스트림

### 2.1 iostream 기본

```cpp
#include <iostream>
#include <iomanip>

int main() {
    using namespace std;

    // 출력 포맷
    int num = 42;
    cout << "Decimal: " << num << endl;
    cout << "Hex: " << hex << num << endl;       // 2a
    cout << "Oct: " << oct << num << endl;       // 52
    cout << "Dec: " << dec << num << endl;       // 42

    // 부동소수점 포맷
    double pi = 3.14159265358979;
    cout << "Default: " << pi << endl;
    cout << "Fixed: " << fixed << pi << endl;
    cout << "Scientific: " << scientific << pi << endl;
    cout << "Precision: " << setprecision(3) << pi << endl;

    // 정렬과 폭
    cout << setw(10) << "Right" << endl;
    cout << left << setw(10) << "Left" << endl;
    cout << setfill('*') << setw(10) << 42 << endl;

    return 0;
}
```

### 2.2 파일 입출력

```cpp
#include <fstream>
#include <iostream>
#include <string>

int main() {
    // 파일 쓰기
    std::ofstream outfile("output.txt");
    if (!outfile) {
        std::cerr << "Cannot open file" << std::endl;
        return 1;
    }

    outfile << "Hello, File!" << std::endl;
    outfile << "Number: " << 42 << std::endl;
    outfile.close();

    // 파일 읽기
    std::ifstream infile("input.txt");
    if (!infile) {
        std::cerr << "Cannot open file" << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::cout << line << std::endl;
    }
    infile.close();

    // 바이너리 파일
    std::ofstream binout("data.bin", std::ios::binary);
    int data[] = {1, 2, 3, 4, 5};
    binout.write(reinterpret_cast<char*>(data), sizeof(data));
    binout.close();

    std::ifstream binin("data.bin", std::ios::binary);
    int read_data[5];
    binin.read(reinterpret_cast<char*>(read_data), sizeof(read_data));
    binin.close();

    return 0;
}
```

### 2.3 문자열 스트림

```cpp
#include <sstream>
#include <string>
#include <iostream>

int main() {
    // ostringstream - 문자열 만들기
    std::ostringstream oss;
    oss << "Age: " << 25 << ", Score: " << 95.5;
    std::string result = oss.str();
    std::cout << result << std::endl;

    // istringstream - 문자열 파싱
    std::string data = "42 3.14 Hello";
    std::istringstream iss(data);

    int num;
    double pi;
    std::string word;
    iss >> num >> pi >> word;

    std::cout << num << " " << pi << " " << word << std::endl;

    // stringstream - 양방향
    std::stringstream ss;
    ss << 100;
    int value;
    ss >> value;

    return 0;
}
```

---

## 3. 참조자와 포인터

### 3.1 참조자 기본

```cpp
// 참조자: 변수의 별칭
int x = 10;
int &ref = x;  // ref는 x의 별칭

ref = 20;  // x도 20으로 변경
std::cout << x << std::endl;  // 20

// 참조자는 초기화 필수
// int &ref2;  // 에러!

// 참조자는 재할당 불가
int y = 30;
ref = y;  // ref가 y를 참조하는 것이 아니라, x = y
std::cout << x << std::endl;  // 30

// const 참조자
const int &cref = x;
// cref = 40;  // 에러! const 참조자는 수정 불가

// 임시 객체에 대한 const 참조
const int &temp_ref = 42;  // OK
// int &temp_ref2 = 42;  // 에러! 비const 참조는 불가
```

### 3.2 참조자와 함수

```cpp
// 값에 의한 전달 (복사)
void by_value(int x) {
    x = 100;  // 원본에 영향 없음
}

// 참조에 의한 전달 (별칭)
void by_reference(int &x) {
    x = 100;  // 원본 변경
}

// const 참조 (복사 비용 절감, 수정 방지)
void by_const_ref(const std::string &str) {
    std::cout << str << std::endl;
    // str[0] = 'X';  // 에러!
}

// 포인터와 비교
void by_pointer(int *ptr) {
    if (ptr != nullptr) {
        *ptr = 100;
    }
}

int main() {
    int num = 10;
    by_value(num);      // num은 여전히 10
    by_reference(num);  // num은 100
    by_pointer(&num);   // num은 100

    std::string str = "Hello";
    by_const_ref(str);  // 복사 없이 전달

    return 0;
}
```

### 3.3 참조자 반환

```cpp
// 참조자 반환
int& get_element(int arr[], int index) {
    return arr[index];
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    get_element(arr, 2) = 100;  // arr[2] = 100

    std::cout << arr[2] << std::endl;  // 100

    return 0;
}

// 댕글링 참조 (위험!)
int& dangerous() {
    int local = 10;
    return local;  // 지역 변수 참조 반환 - 위험!
}

// 안전한 방법
const std::string& safe() {
    static std::string str = "Safe";
    return str;
}
```

### 3.4 rvalue 참조 (C++11)

```cpp
// lvalue: 이름이 있고, 주소를 가질 수 있는 값
// rvalue: 임시 값, 이름이 없음

int x = 10;  // x는 lvalue, 10은 rvalue

// rvalue 참조
int &&rref = 20;  // rvalue만 바인딩 가능
// int &&rref2 = x;  // 에러! x는 lvalue

// 이동 의미론
std::string str1 = "Hello";
std::string str2 = std::move(str1);  // str1을 str2로 이동
// str1은 이제 비어있음 (moved-from 상태)

// 완벽한 전달 (Perfect Forwarding)
template<typename T>
void wrapper(T &&arg) {
    func(std::forward<T>(arg));
}
```

---

## 4. 함수 오버로딩과 기본 매개변수

### 4.1 함수 오버로딩

```cpp
#include <iostream>
#include <string>

// 같은 이름, 다른 매개변수
int add(int a, int b) {
    return a + b;
}

double add(double a, double b) {
    return a + b;
}

std::string add(const std::string &a, const std::string &b) {
    return a + b;
}

// 매개변수 개수로 오버로딩
void print(int x) {
    std::cout << x << std::endl;
}

void print(int x, int y) {
    std::cout << x << ", " << y << std::endl;
}

// const 오버로딩
void func(int &x) {
    std::cout << "Non-const reference" << std::endl;
}

void func(const int &x) {
    std::cout << "Const reference" << std::endl;
}

int main() {
    std::cout << add(1, 2) << std::endl;        // int 버전
    std::cout << add(1.5, 2.5) << std::endl;    // double 버전
    std::cout << add(std::string("Hello"), std::string(" World")) << std::endl;

    int x = 10;
    const int y = 20;
    func(x);  // Non-const
    func(y);  // Const
    func(30); // Const (rvalue)

    return 0;
}
```

### 4.2 기본 매개변수

```cpp
// 기본 매개변수는 오른쪽부터
void greet(const std::string &name, const std::string &greeting = "Hello") {
    std::cout << greeting << ", " << name << "!" << std::endl;
}

void set_value(int x, int y = 0, int z = 0) {
    std::cout << x << ", " << y << ", " << z << std::endl;
}

// 헤더 파일에서 선언
// header.h
void func(int x, int y = 10);

// source.cpp
void func(int x, int y /* = 10 */) {  // 구현부에서는 기본값 반복 안함
    std::cout << x + y << std::endl;
}

int main() {
    greet("Alice");              // Hello, Alice!
    greet("Bob", "Hi");          // Hi, Bob!

    set_value(1);                // 1, 0, 0
    set_value(1, 2);             // 1, 2, 0
    set_value(1, 2, 3);          // 1, 2, 3

    return 0;
}
```

### 4.3 inline 함수

```cpp
// inline 힌트 (컴파일러가 결정)
inline int max(int a, int b) {
    return (a > b) ? a : b;
}

// 작은 함수에 유용
inline int square(int x) {
    return x * x;
}

// 클래스 내부 정의된 함수는 자동으로 inline
class Math {
public:
    int add(int a, int b) {  // 암묵적 inline
        return a + b;
    }
};

// 큰 함수는 inline 부적합
inline void big_function() {
    // 많은 코드...
    // 컴파일러는 inline을 무시할 수 있음
}
```

### 4.4 constexpr 함수 (C++11)

```cpp
// 컴파일 타임 계산 가능
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

constexpr int square(int x) {
    return x * x;
}

int main() {
    // 컴파일 타임 계산
    constexpr int result = factorial(5);  // 120

    // 배열 크기로 사용 가능
    int arr[factorial(3)];  // int arr[6];

    // 런타임에도 사용 가능
    int n;
    std::cin >> n;
    int runtime_result = factorial(n);

    return 0;
}

// C++14: 더 복잡한 constexpr
constexpr int fibonacci(int n) {
    if (n <= 1) return n;

    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        int temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}
```

---

## 5. 클래스와 객체

### 5.1 클래스 기본

```cpp
#include <iostream>
#include <string>

class Person {
private:  // 접근 지정자
    std::string name;
    int age;

public:
    // 멤버 함수
    void set_name(const std::string &n) {
        name = n;
    }

    void set_age(int a) {
        if (a >= 0) {
            age = a;
        }
    }

    std::string get_name() const {  // const 멤버 함수
        return name;
    }

    int get_age() const {
        return age;
    }

    void print() const {
        std::cout << "Name: " << name << ", Age: " << age << std::endl;
    }
};

int main() {
    Person p;
    p.set_name("Alice");
    p.set_age(25);
    p.print();

    return 0;
}
```

### 5.2 struct vs class

```cpp
// struct: 기본이 public
struct Point {
    int x, y;  // public

    void print() {
        std::cout << x << ", " << y << std::endl;
    }
};

// class: 기본이 private
class Rectangle {
    int width, height;  // private

public:
    void set_size(int w, int h) {
        width = w;
        height = h;
    }
};

// C++에서는 struct도 멤버 함수, 생성자 등 가능
// 차이는 기본 접근 지정자뿐
```

### 5.3 접근 지정자

```cpp
class Example {
private:
    int private_var;      // 클래스 내부에서만 접근
    void private_func() {}

protected:
    int protected_var;    // 클래스와 파생 클래스에서 접근
    void protected_func() {}

public:
    int public_var;       // 어디서든 접근
    void public_func() {}

    void access_test() {
        private_var = 10;     // OK
        protected_var = 20;   // OK
        public_var = 30;      // OK
    }
};

class Derived : public Example {
public:
    void test() {
        // private_var = 10;  // 에러!
        protected_var = 20;   // OK
        public_var = 30;      // OK
    }
};
```

### 5.4 this 포인터

```cpp
class Counter {
private:
    int count;

public:
    Counter() : count(0) {}

    // this 포인터 사용
    Counter& increment() {
        this->count++;
        return *this;  // 자기 자신 반환
    }

    Counter& add(int n) {
        count += n;
        return *this;
    }

    void print() const {
        std::cout << count << std::endl;
    }

    // 체이닝 가능
    void method_chaining() {
        increment().add(5).add(10).print();
    }
};

int main() {
    Counter c;
    c.increment().increment().add(10).print();  // 12

    return 0;
}
```

### 5.5 정적 멤버

```cpp
class BankAccount {
private:
    std::string owner;
    double balance;
    static double interest_rate;  // 모든 객체가 공유
    static int account_count;

public:
    BankAccount(const std::string &name, double initial_balance)
        : owner(name), balance(initial_balance) {
        account_count++;
    }

    ~BankAccount() {
        account_count--;
    }

    static void set_interest_rate(double rate) {  // 정적 멤버 함수
        interest_rate = rate;
    }

    static double get_interest_rate() {
        return interest_rate;
    }

    static int get_account_count() {
        return account_count;
    }

    void apply_interest() {
        balance += balance * interest_rate;
    }
};

// 정적 멤버 초기화 (클래스 외부)
double BankAccount::interest_rate = 0.05;
int BankAccount::account_count = 0;

int main() {
    BankAccount::set_interest_rate(0.03);

    BankAccount acc1("Alice", 1000);
    BankAccount acc2("Bob", 2000);

    std::cout << "Total accounts: " << BankAccount::get_account_count() << std::endl;

    return 0;
}
```

### 5.6 friend

```cpp
class Box {
private:
    double width;

public:
    Box(double w) : width(w) {}

    // friend 함수
    friend void print_width(const Box &box);

    // friend 클래스
    friend class BoxPrinter;
};

void print_width(const Box &box) {
    std::cout << "Width: " << box.width << std::endl;  // private 접근 가능
}

class BoxPrinter {
public:
    void print(const Box &box) {
        std::cout << "Width: " << box.width << std::endl;  // private 접근 가능
    }
};

int main() {
    Box b(10.5);
    print_width(b);

    BoxPrinter printer;
    printer.print(b);

    return 0;
}
```

---

## 6. 생성자와 소멸자

### 6.1 생성자

```cpp
class Point {
private:
    int x, y;

public:
    // 기본 생성자
    Point() : x(0), y(0) {
        std::cout << "Default constructor" << std::endl;
    }

    // 매개변수 생성자
    Point(int x, int y) : x(x), y(y) {
        std::cout << "Parameterized constructor" << std::endl;
    }

    // 복사 생성자
    Point(const Point &other) : x(other.x), y(other.y) {
        std::cout << "Copy constructor" << std::endl;
    }

    // 이동 생성자 (C++11)
    Point(Point &&other) noexcept : x(other.x), y(other.y) {
        std::cout << "Move constructor" << std::endl;
        other.x = other.y = 0;
    }

    void print() const {
        std::cout << "(" << x << ", " << y << ")" << std::endl;
    }
};

int main() {
    Point p1;           // 기본 생성자
    Point p2(10, 20);   // 매개변수 생성자
    Point p3 = p2;      // 복사 생성자
    Point p4(std::move(p2));  // 이동 생성자

    return 0;
}
```

### 6.2 초기화 리스트

```cpp
class Rectangle {
private:
    int width, height;
    const int id;       // const 멤버
    int &ref;           // 참조 멤버

public:
    // const와 참조는 초기화 리스트에서만 초기화 가능
    Rectangle(int w, int h, int i, int &r)
        : width(w), height(h), id(i), ref(r) {
        // width = w;  // 대입 (비효율적)
    }

    // 효율성: 초기화 리스트가 더 효율적
    // 생성자 본문 전에 멤버를 직접 초기화
};

class Complex {
private:
    std::string name;  // 복잡한 객체
    int value;

public:
    // 초기화 리스트: name이 한 번만 생성됨
    Complex(const std::string &n, int v) : name(n), value(v) {}

    // 비효율적: name이 기본 생성 후 대입됨
    // Complex(const std::string &n, int v) {
    //     name = n;
    //     value = v;
    // }
};
```

### 6.3 위임 생성자 (C++11)

```cpp
class Person {
private:
    std::string name;
    int age;
    std::string address;

public:
    // 주 생성자
    Person(const std::string &n, int a, const std::string &addr)
        : name(n), age(a), address(addr) {
        std::cout << "Main constructor" << std::endl;
    }

    // 위임 생성자들
    Person() : Person("Unknown", 0, "") {
        std::cout << "Default constructor" << std::endl;
    }

    Person(const std::string &n) : Person(n, 0, "") {
        std::cout << "Name-only constructor" << std::endl;
    }

    Person(const std::string &n, int a) : Person(n, a, "") {
        std::cout << "Name and age constructor" << std::endl;
    }
};
```

### 6.4 소멸자

```cpp
class Resource {
private:
    int *data;
    int size;

public:
    Resource(int s) : size(s) {
        data = new int[size];
        std::cout << "Constructor: allocated " << size << " ints" << std::endl;
    }

    // 소멸자
    ~Resource() {
        delete[] data;
        std::cout << "Destructor: freed memory" << std::endl;
    }

    // 복사 생성자 (깊은 복사)
    Resource(const Resource &other) : size(other.size) {
        data = new int[size];
        std::copy(other.data, other.data + size, data);
    }

    // 복사 대입 연산자
    Resource& operator=(const Resource &other) {
        if (this != &other) {
            delete[] data;
            size = other.size;
            data = new int[size];
            std::copy(other.data, other.data + size, data);
        }
        return *this;
    }
};

int main() {
    {
        Resource r(100);
        // 블록 끝에서 자동으로 소멸자 호출
    }
    std::cout << "After block" << std::endl;

    return 0;
}
```

### 6.5 Rule of Three/Five/Zero

```cpp
// Rule of Three (C++03)
// 복사 생성자, 복사 대입, 소멸자 중 하나를 정의하면 셋 다 정의해야 함
class RuleOfThree {
public:
    RuleOfThree(const RuleOfThree &);          // 복사 생성자
    RuleOfThree& operator=(const RuleOfThree &);  // 복사 대입
    ~RuleOfThree();                            // 소멸자
};

// Rule of Five (C++11)
// 이동 생성자, 이동 대입 추가
class RuleOfFive {
public:
    RuleOfFive(const RuleOfFive &);            // 복사 생성자
    RuleOfFive& operator=(const RuleOfFive &); // 복사 대입
    RuleOfFive(RuleOfFive &&) noexcept;        // 이동 생성자
    RuleOfFive& operator=(RuleOfFive &&) noexcept;  // 이동 대입
    ~RuleOfFive();                             // 소멸자
};

// Rule of Zero
// 스마트 포인터 사용으로 특수 멤버 함수를 정의하지 않음
class RuleOfZero {
private:
    std::unique_ptr<int[]> data;  // 자동 메모리 관리
    std::string name;             // 자동 메모리 관리

public:
    RuleOfZero(int size, const std::string &n)
        : data(std::make_unique<int[]>(size)), name(n) {}

    // 컴파일러가 자동 생성하는 특수 멤버 함수로 충분
};
```

### 6.6 명시적 생성자

```cpp
class Integer {
private:
    int value;

public:
    // explicit: 암묵적 변환 방지
    explicit Integer(int v) : value(v) {}

    int get() const { return value; }
};

void func(Integer i) {
    std::cout << i.get() << std::endl;
}

int main() {
    Integer i1(10);     // OK
    Integer i2 = 20;    // 에러! explicit 때문에

    func(30);           // 에러! 암묵적 변환 불가
    func(Integer(40));  // OK (명시적 변환)

    return 0;
}

// explicit 없이
class ImplicitInteger {
private:
    int value;

public:
    ImplicitInteger(int v) : value(v) {}
    int get() const { return value; }
};

void func2(ImplicitInteger i) {
    std::cout << i.get() << std::endl;
}

int main() {
    ImplicitInteger i1 = 10;  // OK
    func2(20);                // OK (암묵적 변환)

    return 0;
}
```

---

## 7. 연산자 오버로딩

### 7.1 산술 연산자

```cpp
class Complex {
private:
    double real, imag;

public:
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}

    // + 연산자 (멤버 함수)
    Complex operator+(const Complex &other) const {
        return Complex(real + other.real, imag + other.imag);
    }

    // - 연산자
    Complex operator-(const Complex &other) const {
        return Complex(real - other.real, imag - other.imag);
    }

    // * 연산자
    Complex operator*(const Complex &other) const {
        return Complex(
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        );
    }

    // 단항 - 연산자
    Complex operator-() const {
        return Complex(-real, -imag);
    }

    void print() const {
        std::cout << real << " + " << imag << "i" << std::endl;
    }
};

int main() {
    Complex c1(3, 4);
    Complex c2(1, 2);

    Complex c3 = c1 + c2;  // operator+
    Complex c4 = c1 - c2;  // operator-
    Complex c5 = c1 * c2;  // operator*
    Complex c6 = -c1;      // operator- (단항)

    c3.print();

    return 0;
}
```

### 7.2 비교 연산자

```cpp
class Point {
private:
    int x, y;

public:
    Point(int x = 0, int y = 0) : x(x), y(y) {}

    // == 연산자
    bool operator==(const Point &other) const {
        return x == other.x && y == other.y;
    }

    // != 연산자
    bool operator!=(const Point &other) const {
        return !(*this == other);
    }

    // < 연산자 (거리 기준)
    bool operator<(const Point &other) const {
        return (x*x + y*y) < (other.x*other.x + other.y*other.y);
    }

    // C++20 우주선 연산자 <=>
    // auto operator<=>(const Point &) const = default;
};
```

### 7.3 대입 연산자

```cpp
class String {
private:
    char *data;
    size_t length;

public:
    String(const char *str = "") {
        length = std::strlen(str);
        data = new char[length + 1];
        std::strcpy(data, str);
    }

    // 복사 대입 연산자
    String& operator=(const String &other) {
        if (this != &other) {  // 자기 대입 체크
            delete[] data;     // 기존 메모리 해제

            length = other.length;
            data = new char[length + 1];
            std::strcpy(data, other.data);
        }
        return *this;
    }

    // 이동 대입 연산자 (C++11)
    String& operator=(String &&other) noexcept {
        if (this != &other) {
            delete[] data;

            data = other.data;
            length = other.length;

            other.data = nullptr;
            other.length = 0;
        }
        return *this;
    }

    // += 연산자
    String& operator+=(const String &other) {
        char *new_data = new char[length + other.length + 1];
        std::strcpy(new_data, data);
        std::strcat(new_data, other.data);

        delete[] data;
        data = new_data;
        length += other.length;

        return *this;
    }

    ~String() {
        delete[] data;
    }
};
```

### 7.4 입출력 연산자

```cpp
class Point {
private:
    int x, y;

public:
    Point(int x = 0, int y = 0) : x(x), y(y) {}

    // friend 함수로 << 오버로딩
    friend std::ostream& operator<<(std::ostream &os, const Point &p);
    friend std::istream& operator>>(std::istream &is, Point &p);
};

// << 연산자 (출력)
std::ostream& operator<<(std::ostream &os, const Point &p) {
    os << "(" << p.x << ", " << p.y << ")";
    return os;
}

// >> 연산자 (입력)
std::istream& operator>>(std::istream &is, Point &p) {
    is >> p.x >> p.y;
    return is;
}

int main() {
    Point p(10, 20);
    std::cout << p << std::endl;  // (10, 20)

    Point p2;
    std::cin >> p2;
    std::cout << p2 << std::endl;

    return 0;
}
```

### 7.5 증감 연산자

```cpp
class Counter {
private:
    int count;

public:
    Counter(int c = 0) : count(c) {}

    // 전위 증가 ++i
    Counter& operator++() {
        ++count;
        return *this;
    }

    // 후위 증가 i++
    Counter operator++(int) {  // int는 더미 매개변수
        Counter temp = *this;
        ++count;
        return temp;
    }

    // 전위 감소
    Counter& operator--() {
        --count;
        return *this;
    }

    // 후위 감소
    Counter operator--(int) {
        Counter temp = *this;
        --count;
        return temp;
    }

    int get() const { return count; }
};

int main() {
    Counter c(10);

    ++c;  // 전위 증가
    c++;  // 후위 증가

    std::cout << c.get() << std::endl;  // 12

    return 0;
}
```

### 7.6 함수 호출 연산자와 인덱스 연산자

```cpp
class Matrix {
private:
    int data[3][3];

public:
    Matrix() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                data[i][j] = 0;
            }
        }
    }

    // 함수 호출 연산자 ()
    int& operator()(int row, int col) {
        return data[row][col];
    }

    const int& operator()(int row, int col) const {
        return data[row][col];
    }
};

class Functor {
public:
    // 함수 객체
    int operator()(int a, int b) const {
        return a + b;
    }
};

class Array {
private:
    int *data;
    size_t size;

public:
    Array(size_t s) : size(s) {
        data = new int[size]();
    }

    ~Array() {
        delete[] data;
    }

    // [] 연산자
    int& operator[](size_t index) {
        return data[index];
    }

    const int& operator[](size_t index) const {
        return data[index];
    }
};

int main() {
    Matrix m;
    m(0, 0) = 1;
    m(1, 1) = 5;
    std::cout << m(0, 0) << std::endl;

    Functor add;
    std::cout << add(3, 4) << std::endl;  // 7

    Array arr(10);
    arr[0] = 100;
    std::cout << arr[0] << std::endl;

    return 0;
}
```

---

## 8. 상속

### 8.1 상속 기본

```cpp
// 기반 클래스 (부모)
class Animal {
protected:
    std::string name;
    int age;

public:
    Animal(const std::string &n, int a) : name(n), age(a) {}

    void eat() {
        std::cout << name << " is eating" << std::endl;
    }

    void sleep() {
        std::cout << name << " is sleeping" << std::endl;
    }
};

// 파생 클래스 (자식)
class Dog : public Animal {
private:
    std::string breed;

public:
    Dog(const std::string &n, int a, const std::string &b)
        : Animal(n, a), breed(b) {}

    void bark() {
        std::cout << name << " is barking" << std::endl;
    }
};

class Cat : public Animal {
public:
    Cat(const std::string &n, int a) : Animal(n, a) {}

    void meow() {
        std::cout << name << " is meowing" << std::endl;
    }
};

int main() {
    Dog dog("Buddy", 3, "Golden Retriever");
    dog.eat();   // 상속받은 메서드
    dog.bark();  // 고유 메서드

    Cat cat("Whiskers", 2);
    cat.sleep();
    cat.meow();

    return 0;
}
```

### 8.2 접근 제어와 상속

```cpp
class Base {
private:
    int private_member;
protected:
    int protected_member;
public:
    int public_member;
};

// public 상속 (가장 일반적)
class PublicDerived : public Base {
    // protected_member는 protected로 유지
    // public_member는 public으로 유지
};

// protected 상속
class ProtectedDerived : protected Base {
    // protected_member는 protected로 유지
    // public_member는 protected로 변경
};

// private 상속
class PrivateDerived : private Base {
    // protected_member는 private로 변경
    // public_member는 private로 변경
};

int main() {
    PublicDerived pd;
    pd.public_member = 10;  // OK

    ProtectedDerived prd;
    // prd.public_member = 10;  // 에러! protected로 변경됨

    PrivateDerived pvd;
    // pvd.public_member = 10;  // 에러! private로 변경됨

    return 0;
}
```

### 8.3 다중 상속

```cpp
class Printable {
public:
    virtual void print() const {
        std::cout << "Printable object" << std::endl;
    }
};

class Serializable {
public:
    virtual void serialize() const {
        std::cout << "Serializing..." << std::endl;
    }
};

// 다중 상속
class Document : public Printable, public Serializable {
private:
    std::string content;

public:
    Document(const std::string &c) : content(c) {}

    void print() const override {
        std::cout << "Document: " << content << std::endl;
    }

    void serialize() const override {
        std::cout << "Serializing document: " << content << std::endl;
    }
};

int main() {
    Document doc("Hello, World!");
    doc.print();
    doc.serialize();

    return 0;
}
```

### 8.4 다이아몬드 문제와 가상 상속

```cpp
// 다이아몬드 문제
class Animal {
public:
    void breathe() {
        std::cout << "Breathing" << std::endl;
    }
};

class Mammal : public Animal {};
class Bird : public Animal {};

// 문제: Fish는 Animal을 두 번 상속받음
class Platypus : public Mammal, public Bird {};

int main() {
    Platypus p;
    // p.breathe();  // 에러! 모호함 (Mammal::breathe vs Bird::breathe)
    p.Mammal::breathe();  // 명시적으로 지정해야 함

    return 0;
}

// 해결책: 가상 상속
class Animal2 {
public:
    void breathe() {
        std::cout << "Breathing" << std::endl;
    }
};

class Mammal2 : virtual public Animal2 {};
class Bird2 : virtual public Animal2 {};

class Platypus2 : public Mammal2, public Bird2 {};

int main() {
    Platypus2 p;
    p.breathe();  // OK! Animal2는 한 번만 상속됨

    return 0;
}
```

### 8.5 생성자와 소멸자 순서

```cpp
class Base {
public:
    Base() {
        std::cout << "Base constructor" << std::endl;
    }

    ~Base() {
        std::cout << "Base destructor" << std::endl;
    }
};

class Derived : public Base {
public:
    Derived() {
        std::cout << "Derived constructor" << std::endl;
    }

    ~Derived() {
        std::cout << "Derived destructor" << std::endl;
    }
};

int main() {
    {
        Derived d;
        // 출력:
        // Base constructor
        // Derived constructor
    }
    // 출력:
    // Derived destructor
    // Base destructor

    return 0;
}

// 생성: 부모 -> 자식
// 소멸: 자식 -> 부모
```

---

## 9. 다형성과 가상 함수

### 9.1 가상 함수

```cpp
class Shape {
public:
    virtual double area() const {  // 가상 함수
        return 0.0;
    }

    virtual void draw() const {
        std::cout << "Drawing a shape" << std::endl;
    }

    virtual ~Shape() {}  // 가상 소멸자 (중요!)
};

class Circle : public Shape {
private:
    double radius;

public:
    Circle(double r) : radius(r) {}

    double area() const override {  // override (C++11)
        return 3.14159 * radius * radius;
    }

    void draw() const override {
        std::cout << "Drawing a circle" << std::endl;
    }
};

class Rectangle : public Shape {
private:
    double width, height;

public:
    Rectangle(double w, double h) : width(w), height(h) {}

    double area() const override {
        return width * height;
    }

    void draw() const override {
        std::cout << "Drawing a rectangle" << std::endl;
    }
};

int main() {
    Shape *shapes[] = {
        new Circle(5.0),
        new Rectangle(4.0, 6.0)
    };

    for (Shape *shape : shapes) {
        shape->draw();  // 동적 바인딩
        std::cout << "Area: " << shape->area() << std::endl;
    }

    for (Shape *shape : shapes) {
        delete shape;  // 가상 소멸자 덕분에 올바른 소멸자 호출
    }

    return 0;
}
```

### 9.2 순수 가상 함수와 추상 클래스

```cpp
// 추상 클래스 (인터페이스)
class IDrawable {
public:
    virtual void draw() const = 0;  // 순수 가상 함수
    virtual ~IDrawable() {}
};

class IPrintable {
public:
    virtual void print() const = 0;
    virtual ~IPrintable() {}
};

// 구체 클래스
class Document : public IDrawable, public IPrintable {
private:
    std::string content;

public:
    Document(const std::string &c) : content(c) {}

    void draw() const override {
        std::cout << "Drawing: " << content << std::endl;
    }

    void print() const override {
        std::cout << "Printing: " << content << std::endl;
    }
};

int main() {
    // IDrawable drawable;  // 에러! 추상 클래스는 인스턴스화 불가

    Document doc("Hello");
    doc.draw();
    doc.print();

    IDrawable *drawable = &doc;
    drawable->draw();

    return 0;
}
```

### 9.3 final과 override (C++11)

```cpp
class Base {
public:
    virtual void func1() {}
    virtual void func2() final {}  // final: 더 이상 오버라이드 불가
};

class Derived : public Base {
public:
    void func1() override {}  // OK
    // void func2() override {}  // 에러! func2는 final
};

// 클래스 전체를 final로
class FinalClass final {
    // ...
};

// class CannotDerived : public FinalClass {};  // 에러! FinalClass는 final

// override의 장점: 타이포 방지
class Shape {
public:
    virtual void draw() const {}
};

class Circle : public Shape {
public:
    // void darw() const override {}  // 에러! draw의 타이포, override가 없으면 컴파일 됨
    void draw() const override {}  // OK
};
```

### 9.4 가상 함수 테이블 (vtable)

```cpp
class Base {
public:
    virtual void func1() { std::cout << "Base::func1" << std::endl; }
    virtual void func2() { std::cout << "Base::func2" << std::endl; }
};

class Derived : public Base {
public:
    void func1() override { std::cout << "Derived::func1" << std::endl; }
    // func2는 오버라이드하지 않음
};

/*
vtable 구조:

Base vtable:
[0] -> Base::func1
[1] -> Base::func2

Derived vtable:
[0] -> Derived::func1
[1] -> Base::func2

각 객체는 vptr(가상 함수 포인터)를 가지고 있어 vtable을 가리킴
*/

int main() {
    Base *ptr = new Derived();
    ptr->func1();  // Derived::func1 (동적 바인딩)
    ptr->func2();  // Base::func2 (Derived가 오버라이드 안함)

    delete ptr;

    return 0;
}
```

### 9.5 다형성 활용 패턴

```cpp
// 전략 패턴
class SortStrategy {
public:
    virtual void sort(int arr[], int size) = 0;
    virtual ~SortStrategy() {}
};

class BubbleSort : public SortStrategy {
public:
    void sort(int arr[], int size) override {
        std::cout << "Bubble sorting..." << std::endl;
        // 버블 정렬 구현
    }
};

class QuickSort : public SortStrategy {
public:
    void sort(int arr[], int size) override {
        std::cout << "Quick sorting..." << std::endl;
        // 퀵 정렬 구현
    }
};

class Sorter {
private:
    SortStrategy *strategy;

public:
    Sorter(SortStrategy *s) : strategy(s) {}

    void set_strategy(SortStrategy *s) {
        strategy = s;
    }

    void sort(int arr[], int size) {
        strategy->sort(arr, size);
    }
};

int main() {
    int arr[] = {5, 2, 8, 1, 9};

    BubbleSort bubble;
    QuickSort quick;

    Sorter sorter(&bubble);
    sorter.sort(arr, 5);

    sorter.set_strategy(&quick);
    sorter.sort(arr, 5);

    return 0;
}
```

---

## 10. 템플릿

### 10.1 함수 템플릿

```cpp
// 기본 함수 템플릿
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

// 사용
int i = max(10, 20);           // T = int
double d = max(3.14, 2.71);    // T = double
std::string s = max(std::string("abc"), std::string("xyz"));

// 명시적 타입 지정
auto result = max<double>(10, 20.5);

// 여러 타입 매개변수
template<typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}

// C++14: 반환 타입 추론
template<typename T, typename U>
auto multiply(T a, U b) {
    return a * b;
}

// 비타입 템플릿 매개변수
template<typename T, int N>
class Array {
    T data[N];
public:
    int size() const { return N; }
};

Array<int, 10> arr;
```

### 10.2 클래스 템플릿

```cpp
template<typename T>
class Stack {
private:
    std::vector<T> elements;

public:
    void push(const T& elem) {
        elements.push_back(elem);
    }

    void pop() {
        if (elements.empty()) {
            throw std::out_of_range("Stack<>::pop(): empty stack");
        }
        elements.pop_back();
    }

    T top() const {
        if (elements.empty()) {
            throw std::out_of_range("Stack<>::top(): empty stack");
        }
        return elements.back();
    }

    bool empty() const {
        return elements.empty();
    }
};

// 사용
Stack<int> intStack;
intStack.push(7);
std::cout << intStack.top() << std::endl;

Stack<std::string> stringStack;
stringStack.push("hello");
```

### 10.3 템플릿 특수화

```cpp
// 일반 템플릿
template<typename T>
class Printer {
public:
    void print(const T& value) {
        std::cout << value << std::endl;
    }
};

// 완전 특수화
template<>
class Printer<bool> {
public:
    void print(const bool& value) {
        std::cout << (value ? "true" : "false") << std::endl;
    }
};

// 부분 특수화 (포인터 타입)
template<typename T>
class Printer<T*> {
public:
    void print(T* const& ptr) {
        if (ptr) {
            std::cout << "Pointer to: " << *ptr << std::endl;
        } else {
            std::cout << "nullptr" << std::endl;
        }
    }
};
```

### 10.4 가변 인자 템플릿 (C++11)

```cpp
// 재귀 종료
void print() {
    std::cout << std::endl;
}

// 가변 인자 템플릿
template<typename T, typename... Args>
void print(T first, Args... args) {
    std::cout << first << " ";
    print(args...);  // 재귀 호출
}

// 사용
print(1, 2.5, "hello", 'c');

// sizeof... 연산자
template<typename... Args>
void count(Args... args) {
    std::cout << "Arguments: " << sizeof...(args) << std::endl;
}

// 폴드 표현식 (C++17)
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);  // 단항 오른쪽 폴드
}

std::cout << sum(1, 2, 3, 4, 5) << std::endl;  // 15
```

---

## 11. STL (표준 템플릿 라이브러리)

### 11.1 컨테이너

```cpp
#include <vector>
#include <list>
#include <deque>
#include <set>
#include <map>
#include <unordered_map>

// vector
std::vector<int> vec = {1, 2, 3, 4, 5};
vec.push_back(6);
vec.pop_back();
vec[0] = 10;
vec.at(1) = 20;

// list (양방향 연결 리스트)
std::list<int> lst = {1, 2, 3};
lst.push_front(0);
lst.push_back(4);

// deque
std::deque<int> deq;
deq.push_back(1);
deq.push_front(0);

// set (정렬된 집합)
std::set<int> s = {3, 1, 4, 1, 5};  // {1, 3, 4, 5}
s.insert(2);
s.erase(3);

// map (정렬된 키-값)
std::map<std::string, int> m;
m["apple"] = 1;
m["banana"] = 2;
m.insert({"cherry", 3});

// unordered_map (해시 테이블)
std::unordered_map<std::string, int> um;
um["key1"] = 100;
um["key2"] = 200;
```

### 11.2 반복자 (Iterators)

```cpp
std::vector<int> vec = {1, 2, 3, 4, 5};

// 순회
for (auto it = vec.begin(); it != vec.end(); ++it) {
    std::cout << *it << " ";
}

// 역순회
for (auto it = vec.rbegin(); it != vec.rend(); ++it) {
    std::cout << *it << " ";
}

// const 반복자
for (auto it = vec.cbegin(); it != vec.cend(); ++it) {
    // *it = 10;  // 에러
}

// 범위 기반 for (C++11)
for (const auto& elem : vec) {
    std::cout << elem << " ";
}

// 반복자 연산
auto it = vec.begin();
std::advance(it, 2);  // 2칸 전진
auto dist = std::distance(vec.begin(), vec.end());
```

### 11.3 알고리즘

```cpp
#include <algorithm>
#include <numeric>

std::vector<int> vec = {3, 1, 4, 1, 5, 9, 2, 6};

// 정렬
std::sort(vec.begin(), vec.end());
std::sort(vec.begin(), vec.end(), std::greater<int>());

// 검색
auto it = std::find(vec.begin(), vec.end(), 5);
if (it != vec.end()) {
    std::cout << "Found: " << *it << std::endl;
}

// 이진 검색 (정렬된 범위)
bool found = std::binary_search(vec.begin(), vec.end(), 5);

// 개수 세기
int count = std::count(vec.begin(), vec.end(), 1);

// 최소/최대
auto min_it = std::min_element(vec.begin(), vec.end());
auto max_it = std::max_element(vec.begin(), vec.end());

// 변환
std::vector<int> squared(vec.size());
std::transform(vec.begin(), vec.end(), squared.begin(),
               [](int x) { return x * x; });

// 필터링
std::vector<int> evens;
std::copy_if(vec.begin(), vec.end(), std::back_inserter(evens),
             [](int x) { return x % 2 == 0; });

// 축적
int sum = std::accumulate(vec.begin(), vec.end(), 0);
int product = std::accumulate(vec.begin(), vec.end(), 1,
                               std::multiplies<int>());

// 제거
vec.erase(std::remove(vec.begin(), vec.end(), 1), vec.end());

// 고유값만 유지
std::sort(vec.begin(), vec.end());
vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
```

### 11.4 함수 객체와 람다

```cpp
// 함수 객체 (Functor)
struct Multiply {
    int factor;
    Multiply(int f) : factor(f) {}
    int operator()(int x) const {
        return x * factor;
    }
};

std::vector<int> vec = {1, 2, 3, 4, 5};
std::transform(vec.begin(), vec.end(), vec.begin(), Multiply(2));

// 람다
std::transform(vec.begin(), vec.end(), vec.begin(),
               [](int x) { return x * 2; });

// 캡처
int factor = 3;
std::transform(vec.begin(), vec.end(), vec.begin(),
               [factor](int x) { return x * factor; });

// 가변 람다
int count = 0;
std::for_each(vec.begin(), vec.end(),
              [&count](int x) { count += x; });
```

---

## 12. 예외 처리

### 12.1 예외 기본

```cpp
#include <stdexcept>

void divide(int a, int b) {
    if (b == 0) {
        throw std::runtime_error("Division by zero");
    }
    std::cout << a / b << std::endl;
}

try {
    divide(10, 0);
} catch (const std::runtime_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
} catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
} catch (...) {
    std::cerr << "Unknown exception" << std::endl;
}
```

### 12.2 표준 예외 계층

```cpp
// std::exception (기반 클래스)
//   ├─ std::logic_error
//   │   ├─ std::invalid_argument
//   │   ├─ std::domain_error
//   │   ├─ std::length_error
//   │   └─ std::out_of_range
//   └─ std::runtime_error
//       ├─ std::range_error
//       ├─ std::overflow_error
//       └─ std::underflow_error

try {
    std::vector<int> vec(10);
    vec.at(20);  // std::out_of_range
} catch (const std::out_of_range& e) {
    std::cerr << e.what() << std::endl;
}
```

### 12.3 커스텀 예외

```cpp
class MyException : public std::exception {
private:
    std::string message;
public:
    MyException(const std::string& msg) : message(msg) {}

    const char* what() const noexcept override {
        return message.c_str();
    }
};

throw MyException("Something went wrong");
```

### 12.4 RAII와 예외 안전성

```cpp
class Resource {
public:
    Resource() { std::cout << "Acquired\n"; }
    ~Resource() { std::cout << "Released\n"; }
};

void process() {
    Resource r;  // RAII
    // 예외 발생해도 소멸자 자동 호출
    throw std::runtime_error("Error");
}

// 예외 안전성 보장
class SafeVector {
    int* data;
    size_t size;
public:
    void resize(size_t new_size) {
        int* new_data = new int[new_size];
        try {
            // 복사...
            delete[] data;
            data = new_data;
            size = new_size;
        } catch (...) {
            delete[] new_data;
            throw;
        }
    }
};
```

---

## 13. 네임스페이스

```cpp
namespace MyNamespace {
    int value = 10;
    void func() {
        std::cout << "MyNamespace::func()" << std::endl;
    }

    namespace Nested {
        void func() {
            std::cout << "Nested::func()" << std::endl;
        }
    }
}

// 사용
MyNamespace::func();
MyNamespace::Nested::func();

// using 선언
using MyNamespace::func;
func();

// using 지시자 (권장하지 않음)
using namespace MyNamespace;

// 별칭 (C++11)
namespace MN = MyNamespace;
MN::func();

// 인라인 네임스페이스 (C++11)
namespace MyLib {
    inline namespace v2 {
        void func() { std::cout << "v2\n"; }
    }
    namespace v1 {
        void func() { std::cout << "v1\n"; }
    }
}

MyLib::func();     // v2::func()
MyLib::v1::func(); // v1::func()
```

---

## 14. 형변환 연산자

```cpp
// static_cast (컴파일 타임 캐스팅)
double d = 3.14;
int i = static_cast<int>(d);

// const_cast (const 제거)
const int x = 10;
int* ptr = const_cast<int*>(&x);
*ptr = 20;  // 미정의 동작!

// reinterpret_cast (비트 패턴 재해석)
int num = 42;
void* void_ptr = reinterpret_cast<void*>(num);

// dynamic_cast (런타임 타입 검사)
class Base {
    virtual ~Base() {}
};
class Derived : public Base {};

Base* base = new Derived();
Derived* derived = dynamic_cast<Derived*>(base);
if (derived) {
    std::cout << "Cast successful" << std::endl;
}
```

---

## 15. 스마트 포인터

### 15.1 unique_ptr

```cpp
#include <memory>

// 독점 소유권
std::unique_ptr<int> ptr1(new int(42));
std::unique_ptr<int> ptr2 = std::make_unique<int>(42);  // C++14

// 이동만 가능, 복사 불가
std::unique_ptr<int> ptr3 = std::move(ptr2);
// ptr2는 이제 nullptr

// 배열
std::unique_ptr<int[]> arr(new int[10]);
arr[0] = 10;

// 커스텀 deleter
auto deleter = [](FILE* fp) {
    if (fp) fclose(fp);
};
std::unique_ptr<FILE, decltype(deleter)> file(fopen("file.txt", "r"), deleter);
```

### 15.2 shared_ptr

```cpp
// 공유 소유권 (참조 카운팅)
std::shared_ptr<int> ptr1 = std::make_shared<int>(42);
std::shared_ptr<int> ptr2 = ptr1;  // 참조 카운트 증가

std::cout << ptr1.use_count() << std::endl;  // 2

ptr2.reset();  // 참조 카운트 감소
std::cout << ptr1.use_count() << std::endl;  // 1

// 순환 참조 문제
class Node {
public:
    std::shared_ptr<Node> next;
    ~Node() { std::cout << "Node destroyed\n"; }
};

std::shared_ptr<Node> n1 = std::make_shared<Node>();
std::shared_ptr<Node> n2 = std::make_shared<Node>();
n1->next = n2;
n2->next = n1;  // 메모리 누수!
```

### 15.3 weak_ptr

```cpp
// 순환 참조 해결
class Node {
public:
    std::weak_ptr<Node> next;  // weak_ptr 사용
    ~Node() { std::cout << "Node destroyed\n"; }
};

std::shared_ptr<Node> n1 = std::make_shared<Node>();
std::shared_ptr<Node> n2 = std::make_shared<Node>();
n1->next = n2;
n2->next = n1;  // 이제 메모리 누수 없음

// weak_ptr 사용
std::weak_ptr<int> weak = ptr1;
if (auto shared = weak.lock()) {  // shared_ptr로 변환
    std::cout << *shared << std::endl;
} else {
    std::cout << "Object destroyed" << std::endl;
}
```

---

## 16. 이동 의미론과 rvalue 참조

### 16.1 이동 의미론

```cpp
class String {
    char* data;
    size_t size;
public:
    // 복사 생성자
    String(const String& other) {
        size = other.size;
        data = new char[size];
        std::copy(other.data, other.data + size, data);
        std::cout << "Copy constructor\n";
    }

    // 이동 생성자
    String(String&& other) noexcept {
        data = other.data;
        size = other.size;
        other.data = nullptr;
        other.size = 0;
        std::cout << "Move constructor\n";
    }

    // 복사 대입
    String& operator=(const String& other) {
        if (this != &other) {
            delete[] data;
            size = other.size;
            data = new char[size];
            std::copy(other.data, other.data + size, data);
        }
        return *this;
    }

    // 이동 대입
    String& operator=(String&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
};

String s1("hello");
String s2 = std::move(s1);  // 이동 생성자 호출
```

### 16.2 완벽한 전달 (Perfect Forwarding)

```cpp
template<typename T>
void wrapper(T&& arg) {
    func(std::forward<T>(arg));
}

// lvalue는 lvalue로, rvalue는 rvalue로 전달됨
```

---

## 17. 람다 표현식

```cpp
// 기본
auto lambda = []() {
    std::cout << "Hello" << std::endl;
};
lambda();

// 매개변수와 반환값
auto add = [](int a, int b) -> int {
    return a + b;
};

// 캡처
int x = 10;
auto f1 = [x]() { std::cout << x << std::endl; };      // 값 캡처
auto f2 = [&x]() { x++; };                             // 참조 캡처
auto f3 = [=]() { std::cout << x << std::endl; };      // 모든 값 캡처
auto f4 = [&]() { x++; };                              // 모든 참조 캡처
auto f5 = [x, &y]() { /* ... */ };                     // 혼합 캡처

// 초기화 캡처 (C++14)
auto f6 = [ptr = std::make_unique<int>(42)]() {
    std::cout << *ptr << std::endl;
};

// 제네릭 람다 (C++14)
auto generic = [](auto x, auto y) {
    return x + y;
};
```

---

## 18. 모던 C++ (C++11/14/17/20/23)

### 18.1 C++11 핵심 기능

```cpp
// auto
auto x = 42;
auto d = 3.14;
auto s = std::string("hello");

// decltype
int i = 0;
decltype(i) j = 1;

// 범위 기반 for
std::vector<int> vec = {1, 2, 3};
for (const auto& elem : vec) {
    std::cout << elem << std::endl;
}

// nullptr
int* ptr = nullptr;

// constexpr
constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}

// 초기화 리스트
std::vector<int> v = {1, 2, 3, 4, 5};

// 위임 생성자
class MyClass {
public:
    MyClass() : MyClass(0, 0) {}
    MyClass(int x) : MyClass(x, 0) {}
    MyClass(int x, int y) { /* 주 생성자 */ }
};
```

### 18.2 C++14

```cpp
// 제네릭 람다
auto lambda = [](auto x, auto y) { return x + y; };

// 람다 초기화 캡처
auto ptr = std::make_unique<int>(10);
auto lambda2 = [ptr = std::move(ptr)]() {
    std::cout << *ptr << std::endl;
};

// 반환 타입 추론
auto func(int x) {
    if (x > 0) return x;
    return -x;
}

// 이진 리터럴
int binary = 0b1010;

// 숫자 구분자
int million = 1'000'000;
```

### 18.3 C++17

```cpp
// 구조화된 바인딩
std::map<std::string, int> m = {{"a", 1}, {"b", 2}};
for (const auto& [key, value] : m) {
    std::cout << key << ": " << value << std::endl;
}

// if/switch 초기화
if (auto it = m.find("a"); it != m.end()) {
    std::cout << it->second << std::endl;
}

// std::optional
std::optional<int> maybe_int = find_value();
if (maybe_int) {
    std::cout << *maybe_int << std::endl;
}

// std::variant
std::variant<int, std::string> v = 42;
v = "hello";

// std::any
std::any a = 1;
a = 3.14;
a = std::string("hello");

// 폴드 표현식
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);
}
```

### 18.4 C++20

```cpp
// Concepts
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T>
T add(T a, T b) {
    return a + b;
}

// Ranges
std::vector<int> vec = {1, 2, 3, 4, 5};
auto result = vec | std::views::filter([](int x) { return x % 2 == 0; })
                  | std::views::transform([](int x) { return x * 2; });

// Coroutines
generator<int> fibonacci() {
    int a = 0, b = 1;
    while (true) {
        co_yield a;
        auto next = a + b;
        a = b;
        b = next;
    }
}

// Modules
import std.core;

// 삼원 비교 연산자 <=>
auto operator<=>(const Point&) const = default;
```

---

## 19. 동시성과 멀티스레딩

```cpp
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>

// 스레드 생성
void task() {
    std::cout << "Thread ID: " << std::this_thread::get_id() << std::endl;
}

std::thread t(task);
t.join();

// 뮤텍스
std::mutex mtx;
int shared_data = 0;

void increment() {
    std::lock_guard<std::mutex> lock(mtx);
    shared_data++;
}

// 조건 변수
std::condition_variable cv;
bool ready = false;

void wait_for_signal() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, []{ return ready; });
}

void send_signal() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        ready = true;
    }
    cv.notify_one();
}

// future/promise
std::promise<int> prom;
std::future<int> fut = prom.get_future();

std::thread([](std::promise<int> p) {
    p.set_value(42);
}, std::move(prom)).detach();

std::cout << fut.get() << std::endl;

// async
auto future = std::async(std::launch::async, []() {
    return 42;
});
std::cout << future.get() << std::endl;
```

---

## 20. 메타프로그래밍과 고급 기법

```cpp
// SFINAE
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
twice(T value) {
    return value * 2;
}

// constexpr if (C++17)
template<typename T>
auto getValue(T t) {
    if constexpr (std::is_pointer_v<T>) {
        return *t;
    } else {
        return t;
    }
}

// 타입 특성
template<typename T>
struct is_pointer : std::false_type {};

template<typename T>
struct is_pointer<T*> : std::true_type {};

// CRTP 패턴
template<typename Derived>
class Base {
public:
    void interface() {
        static_cast<Derived*>(this)->implementation();
    }
};

class Derived : public Base<Derived> {
public:
    void implementation() {
        std::cout << "Derived implementation\n";
    }
};
```

---

## 결론

C++은 강력하고 유연한 언어로, 이 가이드는 다음을 다룹니다:

1. **기초**: 입출력, 참조자, 함수 오버로딩
2. **OOP**: 클래스, 상속, 다형성
3. **템플릿**: 제네릭 프로그래밍, STL
4. **모던 C++**: 스마트 포인터, 이동 의미론, 람다
5. **고급**: 동시성, 메타프로그래밍

**학습 순서**: 1→5→7→9→10→11→15→16→17→18

계속 연습하며 모던 C++의 강력함을 경험하세요!
