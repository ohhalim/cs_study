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

**(계속됩니다...)**

이 C++ 가이드는 계속해서 다음 주제들을 다룹니다:
- 10. 템플릿
- 11. STL (표준 템플릿 라이브러리)
- 12. 예외 처리
- 13. 네임스페이스
- 14. 형변환 연산자
- 15. 스마트 포인터
- 16. 이동 의미론과 rvalue 참조
- 17. 람다 표현식
- 18. 모던 C++ (C++11/14/17/20/23)
- 19. 동시성과 멀티스레딩
- 20. 메타프로그래밍과 고급 기법

각 섹션은 실전 예제와 함께 심도 있게 다뤄집니다.
