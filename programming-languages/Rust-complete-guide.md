# Rust 완벽 학습 가이드

## 목차
1. [Rust 소개와 설정](#1-rust-소개와-설정)
2. [기본 문법과 변수](#2-기본-문법과-변수)
3. [데이터 타입](#3-데이터-타입)
4. [소유권 (Ownership)](#4-소유권-ownership)
5. [빌림과 참조 (Borrowing & References)](#5-빌림과-참조-borrowing--references)
6. [슬라이스 (Slices)](#6-슬라이스-slices)
7. [구조체 (Structs)](#7-구조체-structs)
8. [열거형과 패턴 매칭](#8-열거형과-패턴-매칭)
9. [모듈과 크레이트](#9-모듈과-크레이트)
10. [컬렉션](#10-컬렉션)
11. [에러 처리](#11-에러-처리)
12. [제네릭](#12-제네릭)
13. [트레이트 (Traits)](#13-트레이트-traits)
14. [라이프타임](#14-라이프타임)
15. [클로저](#15-클로저)
16. [반복자 (Iterators)](#16-반복자-iterators)
17. [스마트 포인터](#17-스마트-포인터)
18. [동시성 (Concurrency)](#18-동시성-concurrency)
19. [비동기 프로그래밍](#19-비동기-프로그래밍)
20. [고급 기능과 패턴](#20-고급-기능과-패턴)

---

## 1. Rust 소개와 설정

### 1.1 Rust란?
- 시스템 프로그래밍 언어
- 메모리 안전성 보장 (가비지 컬렉터 없이)
- 제로 코스트 추상화
- 동시성 안전성
- 소유권 시스템

### 1.2 설치

```bash
# rustup 설치 (공식 설치 도구)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 버전 확인
rustc --version
cargo --version

# 업데이트
rustup update

# 문서 보기
rustup doc
```

### 1.3 첫 프로그램

```rust
// main.rs
fn main() {
    println!("Hello, World!");
}

// 컴파일 및 실행
// rustc main.rs
// ./main
```

### 1.4 Cargo (빌드 시스템 및 패키지 관리자)

```bash
# 새 프로젝트 생성
cargo new hello_world
cd hello_world

# 빌드
cargo build

# 빌드 및 실행
cargo run

# 릴리스 빌드 (최적화)
cargo build --release

# 테스트
cargo test

# 문서 생성
cargo doc --open

# 의존성 업데이트
cargo update
```

---

## 2. 기본 문법과 변수

### 2.1 변수와 가변성

```rust
fn main() {
    // 불변 변수 (기본)
    let x = 5;
    // x = 6;  // 에러! 불변 변수는 재할당 불가

    // 가변 변수
    let mut y = 10;
    y = 20;  // OK
    println!("y = {}", y);

    // 상수 (타입 명시 필수, 대문자 관례)
    const MAX_POINTS: u32 = 100_000;

    // 섀도잉 (shadowing)
    let x = x + 1;  // 새로운 변수 x
    let x = x * 2;  // 또 다른 새로운 변수 x
    println!("x = {}", x);  // 12

    // 타입 변경 가능
    let spaces = "   ";
    let spaces = spaces.len();  // 타입 변경 (str -> usize)
}
```

### 2.2 주석과 출력

```rust
fn main() {
    // 한 줄 주석

    /*
     * 여러 줄 주석
     */

    /// 문서화 주석 (외부)
    /// 함수나 구조체 위에 사용

    //! 문서화 주석 (내부)
    //! 모듈이나 크레이트 설명

    // 출력
    println!("Hello!");                        // 매크로
    println!("x = {}", 42);                    // 포맷팅
    println!("x = {}, y = {}", 10, 20);
    println!("x = {x}, y = {y}", x=1, y=2);   // 이름 지정

    // 디버그 출력
    let point = (3, 4);
    println!("{:?}", point);    // (3, 4)
    println!("{:#?}", point);   // 예쁘게 출력

    // 입력
    use std::io;
    let mut input = String::new();
    io::stdin().read_line(&mut input)
        .expect("Failed to read line");
}
```

### 2.3 함수

```rust
// 기본 함수
fn greet() {
    println!("Hello!");
}

// 매개변수 (타입 필수)
fn add(x: i32, y: i32) {
    println!("x + y = {}", x + y);
}

// 반환값 (타입 명시 필수)
fn multiply(x: i32, y: i32) -> i32 {
    x * y  // 세미콜론 없음 = 표현식 (반환)
}

fn divide(x: i32, y: i32) -> i32 {
    return x / y;  // return 키워드 사용 가능
}

// 여러 값 반환 (튜플)
fn swap(x: i32, y: i32) -> (i32, i32) {
    (y, x)
}

fn main() {
    greet();
    add(5, 3);

    let result = multiply(4, 5);
    println!("result = {}", result);

    let (a, b) = swap(1, 2);
    println!("a = {}, b = {}", a, b);
}
```

### 2.4 표현식과 문장

```rust
fn main() {
    // 문장 (statement): 값을 반환하지 않음
    let x = 5;

    // 표현식 (expression): 값을 반환
    let y = {
        let x = 3;
        x + 1  // 세미콜론 없음
    };  // y = 4

    // if는 표현식
    let number = 5;
    let result = if number < 5 {
        "less"
    } else {
        "greater or equal"
    };

    println!("result = {}", result);
}
```

---

## 3. 데이터 타입

### 3.1 스칼라 타입

```rust
fn main() {
    // 정수형
    let a: i8 = 127;           // -128 ~ 127
    let b: u8 = 255;           // 0 ~ 255
    let c: i16 = 32_767;
    let d: u16 = 65_535;
    let e: i32 = 2_147_483_647;  // 기본값
    let f: u32 = 4_294_967_295;
    let g: i64 = 9_223_372_036_854_775_807;
    let h: u64 = 18_446_744_073_709_551_615;
    let i: i128;
    let j: u128;
    let k: isize;  // 아키텍처 의존적 (32/64비트)
    let l: usize;

    // 리터럴
    let decimal = 98_222;
    let hex = 0xff;
    let octal = 0o77;
    let binary = 0b1111_0000;
    let byte = b'A';  // u8만 가능

    // 부동소수점
    let f1: f32 = 3.14;
    let f2: f64 = 2.718;  // 기본값

    // 불린
    let t: bool = true;
    let f: bool = false;

    // 문자 (4바이트 유니코드)
    let c: char = 'z';
    let emoji: char = '😊';
}
```

### 3.2 복합 타입

```rust
fn main() {
    // 튜플 (고정 크기, 다양한 타입)
    let tup: (i32, f64, u8) = (500, 6.4, 1);

    // 구조 분해
    let (x, y, z) = tup;
    println!("x = {}", x);

    // 인덱스 접근
    let five_hundred = tup.0;
    let six_point_four = tup.1;

    // 빈 튜플 (unit)
    let unit: () = ();

    // 배열 (고정 크기, 동일 타입)
    let arr: [i32; 5] = [1, 2, 3, 4, 5];
    let first = arr[0];

    // 동일 값으로 초기화
    let zeros = [0; 10];  // [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    // 길이와 타입
    println!("length = {}", arr.len());
}
```

### 3.3 문자열

```rust
fn main() {
    // String (힙 할당, 가변)
    let mut s = String::from("hello");
    s.push_str(", world!");
    println!("{}", s);

    // &str (문자열 슬라이스, 불변)
    let slice: &str = "hello";

    // String -> &str
    let s2 = String::from("hello");
    let slice2: &str = &s2;

    // 연결
    let s1 = String::from("Hello, ");
    let s2 = String::from("world!");
    let s3 = s1 + &s2;  // s1은 이동됨
    // println!("{}", s1);  // 에러!

    // format! 매크로
    let s1 = String::from("tic");
    let s2 = String::from("tac");
    let s3 = String::from("toe");
    let s = format!("{}-{}-{}", s1, s2, s3);
    // s1, s2, s3는 여전히 유효

    // 바이트 접근
    for b in "नमस्ते".bytes() {
        println!("{}", b);
    }

    // 문자 접근
    for c in "नमस्ते".chars() {
        println!("{}", c);
    }
}
```

---

## 4. 소유권 (Ownership)

### 4.1 소유권 규칙

```rust
/*
1. Rust의 각 값은 소유자(owner)가 있다
2. 한 번에 하나의 소유자만 존재한다
3. 소유자가 스코프를 벗어나면 값이 버려진다 (dropped)
*/

fn main() {
    {
        let s = String::from("hello");  // s는 여기부터 유효
        // s 사용
    }  // s의 스코프 끝, 메모리 자동 해제 (drop 호출)

    // println!("{}", s);  // 에러! s는 스코프 밖
}
```

### 4.2 이동 (Move)

```rust
fn main() {
    // 스택 데이터 (Copy)
    let x = 5;
    let y = x;  // 복사 (Copy)
    println!("x = {}, y = {}", x, y);  // 둘 다 유효

    // 힙 데이터 (Move)
    let s1 = String::from("hello");
    let s2 = s1;  // 이동! s1은 더 이상 유효하지 않음
    // println!("{}", s1);  // 에러!
    println!("{}", s2);  // OK

    // 함수 호출 시 이동
    let s = String::from("hello");
    takes_ownership(s);
    // println!("{}", s);  // 에러! s가 이동됨

    let x = 5;
    makes_copy(x);
    println!("{}", x);  // OK (i32는 Copy)
}

fn takes_ownership(some_string: String) {
    println!("{}", some_string);
}  // some_string이 drop됨

fn makes_copy(some_integer: i32) {
    println!("{}", some_integer);
}
```

### 4.3 클론 (Clone)

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1.clone();  // 깊은 복사
    println!("s1 = {}, s2 = {}", s1, s2);  // 둘 다 유효

    // Copy 트레이트
    // 스택에만 저장되는 타입들
    // i32, u32, f64, bool, char, 튜플 (Copy 타입만 포함)
}
```

### 4.4 소유권과 함수

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = takes_and_gives_back(s1);
    // s1은 무효, s2는 유효

    let s3 = String::from("hello");
    let (s4, len) = calculate_length(s3);
    println!("'{}' has length {}", s4, len);
}

fn takes_and_gives_back(a_string: String) -> String {
    a_string  // 소유권 반환
}

fn calculate_length(s: String) -> (String, usize) {
    let length = s.len();
    (s, length)
}
```

---

## 5. 빌림과 참조 (Borrowing & References)

### 5.1 참조 (References)

```rust
fn main() {
    let s1 = String::from("hello");
    let len = calculate_length(&s1);  // 참조 전달 (빌림)
    println!("'{}' has length {}", s1, len);  // s1 여전히 유효
}

fn calculate_length(s: &String) -> usize {  // 참조 매개변수
    s.len()
}  // s는 소유권이 없으므로 drop되지 않음

// 참조를 만드는 것 = 빌림 (borrowing)
```

### 5.2 가변 참조

```rust
fn main() {
    let mut s = String::from("hello");
    change(&mut s);  // 가변 참조
    println!("{}", s);  // "hello, world"
}

fn change(some_string: &mut String) {
    some_string.push_str(", world");
}

// 제약사항
fn restrictions() {
    let mut s = String::from("hello");

    let r1 = &mut s;
    // let r2 = &mut s;  // 에러! 동시에 여러 가변 참조 불가

    println!("{}", r1);

    // r1 사용 후에는 새 가변 참조 가능
    let r2 = &mut s;
    println!("{}", r2);
}

fn mixed_references() {
    let mut s = String::from("hello");

    let r1 = &s;  // 불변 참조
    let r2 = &s;  // 불변 참조 (여러 개 OK)
    println!("{} and {}", r1, r2);

    // let r3 = &mut s;  // 에러! 불변 참조와 가변 참조 동시 불가
}
```

### 5.3 댕글링 참조 방지

```rust
// 에러! 댕글링 참조
// fn dangle() -> &String {
//     let s = String::from("hello");
//     &s  // s의 참조를 반환하지만 s는 drop됨
// }

// 해결: 소유권 이동
fn no_dangle() -> String {
    let s = String::from("hello");
    s  // 소유권 이동
}
```

### 5.4 빌림 규칙 요약

```rust
/*
1. 어느 시점에서든 다음 중 하나만 가능:
   - 하나의 가변 참조
   - 여러 개의 불변 참조

2. 참조는 항상 유효해야 함 (댕글링 참조 불가)
*/

fn main() {
    let mut s = String::from("hello");

    {
        let r1 = &mut s;
    }  // r1 스코프 종료

    let r2 = &mut s;  // OK
}
```

---

## 6. 슬라이스 (Slices)

### 6.1 문자열 슬라이스

```rust
fn main() {
    let s = String::from("hello world");

    let hello = &s[0..5];   // "hello"
    let world = &s[6..11];  // "world"

    // 단축 문법
    let slice = &s[0..2];  // "he"
    let slice = &s[..2];   // 동일

    let len = s.len();
    let slice = &s[3..len];  // "lo world"
    let slice = &s[3..];     // 동일

    let slice = &s[0..len];  // "hello world"
    let slice = &s[..];      // 동일

    // 문자열 리터럴은 슬라이스
    let s: &str = "Hello, world!";

    // 예제: 첫 단어 찾기
    let mut s = String::from("hello world");
    let word = first_word(&s);
    // s.clear();  // 에러! s가 빌려진 상태
    println!("first word: {}", word);
}

fn first_word(s: &String) -> &str {
    let bytes = s.as_bytes();

    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }

    &s[..]
}

// 더 나은 버전
fn first_word_improved(s: &str) -> &str {  // &str로 받으면 더 유연
    let bytes = s.as_bytes();

    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }

    &s[..]
}
```

### 6.2 배열 슬라이스

```rust
fn main() {
    let a = [1, 2, 3, 4, 5];

    let slice = &a[1..3];  // [2, 3]
    assert_eq!(slice, &[2, 3]);

    // 타입: &[i32]
}
```

---

## 7. 구조체 (Structs)

### 7.1 구조체 정의와 인스턴스

```rust
struct User {
    username: String,
    email: String,
    sign_in_count: u64,
    active: bool,
}

fn main() {
    // 인스턴스 생성
    let user1 = User {
        email: String::from("someone@example.com"),
        username: String::from("someusername123"),
        active: true,
        sign_in_count: 1,
    };

    // 필드 접근
    println!("{}", user1.email);

    // 가변 인스턴스
    let mut user2 = User {
        email: String::from("another@example.com"),
        username: String::from("anotherusername456"),
        active: true,
        sign_in_count: 1,
    };

    user2.email = String::from("new@example.com");

    // 구조체 업데이트 문법
    let user3 = User {
        email: String::from("third@example.com"),
        username: String::from("thirdusername789"),
        ..user1  // 나머지 필드는 user1에서 가져옴
    };
}

// 빌더 함수
fn build_user(email: String, username: String) -> User {
    User {
        email,     // 필드 초기화 단축 문법
        username,  // 변수명과 필드명이 같으면 생략 가능
        active: true,
        sign_in_count: 1,
    }
}
```

### 7.2 튜플 구조체

```rust
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

fn main() {
    let black = Color(0, 0, 0);
    let origin = Point(0, 0, 0);

    // 인덱스로 접근
    println!("R: {}", black.0);

    // Color와 Point는 다른 타입
}
```

### 7.3 유닛 구조체

```rust
struct AlwaysEqual;  // 필드 없음

fn main() {
    let subject = AlwaysEqual;
}
```

### 7.4 메서드

```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    // 메서드
    fn area(&self) -> u32 {
        self.width * self.height
    }

    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }

    // 가변 메서드
    fn expand(&mut self, amount: u32) {
        self.width += amount;
        self.height += amount;
    }

    // 소유권을 가져가는 메서드 (드물게 사용)
    fn consume(self) -> u32 {
        self.width * self.height
    }

    // 연관 함수 (생성자로 자주 사용)
    fn new(width: u32, height: u32) -> Rectangle {
        Rectangle { width, height }
    }

    fn square(size: u32) -> Rectangle {
        Rectangle {
            width: size,
            height: size,
        }
    }
}

// 여러 impl 블록 가능
impl Rectangle {
    fn perimeter(&self) -> u32 {
        2 * (self.width + self.height)
    }
}

fn main() {
    let rect = Rectangle::new(30, 50);

    println!("Area: {}", rect.area());
    println!("Perimeter: {}", rect.perimeter());

    let rect2 = Rectangle::square(20);
    println!("Can hold: {}", rect.can_hold(&rect2));

    println!("Rectangle: {:?}", rect);
}
```

---

## 8. 열거형과 패턴 매칭

### 8.1 열거형 (Enum)

```rust
enum IpAddrKind {
    V4,
    V6,
}

enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String),
}

enum Message {
    Quit,
    Move { x: i32, y: i32 },  // 구조체처럼
    Write(String),
    ChangeColor(i32, i32, i32),
}

impl Message {
    fn call(&self) {
        match self {
            Message::Quit => println!("Quit"),
            Message::Move { x, y } => println!("Move to ({}, {})", x, y),
            Message::Write(s) => println!("Write: {}", s),
            Message::ChangeColor(r, g, b) => {
                println!("Change color to ({}, {}, {})", r, g, b)
            }
        }
    }
}

fn main() {
    let four = IpAddrKind::V4;
    let six = IpAddrKind::V6;

    let home = IpAddr::V4(127, 0, 0, 1);
    let loopback = IpAddr::V6(String::from("::1"));

    let msg = Message::Write(String::from("hello"));
    msg.call();
}
```

### 8.2 Option

```rust
fn main() {
    // Option<T> - null 대체
    let some_number: Option<i32> = Some(5);
    let some_string: Option<&str> = Some("a string");
    let absent_number: Option<i32> = None;

    // Option은 T와 다른 타입
    let x: i8 = 5;
    let y: Option<i8> = Some(5);
    // let sum = x + y;  // 에러! i8 + Option<i8> 불가

    // Option 사용
    if let Some(value) = some_number {
        println!("Value: {}", value);
    }

    // unwrap (값이 있으면 반환, 없으면 패닉)
    let x = Some(10);
    println!("{}", x.unwrap());

    // expect (패닉 시 메시지 지정)
    let x: Option<i32> = None;
    // x.expect("No value!");  // 패닉!

    // unwrap_or (값이 없으면 기본값)
    let x: Option<i32> = None;
    println!("{}", x.unwrap_or(0));

    // map
    let x = Some(5);
    let y = x.map(|n| n * 2);  // Some(10)

    // and_then
    let x = Some(5);
    let y = x.and_then(|n| Some(n * 2));  // Some(10)
}
```

### 8.3 match

```rust
enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter(UsState),
}

#[derive(Debug)]
enum UsState {
    Alabama,
    Alaska,
    // ...
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => {
            println!("Lucky penny!");
            1
        }
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter(state) => {
            println!("State quarter from {:?}!", state);
            25
        }
    }
}

fn plus_one(x: Option<i32>) -> Option<i32> {
    match x {
        None => None,
        Some(i) => Some(i + 1),
    }
}

fn main() {
    let coin = Coin::Quarter(UsState::Alaska);
    println!("Value: {}", value_in_cents(coin));

    let five = Some(5);
    let six = plus_one(five);
    let none = plus_one(None);

    // _ 패턴 (기타)
    let some_value = 7u8;
    match some_value {
        1 => println!("one"),
        3 => println!("three"),
        5 => println!("five"),
        7 => println!("seven"),
        _ => (),  // 나머지
    }
}
```

### 8.4 if let

```rust
fn main() {
    let some_value = Some(3);

    // match 사용
    match some_value {
        Some(3) => println!("three"),
        _ => (),
    }

    // if let 사용 (더 간결)
    if let Some(3) = some_value {
        println!("three");
    }

    // else 추가 가능
    let coin = Coin::Penny;
    let mut count = 0;
    if let Coin::Quarter(state) = coin {
        println!("State quarter from {:?}!", state);
    } else {
        count += 1;
    }
}
```

### 8.5 while let

```rust
fn main() {
    let mut stack = Vec::new();
    stack.push(1);
    stack.push(2);
    stack.push(3);

    while let Some(top) = stack.pop() {
        println!("{}", top);
    }
}
```

---

## 9. 모듈과 크레이트

### 9.1 모듈 기본

```rust
// src/lib.rs 또는 src/main.rs
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}

        fn seat_at_table() {}  // private
    }

    mod serving {
        fn take_order() {}
        fn serve_order() {}
        fn take_payment() {}
    }
}

pub fn eat_at_restaurant() {
    // 절대 경로
    crate::front_of_house::hosting::add_to_waitlist();

    // 상대 경로
    front_of_house::hosting::add_to_waitlist();
}

// super 사용
fn serve_order() {}

mod back_of_house {
    fn fix_incorrect_order() {
        cook_order();
        super::serve_order();  // 부모 모듈의 함수
    }

    fn cook_order() {}
}
```

### 9.2 pub 사용

```rust
mod back_of_house {
    pub struct Breakfast {
        pub toast: String,
        seasonal_fruit: String,  // private
    }

    impl Breakfast {
        pub fn summer(toast: &str) -> Breakfast {
            Breakfast {
                toast: String::from(toast),
                seasonal_fruit: String::from("peaches"),
            }
        }
    }

    pub enum Appetizer {
        Soup,    // 자동으로 public
        Salad,   // 자동으로 public
    }
}

pub fn eat_at_restaurant() {
    let mut meal = back_of_house::Breakfast::summer("Rye");
    meal.toast = String::from("Wheat");
    // meal.seasonal_fruit = String::from("blueberries");  // 에러!

    let order1 = back_of_house::Appetizer::Soup;
    let order2 = back_of_house::Appetizer::Salad;
}
```

### 9.3 use 키워드

```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

// use로 가져오기
use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}

// 함수까지 가져오기 (비권장)
use crate::front_of_house::hosting::add_to_waitlist;

pub fn eat() {
    add_to_waitlist();
}

// as로 이름 변경
use std::fmt::Result;
use std::io::Result as IoResult;

// pub use (재수출)
pub use crate::front_of_house::hosting;

// 중첩 경로
use std::io::{self, Write};
use std::collections::{HashMap, BTreeMap, HashSet};

// glob
use std::collections::*;
```

### 9.4 파일로 모듈 분리

```rust
// src/lib.rs
mod front_of_house;  // src/front_of_house.rs를 찾음

pub use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}

// src/front_of_house.rs
pub mod hosting {
    pub fn add_to_waitlist() {}
}

// 또는 src/front_of_house/hosting.rs
// src/front_of_house/mod.rs
pub mod hosting;
```

---

## 10. 컬렉션

### 10.1 벡터 (Vector)

```rust
fn main() {
    // 벡터 생성
    let v: Vec<i32> = Vec::new();

    // vec! 매크로
    let v = vec![1, 2, 3];

    // 추가
    let mut v = Vec::new();
    v.push(5);
    v.push(6);
    v.push(7);

    // 읽기
    let third: &i32 = &v[2];
    println!("Third element: {}", third);

    match v.get(2) {
        Some(third) => println!("Third element: {}", third),
        None => println!("No third element"),
    }

    // 범위를 벗어나면
    // let does_not_exist = &v[100];  // 패닉!
    let does_not_exist = v.get(100);  // None

    // 반복
    let v = vec![100, 32, 57];
    for i in &v {
        println!("{}", i);
    }

    // 가변 반복
    let mut v = vec![100, 32, 57];
    for i in &mut v {
        *i += 50;
    }

    // 다양한 타입 저장 (enum 사용)
    enum SpreadsheetCell {
        Int(i32),
        Float(f64),
        Text(String),
    }

    let row = vec![
        SpreadsheetCell::Int(3),
        SpreadsheetCell::Text(String::from("blue")),
        SpreadsheetCell::Float(10.12),
    ];
}
```

### 10.2 문자열 (String)

```rust
fn main() {
    // 생성
    let mut s = String::new();
    let s = "initial contents".to_string();
    let s = String::from("initial contents");

    // 추가
    let mut s = String::from("foo");
    s.push_str("bar");  // "foobar"
    s.push('!');        // "foobar!"

    // 연결
    let s1 = String::from("Hello, ");
    let s2 = String::from("world!");
    let s3 = s1 + &s2;  // s1은 이동됨
    // println!("{}", s1);  // 에러!

    // format! 매크로
    let s1 = String::from("tic");
    let s2 = String::from("tac");
    let s3 = String::from("toe");
    let s = format!("{}-{}-{}", s1, s2, s3);
    // s1, s2, s3 모두 유효

    // 인덱싱 불가
    let s1 = String::from("hello");
    // let h = s1[0];  // 에러!

    // 슬라이싱 (바이트 단위, 주의 필요)
    let hello = "Здравствуйте";
    let s = &hello[0..4];  // "Зд" (각 2바이트)

    // 반복
    for c in "नमस्ते".chars() {
        println!("{}", c);
    }

    for b in "नमस्ते".bytes() {
        println!("{}", b);
    }
}
```

### 10.3 해시맵 (HashMap)

```rust
use std::collections::HashMap;

fn main() {
    // 생성
    let mut scores = HashMap::new();
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Yellow"), 50);

    // collect로 생성
    let teams = vec![String::from("Blue"), String::from("Yellow")];
    let initial_scores = vec![10, 50];
    let scores: HashMap<_, _> = teams.iter()
        .zip(initial_scores.iter())
        .collect();

    // 읽기
    let team_name = String::from("Blue");
    let score = scores.get(&team_name);  // Option<&i32>

    match score {
        Some(&s) => println!("Score: {}", s),
        None => println!("Team not found"),
    }

    // 반복
    for (key, value) in &scores {
        println!("{}: {}", key, value);
    }

    // 소유권
    let field_name = String::from("Favorite color");
    let field_value = String::from("Blue");
    let mut map = HashMap::new();
    map.insert(field_name, field_value);
    // field_name과 field_value는 이동됨

    // 덮어쓰기
    scores.insert(String::from("Blue"), 25);

    // 키가 없을 때만 삽입
    scores.entry(String::from("Blue")).or_insert(50);
    scores.entry(String::from("Red")).or_insert(50);

    // 기존 값 기반 업데이트
    let text = "hello world wonderful world";
    let mut map = HashMap::new();

    for word in text.split_whitespace() {
        let count = map.entry(word).or_insert(0);
        *count += 1;
    }

    println!("{:?}", map);  // {"hello": 1, "world": 2, "wonderful": 1}
}
```

---

## 11. 에러 처리

### 11.1 Result와 Option

```rust
// Result<T, E>
fn divide(a: i32, b: i32) -> Result<i32, String> {
    if b == 0 {
        Err(String::from("division by zero"))
    } else {
        Ok(a / b)
    }
}

// 사용
match divide(10, 2) {
    Ok(result) => println!("Result: {}", result),
    Err(e) => println!("Error: {}", e),
}

// unwrap (패닉 가능)
let result = divide(10, 2).unwrap();

// expect (커스텀 메시지)
let result = divide(10, 2).expect("Division failed");

// unwrap_or (기본값)
let result = divide(10, 0).unwrap_or(0);

// ? 연산자
fn process() -> Result<i32, String> {
    let a = divide(10, 2)?;  // 에러면 조기 반환
    let b = divide(20, 4)?;
    Ok(a + b)
}
```

### 11.2 에러 전파

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_file(path: &str) -> Result<String, io::Error> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

// 체이닝
fn process_file(path: &str) -> Result<usize, io::Error> {
    Ok(read_file(path)?.len())
}
```

### 11.3 커스텀 에러 타입

```rust
use std::fmt;

#[derive(Debug)]
enum MyError {
    IoError(std::io::Error),
    ParseError(std::num::ParseIntError),
    Custom(String),
}

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MyError::IoError(e) => write!(f, "IO error: {}", e),
            MyError::ParseError(e) => write!(f, "Parse error: {}", e),
            MyError::Custom(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl From<std::io::Error> for MyError {
    fn from(error: std::io::Error) -> Self {
        MyError::IoError(error)
    }
}
```

---

## 12. 제네릭

```rust
// 제네릭 함수
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in list {
        if item > largest {
            largest = item;
        }
    }
    largest
}

// 제네릭 구조체
struct Point<T> {
    x: T,
    y: T,
}

impl<T> Point<T> {
    fn new(x: T, y: T) -> Point<T> {
        Point { x, y }
    }
}

// 특정 타입에 대한 구현
impl Point<f32> {
    fn distance_from_origin(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}

// 여러 타입 매개변수
struct Pair<T, U> {
    first: T,
    second: U,
}
```

---

## 13. 트레이트 (Traits)

```rust
// 트레이트 정의
trait Summary {
    fn summarize(&self) -> String;

    // 기본 구현
    fn default_summary(&self) -> String {
        String::from("(Read more...)")
    }
}

// 트레이트 구현
struct Article {
    headline: String,
    content: String,
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{}: {}", self.headline, self.content)
    }
}

// 트레이트 바운드
fn notify<T: Summary>(item: &T) {
    println!("{}", item.summarize());
}

// 여러 트레이트
fn notify2<T: Summary + Display>(item: &T) {
    // ...
}

// where 절
fn some_function<T, U>(t: &T, u: &U) -> i32
where
    T: Display + Clone,
    U: Clone + Debug,
{
    // ...
}

// 트레이트 반환
fn returns_summarizable() -> impl Summary {
    Article {
        headline: String::from("Title"),
        content: String::from("Content"),
    }
}
```

---

## 14. 라이프타임

```rust
// 라이프타임 명시
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

// 구조체의 라이프타임
struct ImportantExcerpt<'a> {
    part: &'a str,
}

impl<'a> ImportantExcerpt<'a> {
    fn level(&self) -> i32 {
        3
    }

    fn announce_and_return_part(&self, announcement: &str) -> &str {
        println!("Attention: {}", announcement);
        self.part
    }
}

// 라이프타임 생략 규칙
// 1. 각 참조 매개변수는 자신만의 라이프타임을 가짐
// 2. 참조 매개변수가 하나면 반환값도 같은 라이프타임
// 3. 메서드에서 &self가 있으면 반환값도 같은 라이프타임

// 정적 라이프타임
let s: &'static str = "I have a static lifetime.";
```

---

## 15. 클로저

```rust
// 기본 클로저
let add_one = |x| x + 1;
println!("{}", add_one(5));  // 6

// 타입 명시
let add = |x: i32, y: i32| -> i32 { x + y };

// 환경 캡처
let x = 4;
let equal_to_x = |z| z == x;
println!("{}", equal_to_x(4));  // true

// 이동 캡처
let x = vec![1, 2, 3];
let equal_to_x = move |z| z == x;
// x는 더 이상 사용 불가

// 함수 인자로 클로저
fn apply<F>(f: F, x: i32) -> i32
where
    F: Fn(i32) -> i32,
{
    f(x)
}

let result = apply(|x| x * 2, 5);  // 10
```

---

## 16. 반복자 (Iterators)

```rust
// 반복자 생성
let v = vec![1, 2, 3];
let mut iter = v.iter();

assert_eq!(iter.next(), Some(&1));
assert_eq!(iter.next(), Some(&2));
assert_eq!(iter.next(), Some(&3));
assert_eq!(iter.next(), None);

// for 루프와 반복자
for val in &v {
    println!("{}", val);
}

// 반복자 어댑터
let v: Vec<i32> = vec![1, 2, 3];
let v2: Vec<_> = v.iter().map(|x| x + 1).collect();

// 필터링
let evens: Vec<_> = v.iter().filter(|x| *x % 2 == 0).collect();

// 체이닝
let result: i32 = v.iter()
    .filter(|x| *x % 2 == 0)
    .map(|x| x * 2)
    .sum();

// 커스텀 반복자
struct Counter {
    count: u32,
}

impl Counter {
    fn new() -> Counter {
        Counter { count: 0 }
    }
}

impl Iterator for Counter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count < 5 {
            self.count += 1;
            Some(self.count)
        } else {
            None
        }
    }
}
```

---

## 17. 스마트 포인터

### 17.1 Box<T>

```rust
// 힙 할당
let b = Box::new(5);
println!("{}", b);

// 재귀 타입
enum List {
    Cons(i32, Box<List>),
    Nil,
}

use List::{Cons, Nil};
let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
```

### 17.2 Rc<T> (Reference Counted)

```rust
use std::rc::Rc;

let a = Rc::new(5);
let b = Rc::clone(&a);
let c = Rc::clone(&a);

println!("count: {}", Rc::strong_count(&a));  // 3
```

### 17.3 RefCell<T>

```rust
use std::cell::RefCell;

let x = RefCell::new(5);
*x.borrow_mut() += 1;
println!("{}", x.borrow());  // 6

// 내부 가변성 패턴
pub trait Messenger {
    fn send(&self, msg: &str);
}

struct MockMessenger {
    sent_messages: RefCell<Vec<String>>,
}

impl Messenger for MockMessenger {
    fn send(&self, msg: &str) {
        self.sent_messages.borrow_mut().push(String::from(msg));
    }
}
```

---

## 18. 동시성 (Concurrency)

```rust
use std::thread;
use std::time::Duration;

// 스레드 생성
let handle = thread::spawn(|| {
    for i in 1..10 {
        println!("spawned thread: {}", i);
        thread::sleep(Duration::from_millis(1));
    }
});

handle.join().unwrap();

// 이동 캡처
let v = vec![1, 2, 3];
let handle = thread::spawn(move || {
    println!("{:?}", v);
});

// 채널
use std::sync::mpsc;

let (tx, rx) = mpsc::channel();

thread::spawn(move || {
    tx.send(String::from("hi")).unwrap();
});

let received = rx.recv().unwrap();
println!("{}", received);

// Mutex
use std::sync::Mutex;

let m = Mutex::new(5);
{
    let mut num = m.lock().unwrap();
    *num = 6;
}
println!("{:?}", m);

// Arc (Atomic Reference Counting)
use std::sync::Arc;

let counter = Arc::new(Mutex::new(0));
let mut handles = vec![];

for _ in 0..10 {
    let counter = Arc::clone(&counter);
    let handle = thread::spawn(move || {
        let mut num = counter.lock().unwrap();
        *num += 1;
    });
    handles.push(handle);
}

for handle in handles {
    handle.join().unwrap();
}

println!("Result: {}", *counter.lock().unwrap());
```

---

## 19. 비동기 프로그래밍

### 19.1 Async/Await 기초

```rust
use tokio;

// async 함수
async fn say_hello() {
    println!("Hello, async world!");
}

// await
async fn fetch_data() -> Result<String, reqwest::Error> {
    let response = reqwest::get("https://api.example.com/data").await?;
    let body = response.text().await?;
    Ok(body)
}

// tokio 런타임
#[tokio::main]
async fn main() {
    say_hello().await;

    let data = fetch_data().await;
    match data {
        Ok(d) => println!("{}", d),
        Err(e) => eprintln!("Error: {}", e),
    }
}

// 여러 태스크 동시 실행
use tokio::join;

async fn task1() -> i32 {
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    42
}

async fn task2() -> String {
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    "done".to_string()
}

#[tokio::main]
async fn main() {
    let (result1, result2) = join!(task1(), task2());
    println!("{} {}", result1, result2);
}
```

### 19.2 Future 트레이트 깊이 이해

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

// Future의 실제 정의
pub trait Future {
    type Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}

// 커스텀 Future 구현
struct TimerFuture {
    start: std::time::Instant,
    duration: std::time::Duration,
}

impl Future for TimerFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.start.elapsed() >= self.duration {
            Poll::Ready(())
        } else {
            // Waker를 저장해서 나중에 깨울 수 있도록
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

// async 함수는 Future를 반환
async fn example() -> i32 {
    42
}

// 위 코드는 실제로 이렇게 변환됨:
fn example() -> impl Future<Output = i32> {
    async move { 42 }
}
```

### 19.3 Pinning (고정) - 가장 어려운 개념

```rust
use std::pin::Pin;
use std::marker::PhantomPinned;

// Pin이 필요한 이유: 자기 참조 구조체
struct SelfReferential {
    data: String,
    pointer: *const String,  // data를 가리킴
    _pin: PhantomPinned,
}

impl SelfReferential {
    fn new(text: String) -> Pin<Box<Self>> {
        let mut boxed = Box::pin(SelfReferential {
            data: text,
            pointer: std::ptr::null(),
            _pin: PhantomPinned,
        });

        // 안전하지 않지만 필요한 작업
        unsafe {
            let ptr: *const String = &boxed.data;
            let mut_ref: Pin<&mut Self> = Pin::as_mut(&mut boxed);
            Pin::get_unchecked_mut(mut_ref).pointer = ptr;
        }

        boxed
    }

    fn get_data(self: Pin<&Self>) -> &str {
        &self.data
    }

    fn get_pointer_data(self: Pin<&Self>) -> &str {
        unsafe { &*self.pointer }
    }
}

// Pin의 보장: 메모리 위치가 고정됨
// 이동하면 포인터가 무효화되므로 Pin으로 방지
```

### 19.4 실전 Async 에러 처리

```rust
use tokio;
use std::error::Error;

// async + Result 조합
async fn fetch_user(id: u64) -> Result<User, Box<dyn Error>> {
    let url = format!("https://api.example.com/users/{}", id);
    let response = reqwest::get(&url).await?;

    if !response.status().is_success() {
        return Err(format!("HTTP error: {}", response.status()).into());
    }

    let user: User = response.json().await?;
    Ok(user)
}

// 여러 async 작업의 에러 처리
async fn process_users() -> Result<(), Box<dyn Error>> {
    let user1 = fetch_user(1).await?;  // ? 연산자 사용
    let user2 = fetch_user(2).await?;

    println!("Users: {:?} {:?}", user1, user2);
    Ok(())
}

// 병렬 처리 + 에러 처리
use tokio::try_join;

async fn parallel_fetch() -> Result<(User, User), Box<dyn Error>> {
    // 둘 중 하나라도 실패하면 전체 실패
    let (user1, user2) = try_join!(
        fetch_user(1),
        fetch_user(2)
    )?;

    Ok((user1, user2))
}

// select! 매크로 - 먼저 완료되는 것 선택
use tokio::select;

async fn race_condition() {
    let result = select! {
        res1 = fetch_user(1) => res1,
        res2 = fetch_user(2) => res2,
        _ = tokio::time::sleep(tokio::time::Duration::from_secs(5)) => {
            Err("Timeout".into())
        }
    };
}
```

### 19.5 Async 채널과 동시성

```rust
use tokio::sync::{mpsc, oneshot};

// Multiple Producer, Single Consumer
async fn mpsc_example() {
    let (tx, mut rx) = mpsc::channel(100);

    // 생산자 여러 개
    for i in 0..10 {
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            tx_clone.send(i).await.unwrap();
        });
    }
    drop(tx);  // 원본 송신자 닫기

    // 소비자
    while let Some(value) = rx.recv().await {
        println!("Received: {}", value);
    }
}

// One-shot 채널
async fn oneshot_example() {
    let (tx, rx) = oneshot::channel();

    tokio::spawn(async move {
        // 계산 수행
        let result = expensive_computation().await;
        tx.send(result).unwrap();
    });

    // 결과 대기
    match rx.await {
        Ok(result) => println!("Got: {}", result),
        Err(_) => println!("Sender dropped"),
    }
}

// Broadcast 채널
use tokio::sync::broadcast;

async fn broadcast_example() {
    let (tx, mut rx1) = broadcast::channel(16);
    let mut rx2 = tx.subscribe();

    tokio::spawn(async move {
        while let Ok(msg) = rx1.recv().await {
            println!("Receiver 1: {}", msg);
        }
    });

    tokio::spawn(async move {
        while let Ok(msg) = rx2.recv().await {
            println!("Receiver 2: {}", msg);
        }
    });

    tx.send("Hello").unwrap();
    tx.send("World").unwrap();
}
```

### 19.6 Async 스트림 (Stream)

```rust
use tokio_stream::{Stream, StreamExt};
use std::pin::Pin;

// Stream 트레이트
trait Stream {
    type Item;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>)
        -> Poll<Option<Self::Item>>;
}

// 실전 스트림 사용
async fn stream_example() {
    let stream = tokio_stream::iter(vec![1, 2, 3, 4, 5]);

    tokio::pin!(stream);

    while let Some(value) = stream.next().await {
        println!("{}", value);
    }
}

// 스트림 변환
async fn stream_transform() {
    let numbers = tokio_stream::iter(1..=10);

    let doubled = numbers
        .map(|x| x * 2)
        .filter(|x| x % 4 == 0)
        .take(3);

    tokio::pin!(doubled);

    while let Some(value) = doubled.next().await {
        println!("{}", value);  // 4, 8, 12
    }
}
```

### 19.7 Tokio 런타임 심화

```rust
use tokio::runtime::{Runtime, Builder};

// 커스텀 런타임
fn custom_runtime() {
    let rt = Builder::new_multi_thread()
        .worker_threads(4)
        .thread_name("my-custom-thread")
        .thread_stack_size(3 * 1024 * 1024)
        .build()
        .unwrap();

    rt.block_on(async {
        println!("Running on custom runtime");
    });
}

// 현재 스레드 런타임 (단일 스레드)
#[tokio::main(flavor = "current_thread")]
async fn main() {
    // 단일 스레드에서 실행
}

// 작업 스폰
async fn spawn_tasks() {
    let handle = tokio::spawn(async {
        // 백그라운드 작업
        expensive_computation().await
    });

    // 다른 작업 수행...

    // 결과 대기
    let result = handle.await.unwrap();
}

// 블로킹 작업 처리
async fn blocking_task() {
    let result = tokio::task::spawn_blocking(|| {
        // CPU 집약적 작업 또는 블로킹 I/O
        std::thread::sleep(std::time::Duration::from_secs(1));
        42
    }).await.unwrap();

    println!("Blocking task result: {}", result);
}
```

---

## 20. 고급 기능과 패턴

### 20.1 매크로

```rust
// 선언적 매크로
macro_rules! vec_macro {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            temp_vec
        }
    };
}

// 절차적 매크로
use proc_macro;

#[proc_macro_derive(HelloMacro)]
pub fn hello_macro_derive(input: TokenStream) -> TokenStream {
    // ...
}
```

### 20.2 unsafe Rust - 완전 가이드

**왜 unsafe가 필요한가?**

```rust
// 1. 원시 포인터 역참조
let mut num = 5;
let r1 = &num as *const i32;  // 불변 원시 포인터
let r2 = &mut num as *mut i32;  // 가변 원시 포인터

// 원시 포인터의 특징:
// - null 가능, 빌림 검사 무시, 자동 정리 안됨, 데이터 레이스 가능
unsafe {
    println!("r1: {}", *r1);
    *r2 = 10;
}

// 2. unsafe 함수/메서드 호출
unsafe fn dangerous() {
    println!("Dangerous operation!");
}

unsafe {
    dangerous();
}

// 3. 가변 정적 변수 접근
static mut COUNTER: u32 = 0;

fn increment() {
    unsafe {
        COUNTER += 1;
    }
}

// 4. unsafe 트레이트 구현
unsafe trait Foo {}
unsafe impl Foo for i32 {}
```

**안전한 추상화 만들기**

```rust
use std::slice;

// 원시 포인터를 사용하지만 안전한 API 제공
fn split_at_mut(slice: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    let len = slice.len();
    let ptr = slice.as_mut_ptr();
    assert!(mid <= len);

    unsafe {
        (
            slice::from_raw_parts_mut(ptr, mid),
            slice::from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}
```

**FFI (Foreign Function Interface)**

```rust
extern "C" {
    fn abs(input: i32) -> i32;
}

#[no_mangle]
pub extern "C" fn call_from_c() -> i32 {
    42
}

#[repr(C)]
struct Point {
    x: f64,
    y: f64,
}
```

### 20.3 고급 트레이트

```rust
// 연관 타입
pub trait Iterator {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}

// 기본 타입 매개변수
use std::ops::Add;

#[derive(Debug, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

impl Add for Point {
    type Output = Point;

    fn add(self, other: Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

// 슈퍼트레이트
trait OutlinePrint: fmt::Display {
    fn outline_print(&self) {
        println!("* {} *", self);
    }
}
```

---

## 결론

Rust는 메모리 안전성과 동시성을 보장하는 시스템 프로그래밍 언어입니다:

1. **소유권**: 가비지 컬렉터 없이 메모리 안전성
2. **타입 시스템**: 컴파일 타임 에러 검출
3. **제로 코스트 추상화**: 고수준 코드, 저수준 성능
4. **동시성**: 데이터 경쟁 방지
5. **패턴 매칭**: 강력한 제어 흐름

**학습 순서**: 1-6 → 4-5 → 7-10 → 11-13 → 14 → 15-18

Rust로 안전하고 빠른 시스템을 만드세요!
