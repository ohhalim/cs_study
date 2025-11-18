# C 언어 완벽 학습 가이드

## 목차
1. [C 언어 기초](#1-c-언어-기초)
2. [데이터 타입과 변수](#2-데이터-타입과-변수)
3. [연산자](#3-연산자)
4. [제어문](#4-제어문)
5. [함수](#5-함수)
6. [포인터](#6-포인터)
7. [배열과 문자열](#7-배열과-문자열)
8. [구조체와 공용체](#8-구조체와-공용체)
9. [동적 메모리 할당](#9-동적-메모리-할당)
10. [파일 입출력](#10-파일-입출력)
11. [전처리기](#11-전처리기)
12. [고급 포인터](#12-고급-포인터)
13. [비트 연산](#13-비트-연산)
14. [메모리 관리](#14-메모리-관리)
15. [컴파일과 링킹](#15-컴파일과-링킹)
16. [표준 라이브러리](#16-표준-라이브러리)
17. [고급 기법](#17-고급-기법)
18. [최적화와 성능](#18-최적화와-성능)
19. [디버깅과 도구](#19-디버깅과-도구)
20. [실전 프로젝트 패턴](#20-실전-프로젝트-패턴)

---

## 1. C 언어 기초

### 1.1 C 언어란?
- 1972년 Dennis Ritchie가 UNIX 개발을 위해 만든 범용 프로그래밍 언어
- 저수준 메모리 접근과 고수준 추상화를 모두 제공
- 시스템 프로그래밍, 임베디드, OS 커널 개발에 필수

### 1.2 첫 프로그램
```c
#include <stdio.h>

int main(void) {
    printf("Hello, World!\n");
    return 0;
}
```

**분석:**
- `#include <stdio.h>`: 표준 입출력 헤더 포함
- `int main(void)`: 프로그램 진입점
- `printf()`: 표준 출력 함수
- `return 0`: 정상 종료 (0 = 성공)

### 1.3 컴파일 과정
```bash
# 전처리
gcc -E source.c -o source.i

# 컴파일 (어셈블리 생성)
gcc -S source.i -o source.s

# 어셈블 (오브젝트 파일 생성)
gcc -c source.s -o source.o

# 링킹 (실행 파일 생성)
gcc source.o -o program

# 한번에
gcc source.c -o program
```

---

## 2. 데이터 타입과 변수

### 2.1 기본 데이터 타입

**📌 C 표준 버전별 기능**
- **C89/C90**: 최초 ANSI/ISO 표준
- **C99**: `long long`, `bool`, 가변 길이 배열, inline, 지정 초기화
- **C11**: `_Atomic`, `_Generic`, `_Thread_local`, 익명 구조체/공용체
- **C17/C18**: C11의 결함 수정, 새 기능 없음
- **C23**: `typeof`, 이진 리터럴, `constexpr` (제한적), `#embed`, `_BitInt`

```c
// 정수형 (C89부터)
char c = 'A';           // 1 byte, -128 ~ 127
unsigned char uc = 255; // 1 byte, 0 ~ 255
short s = 32000;        // 2 bytes, -32,768 ~ 32,767
unsigned short us;      // 2 bytes, 0 ~ 65,535
int i = 100;            // 4 bytes (보통), -2^31 ~ 2^31-1
unsigned int ui;        // 4 bytes, 0 ~ 2^32-1
long l = 1000000L;      // 4 or 8 bytes

// C99 이상
long long ll = 1LL;     // 8 bytes (C99+)
unsigned long long ull; // 8 bytes, 0 ~ 2^64-1 (C99+)

// C23 이상
_BitInt(128) big = 0wb;  // 임의 비트 정수 (C23+)

// 실수형 (C89부터)
float f = 3.14f;        // 4 bytes, 정밀도 6-7자리
double d = 3.14159;     // 8 bytes, 정밀도 15-16자리
long double ld = 3.14L; // 10+ bytes

// 불린형 (C99 이상)
#include <stdbool.h>
bool flag = true;       // _Bool 타입 (C99+)

// void
void *ptr;              // 타입 없는 포인터

// 이진/16진 리터럴
int binary = 0b1010;    // 10 (C23+ 또는 GCC 확장)
int hex = 0xFF;         // 255 (C89+)
int oct = 077;          // 63 (C89+)

// 구분자 (C23+)
int million = 1'000'000;  // C23+ (GCC는 이미 지원)
```

### 2.2 크기 확인
```c
#include <stdio.h>
#include <limits.h>
#include <float.h>

int main(void) {
    printf("char: %zu bytes, range: %d ~ %d\n",
           sizeof(char), CHAR_MIN, CHAR_MAX);
    printf("int: %zu bytes, range: %d ~ %d\n",
           sizeof(int), INT_MIN, INT_MAX);
    printf("long long: %zu bytes\n", sizeof(long long));
    printf("float: %zu bytes, precision: %d\n",
           sizeof(float), FLT_DIG);
    printf("double: %zu bytes, precision: %d\n",
           sizeof(double), DBL_DIG);
    return 0;
}
```

### 2.3 타입 한정자 (Type Qualifiers)

```c
// const - 값 변경 불가
const int MAX = 100;
const char *str = "Hello";  // 문자열 내용 변경 불가
char * const ptr = arr;     // 포인터 변경 불가
const char * const p = str; // 둘 다 변경 불가

// volatile - 컴파일러 최적화 방지 (하드웨어 레지스터 등)
volatile int *hardware_register = (volatile int *)0x40021000;

// restrict (C99) - 포인터 앨리어싱 방지로 최적화 향상
void copy(int * restrict dest, const int * restrict src, size_t n);

// _Atomic (C11) - 원자적 연산 보장
#include <stdatomic.h>
_Atomic int counter = 0;
atomic_fetch_add(&counter, 1);
```

### 2.4 저장 클래스 지정자

```c
// auto (기본값, 지역 변수)
auto int x = 10;

// register - 레지스터 저장 요청 (주소 접근 불가)
register int i;
for (i = 0; i < 1000000; i++) { }

// static
// 1. 파일 범위: 파일 내부에서만 접근 가능
static int file_scope_var = 0;

static void internal_function(void) {
    // 다른 파일에서 접근 불가
}

// 2. 함수 범위: 프로그램 종료까지 유지
void counter(void) {
    static int count = 0;  // 한 번만 초기화
    count++;
    printf("%d\n", count);
}

// extern - 다른 파일의 전역 변수/함수 참조
extern int global_var;
extern void external_function(void);
```

### 2.5 타입 변환

```c
// 암시적 변환 (Implicit Conversion)
int i = 10;
float f = i;           // int -> float
double d = 3.14;
int x = d;             // double -> int (손실 발생)

// 정수 승격 (Integer Promotion)
char c1 = 100, c2 = 3;
int result = c1 + c2;  // char -> int로 승격 후 연산

// 명시적 변환 (Type Casting)
double pi = 3.14159;
int truncated = (int)pi;           // 3
float f2 = (float)pi;

// 포인터 캐스팅
void *generic_ptr = malloc(10);
int *int_ptr = (int *)generic_ptr;

// 위험한 캐스팅 예시
float val = 3.14f;
int *danger = (int *)&val;  // 비트 패턴 해석 변경
printf("%d\n", *danger);    // 쓰레기 값
```

---

## 3. 연산자

### 3.1 산술 연산자
```c
int a = 10, b = 3;
printf("%d\n", a + b);   // 13 (덧셈)
printf("%d\n", a - b);   // 7  (뺄셈)
printf("%d\n", a * b);   // 30 (곱셈)
printf("%d\n", a / b);   // 3  (정수 나눗셈, 몫)
printf("%d\n", a % b);   // 1  (나머지)

// 실수 나눗셈
printf("%f\n", a / (double)b);  // 3.333333

// 증감 연산자
int x = 5;
printf("%d\n", x++);  // 5 (후위 증가)
printf("%d\n", x);    // 6
printf("%d\n", ++x);  // 7 (전위 증가)
```

### 3.2 관계 및 논리 연산자
```c
int a = 5, b = 10;

// 관계 연산자
printf("%d\n", a == b);  // 0 (false)
printf("%d\n", a != b);  // 1 (true)
printf("%d\n", a < b);   // 1
printf("%d\n", a <= b);  // 1
printf("%d\n", a > b);   // 0
printf("%d\n", a >= b);  // 0

// 논리 연산자
printf("%d\n", (a < 10) && (b > 5));  // 1 (AND)
printf("%d\n", (a < 3) || (b > 5));   // 1 (OR)
printf("%d\n", !(a == 5));            // 0 (NOT)

// 단축 평가 (Short-circuit Evaluation)
int x = 0;
if (x != 0 && 10 / x > 1) {  // 10/x는 평가되지 않음
    // ...
}
```

### 3.3 비트 연산자
```c
unsigned int a = 0b1100;  // 12 (C23 또는 GCC 확장)
unsigned int b = 0b1010;  // 10

printf("%u\n", a & b);    // 0b1000 = 8  (AND)
printf("%u\n", a | b);    // 0b1110 = 14 (OR)
printf("%u\n", a ^ b);    // 0b0110 = 6  (XOR)
printf("%u\n", ~a);       // 비트 반전
printf("%u\n", a << 2);   // 48 (왼쪽 시프트, *4)
printf("%u\n", a >> 2);   // 3  (오른쪽 시프트, /4)

// 실용 예시
// 비트 설정
unsigned int flags = 0;
flags |= (1 << 3);        // 3번 비트 설정

// 비트 해제
flags &= ~(1 << 3);       // 3번 비트 해제

// 비트 토글
flags ^= (1 << 3);        // 3번 비트 반전

// 비트 확인
if (flags & (1 << 3)) {   // 3번 비트 확인
    printf("Bit 3 is set\n");
}
```

### 3.4 대입 연산자
```c
int x = 10;
x += 5;   // x = x + 5
x -= 3;   // x = x - 3
x *= 2;   // x = x * 2
x /= 4;   // x = x / 4
x %= 3;   // x = x % 3
x &= 0xF; // x = x & 0xF
x |= 0x10;// x = x | 0x10
x ^= 0xFF;// x = x ^ 0xFF
x <<= 2;  // x = x << 2
x >>= 1;  // x = x >> 1
```

### 3.5 기타 연산자
```c
// 삼항 연산자
int max = (a > b) ? a : b;
int abs_val = (x >= 0) ? x : -x;

// sizeof 연산자
printf("%zu\n", sizeof(int));
printf("%zu\n", sizeof(arr) / sizeof(arr[0])); // 배열 원소 개수

// 쉼표 연산자
int a, b, c;
c = (a = 1, b = 2, a + b);  // c = 3 (마지막 표현식 값)

// 주소 연산자 &, 간접 연산자 *
int var = 10;
int *ptr = &var;  // 주소
int val = *ptr;   // 역참조

// 멤버 접근 연산자
struct Point {
    int x, y;
} p, *ptr_p;
p.x = 10;         // . (구조체 멤버)
ptr_p = &p;
ptr_p->x = 20;    // -> (포인터를 통한 멤버 접근)
```

### 3.6 연산자 우선순위와 결합성
```c
/*
우선순위 (높음 -> 낮음):
1. () [] -> . (좌->우)
2. ! ~ ++ -- + - * & (type) sizeof (우->좌, 단항)
3. * / % (좌->우)
4. + - (좌->우)
5. << >> (좌->우)
6. < <= > >= (좌->우)
7. == != (좌->우)
8. & (좌->우)
9. ^ (좌->우)
10. | (좌->우)
11. && (좌->우)
12. || (좌->우)
13. ?: (우->좌)
14. = += -= 등 (우->좌)
15. , (좌->우)
*/

// 예시
int x = 5;
int y = x++ + ++x;  // y = 5 + 7 = 12, x = 7
```

**⚠️ 미정의 동작 (Undefined Behavior) 상세 설명**

```c
// ❌ 미정의 동작 예시들

// 1. 시퀀스 포인트 위반
int i = 1;
int j = i++ + ++i;  // UB! i를 여러 번 수정

// 왜 UB인가?
// - 시퀀스 포인트: 모든 이전 연산이 완료되고 다음 연산이 시작되기 전의 지점
// - ; (세미콜론), && || ?: , (쉼표 연산자) 등
// - 한 시퀀스 포인트 사이에 같은 변수를 두 번 이상 수정하면 UB

int arr[10];
int idx = 0;
arr[idx++] = idx;  // UB! idx를 읽고 쓰기를 동시에

// 2. 평가 순서 미정의
int x = 5;
printf("%d %d\n", x++, x++);  // UB! 어느 것이 먼저 평가될지 모름

// 3. 함수 인자 평가 순서
int i = 0;
func(i++, i++, i++);  // UB! 인자 평가 순서는 미정의

// ✅ 안전한 방법
int i = 1;
int temp1 = i++;
int temp2 = ++i;
int j = temp1 + temp2;  // 명확함

// 또는
int i = 1;
i++;        // i = 2
i++;        // i = 3
int j = 1 + 3;  // 명확함

// 명확하게 표현
int a = *ptr++;    // *(ptr++) - 후위 증가가 우선
int b = (*ptr)++;  // 값 증가
int c = *++ptr;    // *(++ptr) - 전위 증가 후 역참조
```

**시퀀스 포인트 위치:**
```c
// 1. ; (문장 끝)
x = 1; y = 2;  // ; 에서 시퀀스 포인트

// 2. && 와 || (단축 평가)
if (ptr != NULL && *ptr == 10)  // && 전후로 시퀀스 포인트
    // ptr != NULL이 먼저 완료된 후 *ptr 평가

// 3. ?: (조건 연산자)
result = (x > 0) ? x : -x;  // ? 와 : 에서 시퀀스 포인트

// 4. , (쉼표 연산자)
x = (a = 1, b = 2, a + b);  // 각 쉼표에서 시퀀스 포인트

// 5. 함수 호출 (인자 평가 후, 함수 진입 전)
func(x++, y++);  // 하지만 x++와 y++ 중 어느 것이 먼저인지는 미정의!
```

**안전한 코딩 규칙:**
1. 한 표현식에서 변수를 한 번만 수정하라
2. 읽기와 쓰기를 분리하라
3. 복잡한 표현식을 여러 문장으로 나눠라
4. 컴파일러 경고를 무시하지 마라 (`-Wall -Wextra`)
```

---

## 4. 제어문

### 4.1 조건문

```c
// if-else
int score = 85;
if (score >= 90) {
    printf("A\n");
} else if (score >= 80) {
    printf("B\n");
} else if (score >= 70) {
    printf("C\n");
} else {
    printf("F\n");
}

// 중첩 if
if (x > 0) {
    if (x < 10) {
        printf("0 < x < 10\n");
    }
}

// switch-case
int day = 3;
switch (day) {
    case 1:
        printf("Monday\n");
        break;
    case 2:
        printf("Tuesday\n");
        break;
    case 3:
    case 4:
    case 5:
        printf("Midweek\n");
        break;
    case 6:
    case 7:
        printf("Weekend\n");
        break;
    default:
        printf("Invalid day\n");
        break;
}

// Fall-through 활용
switch (c) {
    case 'a':
    case 'e':
    case 'i':
    case 'o':
    case 'u':
        printf("Vowel\n");
        break;
    default:
        printf("Consonant\n");
}
```

### 4.2 반복문

```c
// for 루프
for (int i = 0; i < 10; i++) {
    printf("%d ", i);
}

// 무한 루프
for (;;) {
    // ...
    if (condition) break;
}

// 여러 변수
for (int i = 0, j = 10; i < j; i++, j--) {
    printf("%d %d\n", i, j);
}

// while 루프
int i = 0;
while (i < 10) {
    printf("%d ", i);
    i++;
}

// do-while (최소 1회 실행)
int count = 0;
do {
    printf("%d ", count);
    count++;
} while (count < 5);

// 중첩 루프
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        printf("(%d,%d) ", i, j);
    }
    printf("\n");
}
```

### 4.3 분기문

```c
// break - 루프 탈출
for (int i = 0; i < 10; i++) {
    if (i == 5) break;
    printf("%d ", i);  // 0 1 2 3 4
}

// continue - 다음 반복으로
for (int i = 0; i < 10; i++) {
    if (i % 2 == 0) continue;
    printf("%d ", i);  // 1 3 5 7 9
}

// goto (사용 자제)
for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
        if (error_condition) {
            goto error_handler;
        }
    }
}
// ...
error_handler:
    cleanup();
    return -1;

// 레이블은 함수 내에서만 사용 가능
```

### 4.4 제어문 패턴

```c
// 가드 클로즈 패턴
int process(int *data) {
    if (data == NULL) {
        return -1;  // 조기 반환
    }
    if (*data < 0) {
        return -2;
    }
    // 정상 처리
    return 0;
}

// 상태 머신 패턴
enum State { IDLE, RUNNING, PAUSED, STOPPED };
enum State current_state = IDLE;

while (1) {
    switch (current_state) {
        case IDLE:
            if (start_requested) {
                current_state = RUNNING;
            }
            break;
        case RUNNING:
            do_work();
            if (pause_requested) {
                current_state = PAUSED;
            }
            break;
        case PAUSED:
            if (resume_requested) {
                current_state = RUNNING;
            }
            break;
        case STOPPED:
            cleanup();
            return 0;
    }
}
```

---

## 5. 함수

### 5.1 함수 기본

```c
// 함수 선언 (프로토타입)
int add(int a, int b);
void print_array(int arr[], int size);
double calculate(double x);

// 함수 정의
int add(int a, int b) {
    return a + b;
}

// void 함수
void greet(const char *name) {
    printf("Hello, %s!\n", name);
}

// 매개변수 없는 함수
int get_random(void) {
    return rand();
}

// 가변 인자 함수
#include <stdarg.h>

int sum(int count, ...) {
    va_list args;
    va_start(args, count);

    int total = 0;
    for (int i = 0; i < count; i++) {
        total += va_arg(args, int);
    }

    va_end(args);
    return total;
}

// 사용
int result = sum(4, 10, 20, 30, 40);  // 100
```

### 5.2 함수 포인터

```c
// 기본 함수 포인터
int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }

int (*operation)(int, int);  // 함수 포인터 선언
operation = add;
printf("%d\n", operation(10, 5));  // 15

operation = subtract;
printf("%d\n", operation(10, 5));  // 5

// typedef로 가독성 향상
typedef int (*BinaryOp)(int, int);
BinaryOp op = add;

// 함수 포인터 배열
BinaryOp operations[] = { add, subtract, multiply, divide };
int result = operations[0](10, 5);  // add 호출

// 콜백 함수
void apply(int *arr, int size, int (*func)(int)) {
    for (int i = 0; i < size; i++) {
        arr[i] = func(arr[i]);
    }
}

int square(int x) { return x * x; }

int arr[] = {1, 2, 3, 4, 5};
apply(arr, 5, square);  // 각 원소 제곱
```

### 5.3 재귀 함수

```c
// 팩토리얼
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// 피보나치 (비효율적)
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// 피보나치 (메모이제이션)
#define MAX 100
int memo[MAX] = {0};

int fibonacci_memo(int n) {
    if (n <= 1) return n;
    if (memo[n] != 0) return memo[n];
    memo[n] = fibonacci_memo(n - 1) + fibonacci_memo(n - 2);
    return memo[n];
}

// 꼬리 재귀 최적화
int factorial_tail(int n, int acc) {
    if (n <= 1) return acc;
    return factorial_tail(n - 1, n * acc);
}

int factorial_optimized(int n) {
    return factorial_tail(n, 1);
}

// 이진 탐색 (재귀)
int binary_search(int arr[], int left, int right, int target) {
    if (left > right) return -1;

    int mid = left + (right - left) / 2;
    if (arr[mid] == target) return mid;
    if (arr[mid] > target) {
        return binary_search(arr, left, mid - 1, target);
    }
    return binary_search(arr, mid + 1, right, target);
}
```

### 5.4 인라인 함수 (C99)

```c
// 인라인 함수 - 함수 호출 오버헤드 제거
inline int max(int a, int b) {
    return (a > b) ? a : b;
}

// static inline - 파일 내부에서만 사용
static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

// 주의: 컴파일러는 inline 힌트를 무시할 수 있음
// 복잡한 함수는 인라인하지 않는 것이 좋음
```

### 5.5 함수와 스택

```c
// 스택 프레임 이해
void func_c(int z) {
    int local_c = z * 2;
    printf("Address of local_c: %p\n", (void*)&local_c);
}

void func_b(int y) {
    int local_b = y + 1;
    printf("Address of local_b: %p\n", (void*)&local_b);
    func_c(local_b);
}

void func_a(int x) {
    int local_a = x;
    printf("Address of local_a: %p\n", (void*)&local_a);
    func_b(local_a);
}

// 호출: func_a(10) -> func_b(10) -> func_c(11)
// 스택: [func_a 프레임] [func_b 프레임] [func_c 프레임]
```

---

## 6. 포인터

### 6.1 포인터 기초

```c
// 포인터 선언과 초기화
int x = 10;
int *ptr = &x;        // ptr은 x의 주소를 저장

printf("x = %d\n", x);          // 10
printf("&x = %p\n", (void*)&x); // 주소
printf("ptr = %p\n", (void*)ptr);    // x의 주소
printf("*ptr = %d\n", *ptr);    // 10 (역참조)

// 포인터를 통한 값 변경
*ptr = 20;
printf("x = %d\n", x);  // 20

// NULL 포인터
int *null_ptr = NULL;
if (null_ptr == NULL) {
    printf("Pointer is NULL\n");
}

// void 포인터 (제네릭 포인터)
void *generic_ptr;
int i = 10;
float f = 3.14f;
generic_ptr = &i;
printf("%d\n", *(int*)generic_ptr);   // 10
generic_ptr = &f;
printf("%f\n", *(float*)generic_ptr); // 3.14
```

### 6.2 포인터와 배열

```c
int arr[] = {1, 2, 3, 4, 5};
int *ptr = arr;  // 배열 이름은 첫 번째 원소의 주소

// 배열 접근 방식
printf("%d\n", arr[0]);     // 1
printf("%d\n", *arr);       // 1
printf("%d\n", *(arr + 0)); // 1

printf("%d\n", arr[2]);     // 3
printf("%d\n", *(arr + 2)); // 3
printf("%d\n", ptr[2]);     // 3

// 포인터 산술 연산
ptr++;           // 다음 int 위치로 이동 (4바이트 전진)
printf("%d\n", *ptr);  // 2

ptr += 2;        // 2칸 이동
printf("%d\n", *ptr);  // 4

// 배열 순회
for (int i = 0; i < 5; i++) {
    printf("%d ", arr[i]);
}

for (int *p = arr; p < arr + 5; p++) {
    printf("%d ", *p);
}

// 주소 차이
int *p1 = &arr[1];
int *p2 = &arr[4];
ptrdiff_t diff = p2 - p1;  // 3 (원소 개수 차이)
```

### 6.3 포인터와 문자열

```c
// 문자열 리터럴
char *str1 = "Hello";  // 읽기 전용 메모리
// str1[0] = 'h';  // 에러! (세그멘테이션 폴트)

// 문자 배열
char str2[] = "Hello";  // 스택에 복사됨
str2[0] = 'h';          // OK
printf("%s\n", str2);   // "hello"

// 문자열 순회
char *p = str1;
while (*p != '\0') {
    printf("%c", *p);
    p++;
}

// 문자열 길이 계산
size_t my_strlen(const char *s) {
    const char *p = s;
    while (*p != '\0') {
        p++;
    }
    return p - s;
}

// 문자열 복사
void my_strcpy(char *dest, const char *src) {
    while ((*dest++ = *src++) != '\0');
}

// 또는
void my_strcpy2(char *dest, const char *src) {
    while (*src) {
        *dest++ = *src++;
    }
    *dest = '\0';
}
```

### 6.4 다중 포인터

```c
// 이중 포인터
int x = 10;
int *ptr = &x;
int **ptr_to_ptr = &ptr;

printf("%d\n", **ptr_to_ptr);  // 10

**ptr_to_ptr = 20;
printf("%d\n", x);  // 20

// 문자열 배열 (포인터 배열)
char *names[] = {
    "Alice",
    "Bob",
    "Charlie"
};

for (int i = 0; i < 3; i++) {
    printf("%s\n", names[i]);
}

// 2차원 배열과 이중 포인터의 차이
int matrix[3][4];        // 연속된 메모리
int *ptrs[3];            // 포인터 배열 (불연속 가능)

// 함수 매개변수로 이중 포인터
void modify_pointer(int **ptr) {
    *ptr = malloc(sizeof(int));
    **ptr = 100;
}

int *p = NULL;
modify_pointer(&p);
printf("%d\n", *p);  // 100
free(p);

// 삼중 포인터 (드물게 사용)
int ***triple_ptr;
```

### 6.5 const와 포인터

```c
int x = 10, y = 20;

// 포인터가 가리키는 값이 상수
const int *ptr1 = &x;
// *ptr1 = 20;  // 에러
ptr1 = &y;      // OK

// 포인터 자체가 상수
int * const ptr2 = &x;
*ptr2 = 30;     // OK
// ptr2 = &y;   // 에러

// 둘 다 상수
const int * const ptr3 = &x;
// *ptr3 = 40;  // 에러
// ptr3 = &y;   // 에러

// 읽는 방법: 오른쪽에서 왼쪽으로
// const int *p       : p는 const int를 가리키는 포인터
// int * const p      : p는 int를 가리키는 const 포인터
// const int * const p: p는 const int를 가리키는 const 포인터
```

### 6.6 포인터와 함수

```c
// 포인터를 반환하는 함수
int* get_pointer(void) {
    static int x = 10;  // static 필수!
    return &x;
}

// 위험한 예시
int* dangerous(void) {
    int x = 10;  // 지역 변수
    return &x;   // 댕글링 포인터! (함수 종료 후 무효)
}

// 배열 반환 (실제로는 포인터)
int* create_array(int size) {
    int *arr = malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        arr[i] = i;
    }
    return arr;
}

// 사용
int *arr = create_array(10);
// 사용 후 반드시 해제
free(arr);

// 함수 포인터 (재방문)
int (*func_ptr)(int, int);
int add(int a, int b) { return a + b; }
func_ptr = add;
printf("%d\n", (*func_ptr)(3, 4));  // 7
printf("%d\n", func_ptr(3, 4));     // 7 (동일)
```

---

## 7. 배열과 문자열

### 7.1 1차원 배열

```c
// 선언과 초기화
int arr1[5];                    // 초기화 안됨 (쓰레기 값)
int arr2[5] = {1, 2, 3, 4, 5}; // 전체 초기화
int arr3[5] = {1, 2};          // {1, 2, 0, 0, 0}
int arr4[] = {1, 2, 3};        // 크기 자동 (3)
int arr5[10] = {0};            // 모두 0으로 초기화

// C99 지정 초기화
int arr6[10] = {[0] = 1, [5] = 2, [9] = 3};

// 배열 크기
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
int size = ARRAY_SIZE(arr2);  // 5

// 배열 순회
for (int i = 0; i < size; i++) {
    printf("%d ", arr2[i]);
}

// 배열 복사 (memcpy 사용)
#include <string.h>
int source[] = {1, 2, 3, 4, 5};
int dest[5];
memcpy(dest, source, sizeof(source));

// 배열 비교
if (memcmp(arr1, arr2, sizeof(arr1)) == 0) {
    printf("Arrays are equal\n");
}
```

### 7.2 다차원 배열

```c
// 2차원 배열
int matrix[3][4] = {
    {1, 2, 3, 4},
    {5, 6, 7, 8},
    {9, 10, 11, 12}
};

// 부분 초기화
int matrix2[3][4] = {{1}, {5}, {9}};
// {1, 0, 0, 0}
// {5, 0, 0, 0}
// {9, 0, 0, 0}

// 접근
printf("%d\n", matrix[1][2]);  // 7

// 순회
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
        printf("%d ", matrix[i][j]);
    }
    printf("\n");
}

// 3차원 배열
int cube[2][3][4] = {
    {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    },
    {
        {13, 14, 15, 16},
        {17, 18, 19, 20},
        {21, 22, 23, 24}
    }
};

// 메모리 레이아웃: 연속된 메모리에 행 우선 순서로 저장
```

### 7.3 가변 길이 배열 (VLA, C99)

```c
#include <stdio.h>

void process(int n) {
    int arr[n];  // 런타임에 크기 결정
    for (int i = 0; i < n; i++) {
        arr[i] = i * i;
    }
    // 함수 종료 시 자동 해제
}

// 2차원 VLA
void matrix_multiply(int n, int m, int k,
                    int a[n][m], int b[m][k], int result[n][k]) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            result[i][j] = 0;
            for (int l = 0; l < m; l++) {
                result[i][j] += a[i][l] * b[l][j];
            }
        }
    }
}

// 주의: VLA는 스택에 할당되므로 큰 배열은 위험
```

### 7.4 문자열 함수

```c
#include <string.h>
#include <stdio.h>

// strlen - 문자열 길이
char str[] = "Hello";
size_t len = strlen(str);  // 5 ('\0' 제외)

// strcpy, strncpy - 문자열 복사
char dest[20];
strcpy(dest, str);              // 복사
strncpy(dest, str, 19);         // 최대 19자 복사
dest[19] = '\0';                // 안전을 위해 널 종료

// strcat, strncat - 문자열 연결
char buffer[50] = "Hello";
strcat(buffer, " World");       // "Hello World"
strncat(buffer, "!!!", 3);      // "Hello World!!!"

// strcmp, strncmp - 문자열 비교
if (strcmp(str1, str2) == 0) {  // 같으면 0
    printf("Equal\n");
}
int result = strncmp(str1, str2, 5);  // 처음 5자만 비교

// strchr, strrchr - 문자 찾기
char *p = strchr(str, 'l');     // 첫 번째 'l'의 위치
char *q = strrchr(str, 'l');    // 마지막 'l'의 위치

// strstr - 부분 문자열 찾기
char *pos = strstr("Hello World", "World");  // "World"의 위치

// strtok - 문자열 토큰화
char text[] = "one,two,three";
char *token = strtok(text, ",");
while (token != NULL) {
    printf("%s\n", token);
    token = strtok(NULL, ",");
}

// sprintf, snprintf - 문자열 포맷팅
char buffer[100];
sprintf(buffer, "Value: %d", 42);
snprintf(buffer, sizeof(buffer), "Safe: %d", 42);  // 버퍼 오버플로우 방지

// 안전한 문자열 함수 (C11)
#ifdef __STDC_LIB_EXT1__
strcpy_s(dest, sizeof(dest), src);
strcat_s(dest, sizeof(dest), src);
#endif
```

**⚠️ 보안 경고: 위험한 문자열 함수들**

```c
// ❌ 절대 사용 금지: gets() - 버퍼 오버플로우 취약
char buffer[10];
// gets(buffer);  // C11에서 제거됨! 버퍼 크기 확인 안함

// ✅ 대신 fgets() 사용
fgets(buffer, sizeof(buffer), stdin);
// 개행 문자 제거 필요
buffer[strcspn(buffer, "\n")] = '\0';

// ❌ 위험: strcpy() - 대상 버퍼 크기 확인 안함
char dest[10];
char src[100] = "Very long string that will overflow";
// strcpy(dest, src);  // 버퍼 오버플로우!

// ✅ 안전: strncpy() (하지만 주의 필요)
strncpy(dest, src, sizeof(dest) - 1);
dest[sizeof(dest) - 1] = '\0';  // 널 종료 보장

// ✅ 더 안전: snprintf()로 복사
snprintf(dest, sizeof(dest), "%s", src);

// ❌ 위험: sprintf() - 버퍼 오버플로우
// sprintf(buffer, "Value: %d", very_large_number);

// ✅ 안전: snprintf()
snprintf(buffer, sizeof(buffer), "Value: %d", very_large_number);

// ❌ 위험: 포맷 스트링 취약점
char user_input[100];
// printf(user_input);  // 사용자가 %s, %n 등을 입력하면 위험!

// ✅ 안전
printf("%s", user_input);
```

**포맷 스트링 공격 예시:**
```c
// 공격자가 입력: "%x %x %x %x"
// 스택 메모리 내용이 노출됨

// 공격자가 입력: "%n"
// 임의의 메모리 위치에 쓰기 가능 → 코드 실행

// 해결책: 항상 포맷 문자열을 리터럴로 사용
printf("%s", user_input);  // 안전
fprintf(stderr, "Error: %s\n", error_msg);  // 안전
```

**문자열 함수 보안 가이드:**

| 위험한 함수 | 안전한 대안 | 이유 |
|------------|-----------|------|
| `gets()` | `fgets()` | 버퍼 크기 확인 안함 |
| `strcpy()` | `strncpy()`, `strlcpy()` | 버퍼 크기 확인 안함 |
| `strcat()` | `strncat()`, `strlcat()` | 버퍼 크기 확인 안함 |
| `sprintf()` | `snprintf()` | 버퍼 크기 확인 안함 |
| `scanf("%s")` | `scanf("%99s")` | 너비 지정 필요 |

### 7.5 문자열 처리 예제

```c
// 문자열 뒤집기
void reverse_string(char *str) {
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        char temp = str[i];
        str[i] = str[len - 1 - i];
        str[len - 1 - i] = temp;
    }
}

// 공백 제거
void trim(char *str) {
    int i = 0, j = 0;
    while (str[i] != '\0') {
        if (str[i] != ' ' && str[i] != '\t' && str[i] != '\n') {
            str[j++] = str[i];
        }
        i++;
    }
    str[j] = '\0';
}

// 대소문자 변환
#include <ctype.h>
void to_upper(char *str) {
    for (int i = 0; str[i]; i++) {
        str[i] = toupper(str[i]);
    }
}

void to_lower(char *str) {
    for (int i = 0; str[i]; i++) {
        str[i] = tolower(str[i]);
    }
}

// 문자열이 숫자인지 확인
int is_numeric(const char *str) {
    for (int i = 0; str[i]; i++) {
        if (!isdigit(str[i])) {
            return 0;
        }
    }
    return 1;
}
```

---

## 8. 구조체와 공용체

### 8.1 구조체 기본

```c
// 구조체 정의
struct Point {
    int x;
    int y;
};

// 변수 선언
struct Point p1;
p1.x = 10;
p1.y = 20;

// 초기화
struct Point p2 = {30, 40};
struct Point p3 = {.y = 50, .x = 60};  // C99 지정 초기화

// typedef 사용
typedef struct {
    int x;
    int y;
} Point;

Point p4 = {70, 80};  // struct 키워드 불필요

// 구조체 복사
Point p5 = p4;  // 멤버별 복사 (얕은 복사)

// 구조체 포인터
Point *ptr = &p4;
printf("%d %d\n", ptr->x, ptr->y);
printf("%d %d\n", (*ptr).x, (*ptr).y);  // 동일
```

### 8.2 중첩 구조체

```c
typedef struct {
    char name[50];
    int age;
} Person;

typedef struct {
    Person person;
    char employee_id[10];
    double salary;
} Employee;

// 초기화
Employee emp = {
    {"John Doe", 30},
    "E001",
    50000.0
};

// 접근
printf("Name: %s\n", emp.person.name);
printf("Age: %d\n", emp.person.age);
printf("Salary: %.2f\n", emp.salary);

// 포인터를 포함하는 구조체
typedef struct {
    char *name;
    int age;
} PersonPtr;

PersonPtr person;
person.name = malloc(50);
strcpy(person.name, "Alice");
person.age = 25;
// 사용 후
free(person.name);
```

### 8.3 구조체 배열

```c
typedef struct {
    char title[100];
    char author[50];
    int year;
} Book;

// 배열 선언 및 초기화
Book library[3] = {
    {"C Programming", "Dennis Ritchie", 1978},
    {"The C++ Programming Language", "Bjarne Stroustrup", 1985},
    {"The Rust Programming Language", "Steve Klabnik", 2018}
};

// 순회
for (int i = 0; i < 3; i++) {
    printf("%s by %s (%d)\n",
           library[i].title, library[i].author, library[i].year);
}

// 동적 할당
Book *books = malloc(10 * sizeof(Book));
for (int i = 0; i < 10; i++) {
    scanf("%s %s %d", books[i].title, books[i].author, &books[i].year);
}
free(books);
```

### 8.4 비트 필드

```c
// 비트 필드로 메모리 절약
struct Flags {
    unsigned int is_active : 1;   // 1비트
    unsigned int is_visible : 1;  // 1비트
    unsigned int priority : 3;    // 3비트 (0-7)
    unsigned int reserved : 27;   // 나머지
};

struct Flags flags = {0};
flags.is_active = 1;
flags.priority = 5;

printf("Size: %zu bytes\n", sizeof(struct Flags));  // 4 bytes

// 실용 예시: RGB 색상
struct Color {
    unsigned int red : 8;
    unsigned int green : 8;
    unsigned int blue : 8;
    unsigned int alpha : 8;
};

struct Color white = {255, 255, 255, 255};
```

### 8.5 공용체 (Union)

```c
// 공용체: 같은 메모리를 여러 타입으로 공유
union Data {
    int i;
    float f;
    char str[20];
};

union Data data;
data.i = 10;
printf("data.i: %d\n", data.i);

data.f = 3.14f;  // 이전 값 덮어씀
printf("data.f: %f\n", data.f);
// printf("data.i: %d\n", data.i);  // 쓰레기 값

printf("Size: %zu\n", sizeof(union Data));  // 가장 큰 멤버 크기

// 태그된 공용체 (권장 패턴)
typedef enum { TYPE_INT, TYPE_FLOAT, TYPE_STRING } DataType;

typedef struct {
    DataType type;
    union {
        int i;
        float f;
        char *str;
    } value;
} TaggedData;

TaggedData data1;
data1.type = TYPE_INT;
data1.value.i = 42;

TaggedData data2;
data2.type = TYPE_FLOAT;
data2.value.f = 3.14f;

// 안전한 접근
void print_data(TaggedData *data) {
    switch (data->type) {
        case TYPE_INT:
            printf("%d\n", data->value.i);
            break;
        case TYPE_FLOAT:
            printf("%f\n", data->value.f);
            break;
        case TYPE_STRING:
            printf("%s\n", data->value.str);
            break;
    }
}
```

### 8.6 구조체 정렬과 패딩

```c
#include <stddef.h>

// 비최적화 구조체
struct Bad {
    char c;      // 1 byte
    // 3 bytes padding
    int i;       // 4 bytes
    char d;      // 1 byte
    // 3 bytes padding
};  // 총 12 bytes

// 최적화 구조체
struct Good {
    int i;       // 4 bytes
    char c;      // 1 byte
    char d;      // 1 byte
    // 2 bytes padding
};  // 총 8 bytes

printf("sizeof(Bad): %zu\n", sizeof(struct Bad));    // 12
printf("sizeof(Good): %zu\n", sizeof(struct Good));  // 8

// offsetof로 멤버 오프셋 확인
printf("Offset of c: %zu\n", offsetof(struct Bad, c));  // 0
printf("Offset of i: %zu\n", offsetof(struct Bad, i));  // 4
printf("Offset of d: %zu\n", offsetof(struct Bad, d));  // 8

// 패킹 (컴파일러 확장)
#pragma pack(push, 1)
struct Packed {
    char c;
    int i;
    char d;
};  // 총 6 bytes (패딩 없음)
#pragma pack(pop)

// 주의: 패킹된 구조체는 정렬되지 않은 접근으로 성능 저하 가능
```

### 8.7 익명 구조체와 공용체 (C11)

```c
typedef struct {
    int id;
    union {
        struct {
            char *name;
            int age;
        };  // 익명 구조체
        struct {
            char *company;
            double revenue;
        };  // 익명 구조체
    };  // 익명 공용체
} Entity;

Entity e;
e.id = 1;
e.name = "Alice";  // 직접 접근 가능
e.age = 30;
```

---

## 9. 동적 메모리 할당

### 9.1 malloc, calloc, realloc, free

```c
#include <stdlib.h>

// malloc - 초기화되지 않은 메모리 할당
int *arr = (int *)malloc(10 * sizeof(int));
if (arr == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
}

// 사용
for (int i = 0; i < 10; i++) {
    arr[i] = i;
}

// 해제
free(arr);
arr = NULL;  // 댕글링 포인터 방지

// calloc - 0으로 초기화된 메모리 할당
int *zeros = (int *)calloc(10, sizeof(int));
// 모든 원소가 0

free(zeros);

// realloc - 메모리 재할당
int *dynamic = (int *)malloc(5 * sizeof(int));
for (int i = 0; i < 5; i++) {
    dynamic[i] = i;
}

// 10개로 확장
int *temp = (int *)realloc(dynamic, 10 * sizeof(int));
if (temp == NULL) {
    free(dynamic);  // 기존 메모리는 유지됨
    exit(EXIT_FAILURE);
}
dynamic = temp;

// 기존 데이터 유지, 새 공간 초기화
for (int i = 5; i < 10; i++) {
    dynamic[i] = i;
}

free(dynamic);

// realloc(ptr, 0)는 free(ptr)와 동일
// realloc(NULL, size)는 malloc(size)와 동일
```

### 9.2 동적 2차원 배열

```c
// 방법 1: 포인터 배열
int rows = 3, cols = 4;
int **matrix = (int **)malloc(rows * sizeof(int *));
for (int i = 0; i < rows; i++) {
    matrix[i] = (int *)malloc(cols * sizeof(int));
}

// 사용
for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
        matrix[i][j] = i * cols + j;
    }
}

// 해제
for (int i = 0; i < rows; i++) {
    free(matrix[i]);
}
free(matrix);

// 방법 2: 연속된 메모리 (더 효율적)
int *data = (int *)malloc(rows * cols * sizeof(int));
int **matrix2 = (int **)malloc(rows * sizeof(int *));
for (int i = 0; i < rows; i++) {
    matrix2[i] = data + i * cols;
}

// 사용 (동일)
matrix2[1][2] = 100;

// 해제
free(data);
free(matrix2);

// 방법 3: 1차원 배열로 시뮬레이션
int *flat = (int *)malloc(rows * cols * sizeof(int));
#define INDEX(i, j) ((i) * cols + (j))
flat[INDEX(1, 2)] = 100;
free(flat);
```

### 9.3 메모리 누수와 방지

```c
// 나쁜 예: 메모리 누수
void leak_example(void) {
    int *ptr = malloc(100 * sizeof(int));
    // free 없음!
}  // 메모리 누수

// 좋은 예: RAII 패턴 모방
typedef struct {
    void *data;
    size_t size;
} Buffer;

Buffer* create_buffer(size_t size) {
    Buffer *buf = malloc(sizeof(Buffer));
    if (buf == NULL) return NULL;

    buf->data = malloc(size);
    if (buf->data == NULL) {
        free(buf);
        return NULL;
    }
    buf->size = size;
    return buf;
}

void destroy_buffer(Buffer *buf) {
    if (buf != NULL) {
        free(buf->data);
        free(buf);
    }
}

// 사용
Buffer *buf = create_buffer(1024);
if (buf != NULL) {
    // 사용
    destroy_buffer(buf);
}

// alloca - 스택 할당 (비표준, 주의 필요)
#include <alloca.h>
void use_alloca(void) {
    int *arr = (int *)alloca(10 * sizeof(int));
    // 함수 종료 시 자동 해제 (free 불필요)
}  // 하지만 스택 오버플로우 위험
```

### 9.4 메모리 디버깅

```c
// Valgrind 사용
// $ gcc -g program.c -o program
// $ valgrind --leak-check=full ./program

// AddressSanitizer 사용
// $ gcc -fsanitize=address -g program.c -o program
// $ ./program

// 커스텀 메모리 추적
#ifdef DEBUG_MEMORY
static size_t total_allocated = 0;

void* debug_malloc(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr != NULL) {
        total_allocated += size;
        printf("MALLOC: %zu bytes at %p (%s:%d)\n", size, ptr, file, line);
    }
    return ptr;
}

void debug_free(void *ptr, const char *file, int line) {
    printf("FREE: %p (%s:%d)\n", ptr, file, line);
    free(ptr);
}

#define malloc(size) debug_malloc(size, __FILE__, __LINE__)
#define free(ptr) debug_free(ptr, __FILE__, __LINE__)
#endif

// 메모리 풀 패턴
typedef struct {
    void *pool;
    size_t block_size;
    size_t num_blocks;
    size_t used;
} MemoryPool;

MemoryPool* create_pool(size_t block_size, size_t num_blocks) {
    MemoryPool *pool = malloc(sizeof(MemoryPool));
    pool->pool = malloc(block_size * num_blocks);
    pool->block_size = block_size;
    pool->num_blocks = num_blocks;
    pool->used = 0;
    return pool;
}

void* pool_alloc(MemoryPool *pool) {
    if (pool->used >= pool->num_blocks) {
        return NULL;
    }
    void *ptr = (char *)pool->pool + pool->used * pool->block_size;
    pool->used++;
    return ptr;
}

void destroy_pool(MemoryPool *pool) {
    free(pool->pool);
    free(pool);
}
```

---

## 10. 파일 입출력

### 10.1 텍스트 파일 입출력

```c
#include <stdio.h>

// 파일 쓰기
FILE *fp = fopen("output.txt", "w");
if (fp == NULL) {
    perror("Error opening file");
    return 1;
}

fprintf(fp, "Hello, File!\n");
fprintf(fp, "Number: %d\n", 42);
fclose(fp);

// 파일 읽기
fp = fopen("input.txt", "r");
if (fp == NULL) {
    perror("Error opening file");
    return 1;
}

char line[256];
while (fgets(line, sizeof(line), fp) != NULL) {
    printf("%s", line);
}
fclose(fp);

// fscanf로 형식화된 읽기
fp = fopen("data.txt", "r");
int num;
char str[50];
while (fscanf(fp, "%d %s", &num, str) == 2) {
    printf("Read: %d %s\n", num, str);
}
fclose(fp);

// 파일 모드
// "r"  - 읽기
// "w"  - 쓰기 (기존 내용 삭제)
// "a"  - 추가 (파일 끝에 쓰기)
// "r+" - 읽기/쓰기
// "w+" - 읽기/쓰기 (기존 내용 삭제)
// "a+" - 읽기/추가
```

### 10.2 바이너리 파일 입출력

```c
typedef struct {
    int id;
    char name[50];
    double salary;
} Employee;

// 바이너리 쓰기
FILE *fp = fopen("employees.dat", "wb");
if (fp == NULL) {
    perror("Error");
    return 1;
}

Employee emp = {1, "John Doe", 50000.0};
fwrite(&emp, sizeof(Employee), 1, fp);

Employee employees[] = {
    {2, "Jane Smith", 60000.0},
    {3, "Bob Johnson", 55000.0}
};
fwrite(employees, sizeof(Employee), 2, fp);
fclose(fp);

// 바이너리 읽기
fp = fopen("employees.dat", "rb");
if (fp == NULL) {
    perror("Error");
    return 1;
}

Employee read_emp;
while (fread(&read_emp, sizeof(Employee), 1, fp) == 1) {
    printf("ID: %d, Name: %s, Salary: %.2f\n",
           read_emp.id, read_emp.name, read_emp.salary);
}
fclose(fp);
```

### 10.3 파일 위치 제어

```c
FILE *fp = fopen("data.txt", "r+");

// ftell - 현재 위치
long pos = ftell(fp);
printf("Current position: %ld\n", pos);

// fseek - 위치 이동
fseek(fp, 0, SEEK_SET);  // 파일 시작
fseek(fp, 0, SEEK_END);  // 파일 끝
fseek(fp, -10, SEEK_CUR);  // 현재 위치에서 -10

// rewind - 파일 시작으로
rewind(fp);

// fgetpos, fsetpos - 위치 저장/복원
fpos_t position;
fgetpos(fp, &position);
// ... 다른 작업
fsetpos(fp, &position);

// 파일 크기 구하기
fseek(fp, 0, SEEK_END);
long size = ftell(fp);
rewind(fp);
printf("File size: %ld bytes\n", size);

fclose(fp);
```

### 10.4 버퍼링

```c
// 버퍼링 모드 설정
FILE *fp = fopen("output.txt", "w");

// 전체 버퍼링
setvbuf(fp, NULL, _IOFBF, 4096);

// 라인 버퍼링
setvbuf(fp, NULL, _IOLBF, 1024);

// 버퍼링 없음
setvbuf(fp, NULL, _IONBF, 0);

// 커스텀 버퍼
char buffer[8192];
setvbuf(fp, buffer, _IOFBF, sizeof(buffer));

// fflush - 버퍼 강제 비우기
fprintf(fp, "Important data");
fflush(fp);  // 즉시 디스크에 쓰기

fclose(fp);
```

### 10.5 에러 처리

```c
FILE *fp = fopen("file.txt", "r");

// feof - 파일 끝 확인
while (!feof(fp)) {
    int c = fgetc(fp);
    if (c == EOF) {
        if (feof(fp)) {
            printf("End of file reached\n");
        } else if (ferror(fp)) {
            perror("Read error");
        }
        break;
    }
    putchar(c);
}

// clearerr - 에러 플래그 클리어
clearerr(fp);

fclose(fp);

// perror - 에러 메시지 출력
if (fopen("nonexistent.txt", "r") == NULL) {
    perror("fopen");  // "fopen: No such file or directory"
}

// 파일 존재 확인
#include <sys/stat.h>
struct stat st;
if (stat("file.txt", &st) == 0) {
    printf("File exists, size: %ld bytes\n", st.st_size);
}
```

### 10.6 고급 파일 작업

```c
// 파일 삭제
#include <stdio.h>
if (remove("temp.txt") == 0) {
    printf("File deleted successfully\n");
} else {
    perror("remove");
}

// 파일 이름 변경
if (rename("old.txt", "new.txt") == 0) {
    printf("File renamed successfully\n");
} else {
    perror("rename");
}

// 임시 파일
FILE *temp = tmpfile();  // 자동으로 삭제됨
fprintf(temp, "Temporary data\n");
rewind(temp);
char buffer[100];
fgets(buffer, sizeof(buffer), temp);
fclose(temp);  // 파일 자동 삭제

// 임시 파일 이름 생성
char temp_name[L_tmpnam];
tmpnam(temp_name);
FILE *fp = fopen(temp_name, "w+");
// 사용 후
fclose(fp);
remove(temp_name);

// 메모리 스트림 (GNU 확장, POSIX)
#include <stdio.h>
char *buffer;
size_t size;
FILE *memstream = open_memstream(&buffer, &size);
fprintf(memstream, "Hello, ");
fprintf(memstream, "Memory!\n");
fclose(memstream);
printf("%s", buffer);  // "Hello, Memory!\n"
free(buffer);
```

---

## 11. 전처리기

### 11.1 매크로 정의

```c
// 단순 매크로
#define PI 3.14159
#define MAX_SIZE 1000
#define GREETING "Hello, World!"

// 함수형 매크로
#define SQUARE(x) ((x) * (x))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// 주의: 괄호 필수
#define BAD_SQUARE(x) x * x
int result = BAD_SQUARE(2 + 3);  // 2 + 3 * 2 + 3 = 11 (예상: 25)

// 다중 줄 매크로
#define SWAP(a, b, type) do { \
    type temp = a;             \
    a = b;                     \
    b = temp;                  \
} while (0)

// do-while(0)으로 감싸는 이유
if (condition)
    SWAP(x, y, int);  // 세미콜론 필요
else
    other_code();

// 가변 인자 매크로
#define DEBUG_PRINT(fmt, ...) \
    fprintf(stderr, "%s:%d: " fmt "\n", __FILE__, __LINE__, __VA_ARGS__)

DEBUG_PRINT("Value: %d", 42);
// 출력: "file.c:10: Value: 42"

// ##로 토큰 결합
#define CONCAT(a, b) a##b
int CONCAT(var, 123) = 10;  // var123

// #로 문자열화
#define STRINGIFY(x) #x
printf("%s\n", STRINGIFY(Hello));  // "Hello"

// 매크로 해제
#undef PI
```

### 11.2 조건부 컴파일

```c
// #ifdef, #ifndef
#define DEBUG

#ifdef DEBUG
    printf("Debug mode\n");
#endif

#ifndef RELEASE
    printf("Not in release mode\n");
#endif

// #if, #elif, #else
#define VERSION 3

#if VERSION == 1
    printf("Version 1\n");
#elif VERSION == 2
    printf("Version 2\n");
#elif VERSION == 3
    printf("Version 3\n");
#else
    printf("Unknown version\n");
#endif

// defined 연산자
#if defined(DEBUG) && !defined(NDEBUG)
    printf("Debug assertions enabled\n");
#endif

// 복잡한 조건
#if (VERSION >= 2) && (defined(FEATURE_X) || defined(FEATURE_Y))
    // ...
#endif

// #error, #warning
#ifndef REQUIRED_DEFINE
    #error "REQUIRED_DEFINE must be defined"
#endif

#if MAX_SIZE < 100
    #warning "MAX_SIZE is very small"
#endif
```

### 11.3 헤더 가드

```c
// header.h
#ifndef HEADER_H
#define HEADER_H

// 헤더 내용
void function(void);

#endif  // HEADER_H

// 또는 (비표준이지만 널리 지원됨)
#pragma once

// 헤더 내용
void function(void);
```

### 11.4 미리 정의된 매크로

```c
// 표준 매크로
printf("File: %s\n", __FILE__);      // 현재 파일명
printf("Line: %d\n", __LINE__);      // 현재 줄 번호
printf("Date: %s\n", __DATE__);      // 컴파일 날짜
printf("Time: %s\n", __TIME__);      // 컴파일 시간
printf("Function: %s\n", __func__);  // 현재 함수명 (C99)

// C 표준 버전
#if __STDC_VERSION__ >= 199901L
    printf("C99 or later\n");
#endif

#if __STDC_VERSION__ >= 201112L
    printf("C11 or later\n");
#endif

#if __STDC_VERSION__ >= 201710L
    printf("C17/C18 or later\n");
#endif

// 컴파일러별 매크로
#ifdef __GNUC__
    printf("GCC version: %d.%d.%d\n",
           __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#endif

#ifdef _MSC_VER
    printf("MSVC version: %d\n", _MSC_VER);
#endif

#ifdef __clang__
    printf("Clang\n");
#endif

// OS 감지
#ifdef _WIN32
    printf("Windows\n");
#endif

#ifdef __linux__
    printf("Linux\n");
#endif

#ifdef __APPLE__
    printf("macOS\n");
#endif
```

### 11.5 #include

```c
// 표준 라이브러리 헤더
#include <stdio.h>
#include <stdlib.h>

// 사용자 정의 헤더
#include "myheader.h"

// 조건부 include
#ifdef USE_CUSTOM_MATH
    #include "custom_math.h"
#else
    #include <math.h>
#endif

// 경로 지정
#include "../../common/utils.h"

// 컴파일러 옵션으로 경로 추가
// gcc -I/path/to/headers program.c
```

### 11.6 실전 매크로 예제

```c
// 최소/최대 안전 매크로
#define SAFE_MAX(a, b) ({ \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a > _b ? _a : _b; \
})

// 배열 크기
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

// 구조체 멤버 오프셋
#define OFFSETOF(type, member) ((size_t)&((type *)0)->member)

// 컨테이너 포인터 얻기
#define CONTAINER_OF(ptr, type, member) ({ \
    const __typeof__(((type *)0)->member) *__mptr = (ptr); \
    (type *)((char *)__mptr - OFFSETOF(type, member)); \
})

// 비트 조작
#define BIT(n) (1U << (n))
#define SET_BIT(val, n) ((val) |= BIT(n))
#define CLEAR_BIT(val, n) ((val) &= ~BIT(n))
#define TOGGLE_BIT(val, n) ((val) ^= BIT(n))
#define CHECK_BIT(val, n) (((val) & BIT(n)) != 0)

// 범위 체크
#define IN_RANGE(val, min, max) ((val) >= (min) && (val) <= (max))

// 정렬
#define ALIGN_UP(val, align) (((val) + (align) - 1) & ~((align) - 1))
#define ALIGN_DOWN(val, align) ((val) & ~((align) - 1))

// 색상 출력 (ANSI)
#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_RESET "\x1b[0m"
#define PRINT_ERROR(fmt, ...) \
    printf(ANSI_RED "ERROR: " fmt ANSI_RESET "\n", ##__VA_ARGS__)
```

---

## 12. 고급 포인터

### 12.1 함수 포인터 심화

```c
// 함수 포인터 typedef
typedef int (*CompareFn)(const void *, const void *);
typedef void (*CallbackFn)(void *);

// 비교 함수
int int_compare(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

// qsort 사용
int arr[] = {5, 2, 8, 1, 9};
qsort(arr, 5, sizeof(int), int_compare);

// 함수 포인터 구조체
typedef struct {
    const char *name;
    void (*init)(void);
    void (*process)(void *);
    void (*cleanup)(void);
} Module;

void module_a_init(void) { printf("Module A init\n"); }
void module_a_process(void *data) { printf("Module A process\n"); }
void module_a_cleanup(void) { printf("Module A cleanup\n"); }

Module module_a = {
    "ModuleA",
    module_a_init,
    module_a_process,
    module_a_cleanup
};

// 플러그인 시스템
Module *modules[] = {&module_a, &module_b, &module_c};
for (int i = 0; i < 3; i++) {
    modules[i]->init();
    modules[i]->process(NULL);
    modules[i]->cleanup();
}
```

### 12.2 포인터와 const 심화

```c
int x = 10, y = 20;

// 레벨 1: 데이터 상수
const int *p1 = &x;
int const *p2 = &x;  // 동일
// *p1 = 20;  // 에러
p1 = &y;      // OK

// 레벨 2: 포인터 상수
int * const p3 = &x;
*p3 = 30;     // OK
// p3 = &y;   // 에러

// 레벨 3: 둘 다 상수
const int * const p4 = &x;
// *p4 = 40;  // 에러
// p4 = &y;   // 에러

// 포인터의 포인터와 const
const int **pp1;
int * const *pp2;
int ** const pp3;
const int * const *pp4;
const int ** const pp5;
const int * const * const pp6;

// 함수 매개변수에서의 const
void print_string(const char *str) {
    // str[0] = 'X';  // 에러
    printf("%s\n", str);
}

// const 캐스팅 (위험!)
const int ci = 100;
int *p = (int *)&ci;
*p = 200;  // 미정의 동작!
```

### 12.3 restrict 포인터 (C99)

```c
// restrict: 포인터가 유일한 접근 수단임을 보장
// 컴파일러 최적화 향상

void copy_array(int * restrict dest,
                const int * restrict src,
                size_t n) {
    for (size_t i = 0; i < n; i++) {
        dest[i] = src[i];
    }
    // 컴파일러는 dest와 src가 겹치지 않음을 가정
}

// 잘못된 사용 (미정의 동작)
int arr[10];
copy_array(arr + 2, arr, 5);  // 겹침!

// memcpy vs memmove
// memcpy: restrict 사용 (겹치면 안됨)
memcpy(dest, src, n);

// memmove: 겹쳐도 안전
memmove(arr + 2, arr, 5 * sizeof(int));
```

### 12.4 복잡한 포인터 선언 해석

```c
// 오른쪽에서 왼쪽으로, 안에서 밖으로 읽기

int *p;                 // p는 int를 가리키는 포인터
int **pp;               // pp는 int 포인터를 가리키는 포인터
int *ap[10];            // ap는 10개의 int 포인터 배열
int (*pa)[10];          // pa는 10개 int 배열을 가리키는 포인터
int (*pf)(int, int);    // pf는 (int, int)를 받아 int를 반환하는 함수 포인터
int *(*pfa)[10];        // pfa는 10개의 int 포인터 배열을 가리키는 포인터
int (*apf[10])(int);    // apf는 10개의 (int -> int) 함수 포인터 배열
int *(*fp)(int *, int); // fp는 (int *, int)를 받아 int *를 반환하는 함수 포인터

// cdecl 도구 사용 권장
// https://cdecl.org/
// "int *(*fp)(int *, int)" ->
// "declare fp as pointer to function (pointer to int, int) returning pointer to int"
```

### 12.5 포인터 배열 vs 다차원 배열

```c
// 포인터 배열 (비연속 메모리)
char *names[] = {
    "Alice",
    "Bob",
    "Charlie"
};
// names는 char * 3개의 배열
// 각 문자열은 다른 위치에 있을 수 있음

// 2차원 문자 배열 (연속 메모리)
char names2[][10] = {
    "Alice",
    "Bob",
    "Charlie"
};
// 30바이트 연속 메모리

// 함수 매개변수
void func1(char *arr[]);        // 포인터 배열
void func2(char arr[][10]);     // 2차원 배열
void func3(char (*arr)[10]);    // 10개 char 배열을 가리키는 포인터

// 2D 배열 포인터
int matrix[3][4];
int (*p)[4] = matrix;  // 4개 int 배열을 가리키는 포인터
printf("%d\n", p[0][0]);  // matrix[0][0]
printf("%d\n", (*(p + 1))[2]);  // matrix[1][2]
```

### 12.6 Opaque 포인터 패턴

```c
// mylib.h
typedef struct MyStruct MyStruct;  // 불완전 타입

MyStruct* create_mystruct(void);
void destroy_mystruct(MyStruct *obj);
void mystruct_operation(MyStruct *obj);

// mylib.c
struct MyStruct {
    int private_data;
    char *secret;
};  // 구현 숨김

MyStruct* create_mystruct(void) {
    MyStruct *obj = malloc(sizeof(MyStruct));
    obj->private_data = 0;
    obj->secret = strdup("hidden");
    return obj;
}

void destroy_mystruct(MyStruct *obj) {
    if (obj) {
        free(obj->secret);
        free(obj);
    }
}

// 사용자는 내부 구조를 알 수 없음
// 캡슐화 제공
```

---

## 13. 비트 연산

### 13.1 비트 조작 기법

```c
// 비트 설정, 해제, 토글, 확인
unsigned int flags = 0;

#define BIT(n) (1U << (n))
#define SET_BIT(val, n) ((val) |= BIT(n))
#define CLEAR_BIT(val, n) ((val) &= ~BIT(n))
#define TOGGLE_BIT(val, n) ((val) ^= BIT(n))
#define CHECK_BIT(val, n) (((val) & BIT(n)) != 0)

SET_BIT(flags, 3);      // flags |= (1 << 3)
CLEAR_BIT(flags, 2);    // flags &= ~(1 << 2)
TOGGLE_BIT(flags, 5);   // flags ^= (1 << 5)
if (CHECK_BIT(flags, 3)) {
    printf("Bit 3 is set\n");
}

// 여러 비트 조작
#define BITMASK(n) ((1U << (n)) - 1)
unsigned int mask = BITMASK(4);  // 0b1111

// 범위 설정
void set_bits(unsigned int *val, int pos, int n) {
    unsigned int mask = ((1U << n) - 1) << pos;
    *val |= mask;
}

// 범위 해제
void clear_bits(unsigned int *val, int pos, int n) {
    unsigned int mask = ((1U << n) - 1) << pos;
    *val &= ~mask;
}

// 범위 추출
unsigned int extract_bits(unsigned int val, int pos, int n) {
    return (val >> pos) & ((1U << n) - 1);
}

// 범위 설정
void write_bits(unsigned int *val, int pos, int n, unsigned int data) {
    unsigned int mask = ((1U << n) - 1) << pos;
    *val = (*val & ~mask) | ((data << pos) & mask);
}
```

### 13.2 비트 카운팅

```c
// 설정된 비트 개수 (Population Count)
int popcount(unsigned int x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

// Brian Kernighan's 알고리즘
int popcount_fast(unsigned int x) {
    int count = 0;
    while (x) {
        x &= x - 1;  // 가장 낮은 1 비트 제거
        count++;
    }
    return count;
}

// GCC 내장 함수
int count = __builtin_popcount(x);
int count64 = __builtin_popcountll(x);

// 선행 0 개수 (Count Leading Zeros)
int clz(unsigned int x) {
    if (x == 0) return 32;
    int count = 0;
    while ((x & 0x80000000) == 0) {
        count++;
        x <<= 1;
    }
    return count;
}

// GCC 내장
int leading_zeros = __builtin_clz(x);

// 후행 0 개수
int trailing_zeros = __builtin_ctz(x);
```

### 13.3 비트 트릭

```c
// 2의 거듭제곱 확인
bool is_power_of_two(unsigned int x) {
    return x != 0 && (x & (x - 1)) == 0;
}

// 다음 2의 거듭제곱
unsigned int next_power_of_two(unsigned int x) {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}

// 절댓값 (부호 있는 정수)
int abs_value(int x) {
    int mask = x >> 31;  // 음수면 -1, 양수면 0
    return (x + mask) ^ mask;
}

// 최소값 (분기 없이)
int min_branchless(int a, int b) {
    return b ^ ((a ^ b) & -(a < b));
}

// 최대값
int max_branchless(int a, int b) {
    return a ^ ((a ^ b) & -(a < b));
}

// 두 값 교환 (XOR swap, 권장하지 않음 - 가독성 낮음)
void xor_swap(int *a, int *b) {
    if (a != b) {  // 같은 주소면 0이 됨
        *a ^= *b;
        *b ^= *a;
        *a ^= *b;
    }
}

// 부호 반전
int negate(int x) {
    return ~x + 1;
}

// 비트 반전
unsigned int reverse_bits(unsigned int x) {
    x = ((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4);
    x = ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8);
    x = (x >> 16) | (x << 16);
    return x;
}
```

### 13.4 비트 필드 활용

```c
// 권한 시스템
#define PERM_READ    (1 << 0)  // 0b001
#define PERM_WRITE   (1 << 1)  // 0b010
#define PERM_EXECUTE (1 << 2)  // 0b100

unsigned int permissions = 0;
permissions |= PERM_READ | PERM_WRITE;  // 읽기, 쓰기 부여

if (permissions & PERM_READ) {
    printf("Can read\n");
}

if ((permissions & (PERM_READ | PERM_WRITE)) == (PERM_READ | PERM_WRITE)) {
    printf("Can read and write\n");
}

permissions &= ~PERM_WRITE;  // 쓰기 권한 제거

// 상태 플래그
typedef enum {
    STATE_IDLE      = 0,
    STATE_RUNNING   = (1 << 0),
    STATE_PAUSED    = (1 << 1),
    STATE_ERROR     = (1 << 2),
    STATE_COMPLETED = (1 << 3)
} State;

State current_state = STATE_IDLE;
current_state |= STATE_RUNNING;

// IP 주소 조작
typedef union {
    uint32_t addr;
    uint8_t octets[4];
} IPv4Address;

IPv4Address ip;
ip.octets[0] = 192;
ip.octets[1] = 168;
ip.octets[2] = 1;
ip.octets[3] = 1;
printf("IP: 0x%08X\n", ip.addr);
```

### 13.5 비트맵

```c
// 간단한 비트맵
#define BITMAP_SIZE 1024
#define BITMAP_BYTES ((BITMAP_SIZE + 7) / 8)

unsigned char bitmap[BITMAP_BYTES] = {0};

void set_bit(unsigned char *bitmap, int index) {
    bitmap[index / 8] |= (1 << (index % 8));
}

void clear_bit(unsigned char *bitmap, int index) {
    bitmap[index / 8] &= ~(1 << (index % 8));
}

int test_bit(unsigned char *bitmap, int index) {
    return (bitmap[index / 8] & (1 << (index % 8))) != 0;
}

// 사용 예: 소수 찾기 (에라토스테네스의 체)
void sieve_of_eratosthenes(int n) {
    int bitmap_size = (n + 7) / 8;
    unsigned char *primes = calloc(bitmap_size, 1);

    // 0과 1은 소수 아님
    set_bit(primes, 0);
    set_bit(primes, 1);

    for (int i = 2; i * i <= n; i++) {
        if (!test_bit(primes, i)) {
            for (int j = i * i; j <= n; j += i) {
                set_bit(primes, j);
            }
        }
    }

    printf("Primes up to %d: ", n);
    for (int i = 2; i <= n; i++) {
        if (!test_bit(primes, i)) {
            printf("%d ", i);
        }
    }
    printf("\n");

    free(primes);
}
```

---

## 14. 메모리 관리

### 14.1 메모리 레이아웃

```c
#include <stdio.h>
#include <stdlib.h>

int global_var = 10;           // 데이터 세그먼트
static int static_var = 20;    // 데이터 세그먼트
const int const_var = 30;      // 읽기 전용 데이터

void function() {
    int local_var = 40;        // 스택
    static int static_local = 50;  // 데이터 세그먼트
    int *heap_var = malloc(sizeof(int));  // 힙
    *heap_var = 60;

    printf("Global: %p\n", (void*)&global_var);
    printf("Static: %p\n", (void*)&static_var);
    printf("Const: %p\n", (void*)&const_var);
    printf("Local: %p\n", (void*)&local_var);
    printf("Static Local: %p\n", (void*)&static_local);
    printf("Heap: %p\n", (void*)heap_var);

    free(heap_var);
}

/*
메모리 레이아웃 (낮은 주소 → 높은 주소):
1. 텍스트 세그먼트 (코드)
2. 읽기 전용 데이터
3. 초기화된 데이터 (.data)
4. 초기화되지 않은 데이터 (.bss)
5. 힙 (↓ 아래로 성장)
6. 스택 (↑ 위로 성장)
*/
```

### 14.2 스택 vs 힙

```c
// 스택 할당
void stack_allocation() {
    int arr[1000];  // 스택에 4000바이트
    // 빠르지만 크기 제한, 함수 종료 시 자동 해제
}

// 힙 할당
void heap_allocation() {
    int *arr = malloc(1000 * sizeof(int));
    // 느리지만 크기 유연, 수동 해제 필요
    free(arr);
}

// VLA (가변 길이 배열) - 스택
void vla_example(int n) {
    int arr[n];  // C99, 스택 오버플로우 위험
}
```

### 14.3 메모리 누수 탐지

```c
// 메모리 누수 예제
void memory_leak() {
    int *ptr = malloc(100 * sizeof(int));
    // free 없음!
}  // 누수!

// 이중 해제
void double_free() {
    int *ptr = malloc(sizeof(int));
    free(ptr);
    // free(ptr);  // 위험! 미정의 동작
}

// 해제 후 사용 (Use-After-Free)
void use_after_free() {
    int *ptr = malloc(sizeof(int));
    *ptr = 10;
    free(ptr);
    // printf("%d\n", *ptr);  // 위험! 댕글링 포인터
}

// 올바른 패턴
void correct_pattern() {
    int *ptr = malloc(sizeof(int));
    if (ptr == NULL) {
        return;
    }

    *ptr = 10;
    // 사용...

    free(ptr);
    ptr = NULL;  // 댕글링 포인터 방지
}
```

### 14.4 메모리 정렬

```c
#include <stddef.h>
#include <stdalign.h>  // C11

// 구조체 정렬
struct Aligned {
    char c;      // 1 byte
    // 3 bytes padding
    int i;       // 4 bytes
    char d;      // 1 byte
    // 3 bytes padding
};  // 총 12 bytes

// 최적화된 구조체
struct Optimized {
    int i;       // 4 bytes
    char c;      // 1 byte
    char d;      // 1 byte
    // 2 bytes padding
};  // 총 8 bytes

// 정렬 지정
struct alignas(16) AlignedTo16 {
    int x;
};

// 정렬 확인
printf("Alignment of int: %zu\n", alignof(int));
printf("Size: %zu, Alignment: %zu\n",
       sizeof(struct Aligned), alignof(struct Aligned));
```

### 14.5 커스텀 메모리 할당자

```c
// 간단한 메모리 풀
#define POOL_SIZE 1024
#define BLOCK_SIZE 32

typedef struct {
    unsigned char pool[POOL_SIZE];
    int used[POOL_SIZE / BLOCK_SIZE];
} MemoryPool;

MemoryPool* create_pool() {
    MemoryPool *pool = malloc(sizeof(MemoryPool));
    memset(pool->used, 0, sizeof(pool->used));
    return pool;
}

void* pool_alloc(MemoryPool *pool) {
    for (int i = 0; i < POOL_SIZE / BLOCK_SIZE; i++) {
        if (!pool->used[i]) {
            pool->used[i] = 1;
            return &pool->pool[i * BLOCK_SIZE];
        }
    }
    return NULL;
}

void pool_free(MemoryPool *pool, void *ptr) {
    int index = ((unsigned char*)ptr - pool->pool) / BLOCK_SIZE;
    if (index >= 0 && index < POOL_SIZE / BLOCK_SIZE) {
        pool->used[index] = 0;
    }
}

void destroy_pool(MemoryPool *pool) {
    free(pool);
}
```

---

## 15. 컴파일과 링킹

### 15.1 컴파일 단계

```bash
# 1. 전처리 (Preprocessing)
gcc -E source.c -o source.i
# - #include 확장
# - 매크로 치환
# - 조건부 컴파일

# 2. 컴파일 (Compilation)
gcc -S source.i -o source.s
# - C 코드 → 어셈블리

# 3. 어셈블 (Assembly)
gcc -c source.s -o source.o
# - 어셈블리 → 기계어 (오브젝트 파일)

# 4. 링킹 (Linking)
gcc source.o -o program
# - 오브젝트 파일 결합
# - 라이브러리 링크
# - 실행 파일 생성

# 한번에
gcc -Wall -Wextra -O2 source.c -o program
```

### 15.2 헤더 파일과 소스 파일 분리

```c
// math_utils.h
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

int add(int a, int b);
int multiply(int a, int b);

#endif

// math_utils.c
#include "math_utils.h"

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

// main.c
#include <stdio.h>
#include "math_utils.h"

int main() {
    printf("%d\n", add(3, 4));
    printf("%d\n", multiply(3, 4));
    return 0;
}

// 컴파일
// gcc -c math_utils.c -o math_utils.o
// gcc -c main.c -o main.o
// gcc math_utils.o main.o -o program
```

### 15.3 정적 라이브러리 vs 동적 라이브러리

```bash
# 정적 라이브러리 (.a)
gcc -c math_utils.c -o math_utils.o
ar rcs libmath.a math_utils.o
gcc main.c -L. -lmath -o program

# 동적 라이브러리 (.so / .dll)
gcc -fPIC -c math_utils.c -o math_utils.o
gcc -shared -o libmath.so math_utils.o
gcc main.c -L. -lmath -o program
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
./program

# 정적 vs 동적
# 정적: 실행 파일에 포함, 크기 큼, 의존성 없음
# 동적: 런타임 링크, 크기 작음, 라이브러리 공유
```

### 15.4 Makefile

```makefile
# Makefile
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2
LDFLAGS = -lm

SRCS = main.c math_utils.c string_utils.c
OBJS = $(SRCS:.c=.o)
TARGET = program

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
```

---

## 16. 표준 라이브러리

### 16.1 stdio.h

```c
#include <stdio.h>

// 입출력
printf("Hello, %s!\n", "World");
fprintf(stderr, "Error: %d\n", errno);
sprintf(buffer, "Value: %d", 42);
snprintf(buffer, sizeof(buffer), "Safe: %d", 42);

scanf("%d", &num);
fscanf(file, "%d %s", &num, str);
sscanf("42 hello", "%d %s", &num, str);

// 파일 입출력
FILE *fp = fopen("file.txt", "r");
fgets(line, sizeof(line), fp);
fputs("Hello\n", fp);
fread(buffer, 1, size, fp);
fwrite(buffer, 1, size, fp);
fclose(fp);

// 스트림 위치
fseek(fp, 0, SEEK_SET);
ftell(fp);
rewind(fp);
```

### 16.2 stdlib.h

```c
#include <stdlib.h>

// 메모리
void *malloc(size_t size);
void *calloc(size_t count, size_t size);
void *realloc(void *ptr, size_t size);
void free(void *ptr);

// 문자열 변환
int atoi(const char *str);
long atol(const char *str);
double atof(const char *str);
long strtol(const char *str, char **endptr, int base);

// 난수
srand(time(NULL));
int random = rand() % 100;

// 프로세스
exit(EXIT_SUCCESS);
abort();
int system("ls -l");

// 검색/정렬
qsort(array, count, sizeof(int), compare);
bsearch(&key, array, count, sizeof(int), compare);

// 환경
char *env = getenv("PATH");
```

### 16.3 string.h

```c
#include <string.h>

// 복사
strcpy(dest, src);
strncpy(dest, src, n);
memcpy(dest, src, n);
memmove(dest, src, n);

// 연결
strcat(dest, src);
strncat(dest, src, n);

// 비교
strcmp(s1, s2);
strncmp(s1, s2, n);
memcmp(p1, p2, n);

// 검색
strchr(str, 'c');
strrchr(str, 'c');
strstr(haystack, needle);
strpbrk(str, accept);
strspn(str, accept);
strcspn(str, reject);

// 토큰화
strtok(str, delim);

// 기타
strlen(str);
memset(ptr, value, n);
```

### 16.4 math.h

```c
#include <math.h>

// 거듭제곱/루트
pow(x, y);
sqrt(x);
cbrt(x);

// 삼각함수
sin(x); cos(x); tan(x);
asin(x); acos(x); atan(x);
atan2(y, x);

// 지수/로그
exp(x);
log(x);   // 자연로그
log10(x);
log2(x);

// 올림/내림
ceil(x);
floor(x);
round(x);
trunc(x);

// 절댓값
fabs(x);
abs(n);

// 기타
fmod(x, y);
hypot(x, y);  // sqrt(x^2 + y^2)
```

### 16.5 time.h

```c
#include <time.h>

// 현재 시간
time_t now = time(NULL);

// 시간 구조체
struct tm *local = localtime(&now);
printf("%d-%02d-%02d %02d:%02d:%02d\n",
       local->tm_year + 1900,
       local->tm_mon + 1,
       local->tm_mday,
       local->tm_hour,
       local->tm_min,
       local->tm_sec);

// 포맷팅
char buffer[80];
strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", local);

// 시간 측정
clock_t start = clock();
// 작업...
clock_t end = clock();
double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;

// 고해상도 시간 (POSIX)
struct timespec ts;
clock_gettime(CLOCK_MONOTONIC, &ts);
```

---

## 17. 고급 기법

### 17.1 함수 포인터 배열과 상태 머신

```c
typedef void (*StateFunc)(void);

void state_idle(void) {
    printf("IDLE state\n");
}

void state_running(void) {
    printf("RUNNING state\n");
}

void state_stopped(void) {
    printf("STOPPED state\n");
}

StateFunc states[] = {
    state_idle,
    state_running,
    state_stopped
};

enum State { IDLE, RUNNING, STOPPED };

void run_state_machine() {
    enum State current = IDLE;

    for (int i = 0; i < 10; i++) {
        states[current]();
        // 상태 전이 로직...
    }
}
```

### 17.2 콜백 패턴

```c
typedef int (*CompareFunc)(const void*, const void*);
typedef void (*Callback)(void*);

// 제네릭 정렬
void generic_sort(void *base, size_t nmemb, size_t size, CompareFunc cmp) {
    // 정렬 구현...
}

// 이벤트 시스템
typedef struct {
    Callback callbacks[10];
    int count;
} EventSystem;

void register_callback(EventSystem *es, Callback cb) {
    es->callbacks[es->count++] = cb;
}

void trigger_event(EventSystem *es, void *data) {
    for (int i = 0; i < es->count; i++) {
        es->callbacks[i](data);
    }
}
```

### 17.3 더블 포인터 활용

```c
// 링크드 리스트 노드 삭제
typedef struct Node {
    int data;
    struct Node *next;
} Node;

// 나쁜 방법: 특수 케이스 필요
void delete_bad(Node **head, int value) {
    if (*head == NULL) return;

    if ((*head)->data == value) {
        Node *temp = *head;
        *head = (*head)->next;
        free(temp);
        return;
    }

    Node *curr = *head;
    while (curr->next != NULL) {
        if (curr->next->data == value) {
            Node *temp = curr->next;
            curr->next = curr->next->next;
            free(temp);
            return;
        }
        curr = curr->next;
    }
}

// 좋은 방법: 더블 포인터
void delete_good(Node **head, int value) {
    Node **indirect = head;

    while (*indirect != NULL) {
        if ((*indirect)->data == value) {
            Node *temp = *indirect;
            *indirect = (*indirect)->next;
            free(temp);
            return;
        }
        indirect = &(*indirect)->next;
    }
}
```

### 17.4 X-Macro 패턴

```c
// 에러 코드 정의
#define ERROR_CODES \
    X(OK, 0, "Success") \
    X(INVALID_ARG, 1, "Invalid argument") \
    X(NOT_FOUND, 2, "Not found") \
    X(OUT_OF_MEMORY, 3, "Out of memory")

// enum 생성
#define X(name, code, msg) ERROR_##name = code,
enum ErrorCode {
    ERROR_CODES
};
#undef X

// 문자열 배열 생성
#define X(name, code, msg) msg,
const char *error_messages[] = {
    ERROR_CODES
};
#undef X

// 사용
void print_error(enum ErrorCode err) {
    printf("Error: %s\n", error_messages[err]);
}
```

---

## 18. 최적화와 성능

### 18.1 컴파일러 최적화

```bash
# 최적화 레벨
gcc -O0  # 최적화 없음 (디버깅용)
gcc -O1  # 기본 최적화
gcc -O2  # 더 많은 최적화 (권장)
gcc -O3  # 공격적 최적화
gcc -Os  # 크기 최적화
gcc -Ofast  # 표준 준수 무시하고 최대 성능

# 아키텍처 특화
gcc -march=native  # 현재 CPU에 최적화
gcc -mtune=native

# 링크 타임 최적화 (LTO)
gcc -flto -O3 source.c -o program
```

### 18.2 프로파일링

```c
// gprof 사용
// gcc -pg program.c -o program
// ./program
// gprof program gmon.out > analysis.txt

// 수동 시간 측정
#include <time.h>

clock_t start = clock();
// 측정할 코드
clock_t end = clock();
double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

// 고해상도 타이머
struct timespec start_ts, end_ts;
clock_gettime(CLOCK_MONOTONIC, &start_ts);
// 측정할 코드
clock_gettime(CLOCK_MONOTONIC, &end_ts);
double elapsed = (end_ts.tv_sec - start_ts.tv_sec) +
                 (end_ts.tv_nsec - start_ts.tv_nsec) / 1e9;
```

### 18.3 캐시 친화적 코드

```c
// 나쁜 예: 캐시 미스 많음 (열 우선)
void bad_matrix_sum(int matrix[1000][1000]) {
    for (int j = 0; j < 1000; j++) {
        for (int i = 0; i < 1000; i++) {
            sum += matrix[i][j];  // 비연속 접근
        }
    }
}

// 좋은 예: 캐시 효율적 (행 우선)
void good_matrix_sum(int matrix[1000][1000]) {
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 1000; j++) {
            sum += matrix[i][j];  // 연속 접근
        }
    }
}

// 구조체 정렬로 캐시 라인 최적화
struct CacheFriendly {
    int frequently_used_together[8];  // 32바이트 (캐시 라인 절반)
    // 자주 함께 사용되는 필드를 가까이 배치
} __attribute__((aligned(64)));  // 캐시 라인 크기에 정렬
```

### 18.4 루프 최적화

```c
// 루프 언롤링
// 나쁜 예
for (int i = 0; i < 1000; i++) {
    sum += array[i];
}

// 좋은 예 (수동 언롤링)
for (int i = 0; i < 1000; i += 4) {
    sum += array[i];
    sum += array[i+1];
    sum += array[i+2];
    sum += array[i+3];
}

// 루프 불변 코드 이동
// 나쁜 예
for (int i = 0; i < n; i++) {
    result += array[i] * expensive_function(x);
}

// 좋은 예
int factor = expensive_function(x);
for (int i = 0; i < n; i++) {
    result += array[i] * factor;
}

// restrict 키워드 활용
void vector_add(int * restrict a, int * restrict b,
                int * restrict c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];  // 컴파일러가 최적화 가능
    }
}
```

### 18.5 인라인과 매크로

```c
// inline 함수
static inline int max(int a, int b) {
    return (a > b) ? a : b;
}

// 매크로 (타입 제네릭, 하지만 부작용 주의)
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// __builtin 함수 활용
int leading_zeros = __builtin_clz(x);
int popcount = __builtin_popcount(x);

// 분기 예측 힌트
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

if (unlikely(error_condition)) {
    handle_error();
}
```

---

## 19. 디버깅과 도구

### 19.1 GDB (GNU Debugger)

```bash
# 컴파일 (디버그 심볼 포함)
gcc -g program.c -o program

# GDB 시작
gdb ./program

# 주요 명령어
(gdb) run                 # 프로그램 실행
(gdb) break main          # main에 중단점
(gdb) break file.c:10     # 파일:줄에 중단점
(gdb) continue            # 계속 실행
(gdb) next                # 다음 줄 (함수 진입 안함)
(gdb) step                # 다음 줄 (함수 진입)
(gdb) print variable      # 변수 출력
(gdb) backtrace           # 스택 추적
(gdb) frame 2             # 프레임 전환
(gdb) watch variable      # 변수 감시
(gdb) info breakpoints    # 중단점 목록
(gdb) delete 1            # 중단점 1 삭제
```

### 19.2 Valgrind

```bash
# 메모리 누수 검사
valgrind --leak-check=full ./program

# 메모리 에러 검사
valgrind --track-origins=yes ./program

# 캐시 프로파일링
valgrind --tool=cachegrind ./program
cg_annotate cachegrind.out.<pid>

# 힙 프로파일링
valgrind --tool=massif ./program
ms_print massif.out.<pid>
```

### 19.3 정적 분석 도구

```bash
# Clang Static Analyzer
scan-build gcc program.c

# Cppcheck
cppcheck --enable=all program.c

# Splint
splint program.c

# AddressSanitizer (ASan)
gcc -fsanitize=address -g program.c -o program
./program

# UndefinedBehaviorSanitizer (UBSan)
gcc -fsanitize=undefined -g program.c -o program
./program
```

### 19.4 어서션과 디버그 매크로

```c
#include <assert.h>

void process(int *ptr, int size) {
    assert(ptr != NULL);
    assert(size > 0);
    // ...
}

// 커스텀 디버그 매크로
#ifdef DEBUG
    #define DEBUG_PRINT(fmt, ...) \
        fprintf(stderr, "[%s:%d] " fmt "\n", \
                __FILE__, __LINE__, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...) do {} while (0)
#endif

// 사용
DEBUG_PRINT("Value: %d", x);

// 컴파일 타임 어서션 (C11)
_Static_assert(sizeof(int) == 4, "int must be 4 bytes");
```

---

## 20. 실전 프로젝트 패턴

### 20.1 에러 처리 패턴

```c
// 1. 에러 코드 반환
typedef enum {
    SUCCESS = 0,
    ERROR_INVALID_ARGUMENT = -1,
    ERROR_OUT_OF_MEMORY = -2,
    ERROR_FILE_NOT_FOUND = -3
} ErrorCode;

ErrorCode process_data(const char *filename, int **result) {
    if (filename == NULL || result == NULL) {
        return ERROR_INVALID_ARGUMENT;
    }

    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        return ERROR_FILE_NOT_FOUND;
    }

    *result = malloc(100 * sizeof(int));
    if (*result == NULL) {
        fclose(fp);
        return ERROR_OUT_OF_MEMORY;
    }

    // 처리...
    fclose(fp);
    return SUCCESS;
}

// 2. errno 활용
#include <errno.h>
#include <string.h>

void safe_file_operation() {
    FILE *fp = fopen("file.txt", "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: %s\n", strerror(errno));
        return;
    }
    // ...
}
```

### 20.2 리소스 관리 패턴 (RAII 모방)

```c
// 자동 정리 매크로
#define CLEANUP_FUNC(func) __attribute__((cleanup(func)))

void cleanup_file(FILE **fp) {
    if (*fp != NULL) {
        fclose(*fp);
        *fp = NULL;
    }
}

void cleanup_malloc(void *ptr) {
    void **p = (void**)ptr;
    if (*p != NULL) {
        free(*p);
        *p = NULL;
    }
}

void example_function() {
    CLEANUP_FUNC(cleanup_file) FILE *fp = fopen("file.txt", "r");
    CLEANUP_FUNC(cleanup_malloc) char *buffer = malloc(1024);

    if (fp == NULL || buffer == NULL) {
        return;  // 자동으로 정리됨
    }

    // 작업...
    // 함수 종료 시 자동으로 cleanup 함수 호출됨
}
```

### 20.3 플러그인 시스템

```c
// plugin.h
typedef struct {
    const char *name;
    int version;
    int (*init)(void);
    void (*process)(void *data);
    void (*cleanup)(void);
} Plugin;

// main.c
#include <dlfcn.h>  // 동적 로딩

void load_plugin(const char *path) {
    void *handle = dlopen(path, RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Cannot load plugin: %s\n", dlerror());
        return;
    }

    Plugin *(*get_plugin)(void) = dlsym(handle, "get_plugin");
    if (!get_plugin) {
        fprintf(stderr, "Cannot find get_plugin: %s\n", dlerror());
        dlclose(handle);
        return;
    }

    Plugin *plugin = get_plugin();
    plugin->init();
    plugin->process(data);
    plugin->cleanup();

    dlclose(handle);
}

// plugin_example.c
static int plugin_init(void) {
    printf("Plugin initialized\n");
    return 0;
}

static void plugin_process(void *data) {
    printf("Processing data\n");
}

static void plugin_cleanup(void) {
    printf("Plugin cleanup\n");
}

static Plugin my_plugin = {
    .name = "Example Plugin",
    .version = 1,
    .init = plugin_init,
    .process = plugin_process,
    .cleanup = plugin_cleanup
};

Plugin* get_plugin(void) {
    return &my_plugin;
}
```

### 20.4 설정 파일 파싱

```c
// INI 파일 파서
typedef struct {
    char section[64];
    char key[64];
    char value[256];
} ConfigEntry;

typedef struct {
    ConfigEntry *entries;
    int count;
    int capacity;
} Config;

Config* parse_config(const char *filename) {
    Config *config = malloc(sizeof(Config));
    config->capacity = 10;
    config->count = 0;
    config->entries = malloc(config->capacity * sizeof(ConfigEntry));

    FILE *fp = fopen(filename, "r");
    if (!fp) return config;

    char line[512];
    char current_section[64] = "";

    while (fgets(line, sizeof(line), fp)) {
        // 주석 제거
        char *comment = strchr(line, '#');
        if (comment) *comment = '\0';

        // 공백 제거
        char *trimmed = line;
        while (isspace(*trimmed)) trimmed++;

        if (*trimmed == '\0') continue;

        // 섹션 파싱
        if (*trimmed == '[') {
            sscanf(trimmed, "[%63[^]]]", current_section);
            continue;
        }

        // 키=값 파싱
        char key[64], value[256];
        if (sscanf(trimmed, "%63[^=]=%255[^\n]", key, value) == 2) {
            if (config->count >= config->capacity) {
                config->capacity *= 2;
                config->entries = realloc(config->entries,
                    config->capacity * sizeof(ConfigEntry));
            }

            ConfigEntry *entry = &config->entries[config->count++];
            strcpy(entry->section, current_section);
            strcpy(entry->key, key);
            strcpy(entry->value, value);
        }
    }

    fclose(fp);
    return config;
}

const char* get_config_value(Config *config,
                              const char *section,
                              const char *key) {
    for (int i = 0; i < config->count; i++) {
        if (strcmp(config->entries[i].section, section) == 0 &&
            strcmp(config->entries[i].key, key) == 0) {
            return config->entries[i].value;
        }
    }
    return NULL;
}
```

### 20.5 로깅 시스템

```c
typedef enum {
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARNING,
    LOG_ERROR
} LogLevel;

typedef struct {
    FILE *file;
    LogLevel min_level;
} Logger;

Logger* create_logger(const char *filename, LogLevel level) {
    Logger *logger = malloc(sizeof(Logger));
    logger->file = fopen(filename, "a");
    logger->min_level = level;
    return logger;
}

void log_message(Logger *logger, LogLevel level,
                 const char *file, int line,
                 const char *fmt, ...) {
    if (level < logger->min_level) return;

    const char *level_str[] = {"DEBUG", "INFO", "WARNING", "ERROR"};

    time_t now = time(NULL);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp),
             "%Y-%m-%d %H:%M:%S", localtime(&now));

    fprintf(logger->file, "[%s] [%s] %s:%d: ",
            timestamp, level_str[level], file, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(logger->file, fmt, args);
    va_end(args);

    fprintf(logger->file, "\n");
    fflush(logger->file);
}

#define LOG_DEBUG(logger, ...) \
    log_message(logger, LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(logger, ...) \
    log_message(logger, LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARNING(logger, ...) \
    log_message(logger, LOG_WARNING, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(logger, ...) \
    log_message(logger, LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)

// 사용
Logger *logger = create_logger("app.log", LOG_INFO);
LOG_INFO(logger, "Application started");
LOG_ERROR(logger, "Failed to open file: %s", filename);
```

---

## 결론

이 C 언어 가이드는 기초부터 고급 기법까지 모든 것을 다룹니다:

1. **기초**: 변수, 타입, 연산자, 제어문
2. **핵심**: 함수, 포인터, 배열, 구조체
3. **메모리**: 동적 할당, 메모리 관리, 최적화
4. **고급**: 비트 연산, 전처리기, 함수 포인터
5. **실전**: 파일 I/O, 표준 라이브러리, 프로젝트 패턴
6. **전문가**: 컴파일, 링킹, 디버깅, 성능 최적화

C를 마스터하면:
- 메모리와 하드웨어를 깊이 이해
- 시스템 프로그래밍 능력 획득
- 다른 언어의 기초 원리 파악
- 성능 최적화 전문성 확보

**추천 학습 순서**: 1→7→6→5→9→10→11→14→15→나머지

계속 연습하고, 작은 프로젝트를 만들어보세요!
