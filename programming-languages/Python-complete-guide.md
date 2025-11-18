# Python 완벽 학습 가이드

## 목차
1. [Python 기초](#1-python-기초)
2. [데이터 타입과 변수](#2-데이터-타입과-변수)
3. [연산자](#3-연산자)
4. [제어문](#4-제어문)
5. [함수](#5-함수)
6. [자료구조](#6-자료구조)
7. [문자열 처리](#7-문자열-처리)
8. [모듈과 패키지](#8-모듈과-패키지)
9. [파일 입출력](#9-파일-입출력)
10. [예외 처리](#10-예외-처리)
11. [객체지향 프로그래밍](#11-객체지향-프로그래밍)
12. [함수형 프로그래밍](#12-함수형-프로그래밍)
13. [반복자와 제너레이터](#13-반복자와-제너레이터)
14. [데코레이터](#14-데코레이터)
15. [컨텍스트 매니저](#15-컨텍스트-매니저)
16. [정규표현식](#16-정규표현식)
17. [표준 라이브러리](#17-표준-라이브러리)
18. [동시성과 병렬성](#18-동시성과-병렬성)
19. [타입 힌팅](#19-타입-힌팅)
20. [고급 기능과 패턴](#20-고급-기능과-패턴)

---

## 1. Python 기초

### 1.1 Python이란?
- 1991년 Guido van Rossum이 만든 고수준 프로그래밍 언어
- 동적 타이핑
- 인터프리터 언어
- 읽기 쉬운 문법 (의사코드와 유사)
- 다중 패러다임 지원 (객체지향, 함수형, 절차적)

### 1.2 설치 및 실행

```bash
# 버전 확인
python --version
python3 --version

# REPL (대화형 인터프리터)
python
>>> print("Hello, Python!")
>>> exit()

# 스크립트 실행
python script.py
python3 script.py

# pip (패키지 관리자)
pip install package_name
pip install -r requirements.txt
pip list
pip freeze > requirements.txt
```

### 1.3 첫 프로그램

```python
# hello.py
print("Hello, World!")

# 실행: python hello.py
```

### 1.4 주석과 Docstring

```python
# 한 줄 주석

"""
여러 줄 주석
또는 문서화 문자열 (docstring)
"""

def greet(name):
    """
    이름을 받아 인사말을 반환하는 함수

    Args:
        name (str): 인사할 대상의 이름

    Returns:
        str: 인사말 문자열
    """
    return f"Hello, {name}!"

# docstring 접근
print(greet.__doc__)
```

---

## 2. 데이터 타입과 변수

### 2.1 변수와 기본 타입

```python
# 변수 (동적 타이핑)
x = 10          # int
y = 3.14        # float
name = "Alice"  # str
flag = True     # bool
nothing = None  # NoneType

# 타입 확인
print(type(x))  # <class 'int'>

# 여러 변수 동시 할당
a, b, c = 1, 2, 3
x = y = z = 0

# 변수명 규칙
# - 영문, 숫자, 언더스코어
# - 숫자로 시작 불가
# - 키워드 사용 불가
# - 관례: snake_case
```

### 2.2 숫자 타입

```python
# 정수 (int) - 임의 정밀도
big_num = 123456789012345678901234567890
print(big_num * 2)

# 진법 표현
binary = 0b1010      # 10 (2진수)
octal = 0o12         # 10 (8진수)
hexadecimal = 0xA    # 10 (16진수)

# 실수 (float)
pi = 3.14159
scientific = 1.5e-3  # 0.0015

# 복소수 (complex)
z = 3 + 4j
print(z.real)   # 3.0
print(z.imag)   # 4.0
print(abs(z))   # 5.0

# 타입 변환
x = int("10")      # 10
y = float("3.14")  # 3.14
s = str(42)        # "42"

# 산술 연산
print(10 + 3)   # 13
print(10 - 3)   # 7
print(10 * 3)   # 30
print(10 / 3)   # 3.3333... (실수 나눗셈)
print(10 // 3)  # 3 (정수 나눗셈)
print(10 % 3)   # 1 (나머지)
print(10 ** 3)  # 1000 (거듭제곱)

# 복합 대입
x = 10
x += 5   # x = x + 5
x -= 3   # x = x - 3
x *= 2   # x = x * 2
x /= 4   # x = x / 4
x //= 2  # x = x // 2
x %= 3   # x = x % 3
x **= 2  # x = x ** 2
```

### 2.3 불린과 None

```python
# 불린
is_active = True
is_empty = False

# 불린 연산
print(True and False)  # False
print(True or False)   # True
print(not True)        # False

# Truthy와 Falsy
# Falsy: False, None, 0, 0.0, "", [], {}, ()
# Truthy: 그 외 모두

if "":
    print("This won't print")

if [1, 2, 3]:
    print("This will print")

# None
result = None
if result is None:
    print("Result is None")

# is vs ==
# is: 같은 객체인지 확인 (identity)
# ==: 값이 같은지 확인 (equality)
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)  # True (값이 같음)
print(a is b)  # False (다른 객체)
print(a is c)  # True (같은 객체)
```

---

## 3. 연산자

### 3.1 비교 연산자

```python
x, y = 10, 20

print(x == y)   # False (같음)
print(x != y)   # True (다름)
print(x < y)    # True
print(x > y)    # False
print(x <= y)   # True
print(x >= y)   # False

# 체이닝
age = 25
print(18 < age < 65)  # True

# 문자열 비교 (사전순)
print("apple" < "banana")  # True
```

### 3.2 논리 연산자

```python
x, y = True, False

print(x and y)  # False
print(x or y)   # True
print(not x)    # False

# 단축 평가 (short-circuit evaluation)
def true_func():
    print("True function called")
    return True

def false_func():
    print("False function called")
    return False

# or: 첫 True에서 멈춤
print(true_func() or false_func())
# 출력: True function called, True

# and: 첫 False에서 멈춤
print(false_func() and true_func())
# 출력: False function called, False
```

### 3.3 비트 연산자

```python
a = 0b1100  # 12
b = 0b1010  # 10

print(bin(a & b))   # 0b1000 (8) - AND
print(bin(a | b))   # 0b1110 (14) - OR
print(bin(a ^ b))   # 0b0110 (6) - XOR
print(bin(~a))      # -0b1101 - NOT
print(bin(a << 2))  # 0b110000 (48) - 왼쪽 시프트
print(bin(a >> 2))  # 0b11 (3) - 오른쪽 시프트
```

### 3.4 멤버십 및 식별 연산자

```python
# in / not in
fruits = ["apple", "banana", "cherry"]
print("apple" in fruits)      # True
print("grape" not in fruits)  # True

# in은 문자열에도 사용
text = "Hello, World"
print("World" in text)  # True

# is / is not
x = [1, 2, 3]
y = [1, 2, 3]
z = x

print(x is y)      # False
print(x is z)      # True
print(x is not y)  # True
```

---

## 4. 제어문

### 4.1 if 문

```python
age = 18

if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")

# 한 줄 if
x = 10
result = "positive" if x > 0 else "non-positive"

# 조건부 표현식 (삼항 연산자)
max_val = a if a > b else b

# match-case (Python 3.10+)
def http_status(status):
    match status:
        case 200:
            return "OK"
        case 404:
            return "Not Found"
        case 500:
            return "Internal Server Error"
        case _:  # default
            return "Unknown"
```

### 4.2 while 루프

```python
# 기본 while
count = 0
while count < 5:
    print(count)
    count += 1

# 무한 루프
# while True:
#     command = input("Enter command: ")
#     if command == "quit":
#         break
#     process(command)

# break와 continue
i = 0
while i < 10:
    i += 1
    if i % 2 == 0:
        continue  # 다음 반복으로
    if i > 7:
        break     # 루프 탈출
    print(i)

# else 절 (break 없이 정상 종료 시)
count = 0
while count < 5:
    print(count)
    count += 1
else:
    print("Loop completed normally")
```

### 4.3 for 루프

```python
# 기본 for
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# range 사용
for i in range(5):        # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 10):    # 2 ~ 9
    print(i)

for i in range(0, 10, 2): # 0, 2, 4, 6, 8 (step=2)
    print(i)

# enumerate (인덱스와 값)
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# zip (여러 시퀀스 동시 순회)
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# 딕셔너리 순회
person = {"name": "Alice", "age": 25, "city": "NYC"}

for key in person:
    print(key, person[key])

for key, value in person.items():
    print(f"{key}: {value}")

# break, continue, else
for i in range(10):
    if i == 3:
        continue
    if i == 7:
        break
    print(i)
else:
    print("No break occurred")

# 리스트 컴프리헨션
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]
```

---

## 5. 함수

### 5.1 함수 기본

```python
# 함수 정의
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")

# 반환값
def add(a, b):
    return a + b

result = add(3, 5)

# 여러 값 반환 (튜플)
def get_name():
    return "Alice", "Bob"

first, second = get_name()

# 기본 매개변수
def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")

greet("Alice")              # Hello, Alice!
greet("Bob", "Hi")          # Hi, Bob!
greet("Charlie", greeting="Hey")  # Hey, Charlie!

# 가변 인자 (*args)
def sum_all(*numbers):
    return sum(numbers)

print(sum_all(1, 2, 3, 4, 5))  # 15

# 키워드 가변 인자 (**kwargs)
def print_info(**info):
    for key, value in info.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="NYC")

# 혼합 사용
def func(a, b, *args, **kwargs):
    print(f"a={a}, b={b}")
    print(f"args={args}")
    print(f"kwargs={kwargs}")

func(1, 2, 3, 4, 5, x=10, y=20)
```

### 5.2 람다 함수

```python
# 람다: 익명 함수
square = lambda x: x ** 2
print(square(5))  # 25

add = lambda a, b: a + b
print(add(3, 4))  # 7

# 주로 고차 함수와 함께 사용
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

# 정렬
pairs = [(1, 'one'), (2, 'two'), (3, 'three')]
sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
```

### 5.3 스코프와 클로저

```python
# LEGB 규칙: Local, Enclosing, Global, Built-in

x = "global"

def outer():
    x = "enclosing"

    def inner():
        x = "local"
        print(x)  # local

    inner()
    print(x)  # enclosing

outer()
print(x)  # global

# global 키워드
count = 0

def increment():
    global count
    count += 1

increment()
print(count)  # 1

# nonlocal 키워드
def outer():
    count = 0

    def inner():
        nonlocal count
        count += 1
        print(count)

    inner()  # 1
    inner()  # 2

outer()

# 클로저
def make_multiplier(n):
    def multiplier(x):
        return x * n
    return multiplier

times3 = make_multiplier(3)
times5 = make_multiplier(5)

print(times3(10))  # 30
print(times5(10))  # 50
```

### 5.4 재귀와 메모이제이션

```python
# 재귀
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # 120

# 피보나치 (비효율적)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 메모이제이션 (functools.lru_cache)
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_memo(n):
    if n <= 1:
        return n
    return fibonacci_memo(n-1) + fibonacci_memo(n-2)

print(fibonacci_memo(100))  # 빠르게 계산됨

# 수동 메모이제이션
def memoize(func):
    cache = {}
    def wrapper(n):
        if n not in cache:
            cache[n] = func(n)
        return cache[n]
    return wrapper

@memoize
def fibonacci_manual(n):
    if n <= 1:
        return n
    return fibonacci_manual(n-1) + fibonacci_manual(n-2)
```

---

## 6. 자료구조

### 6.1 리스트 (List)

```python
# 리스트 생성
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "two", 3.0, [4, 5]]

# 빈 리스트
empty = []
empty2 = list()

# 인덱싱
print(fruits[0])   # apple
print(fruits[-1])  # cherry (마지막)

# 슬라이싱
print(numbers[1:4])   # [2, 3, 4]
print(numbers[:3])    # [1, 2, 3]
print(numbers[3:])    # [4, 5]
print(numbers[::2])   # [1, 3, 5] (step=2)
print(numbers[::-1])  # [5, 4, 3, 2, 1] (역순)

# 수정
fruits[0] = "apricot"
fruits[1:3] = ["blueberry", "coconut"]

# 추가
fruits.append("date")           # 끝에 추가
fruits.insert(1, "avocado")     # 인덱스 1에 삽입
fruits.extend(["fig", "grape"]) # 여러 항목 추가

# 삭제
fruits.remove("banana")  # 값으로 삭제 (첫 번째)
popped = fruits.pop()    # 마지막 항목 제거 및 반환
popped = fruits.pop(0)   # 인덱스 0 제거 및 반환
del fruits[1]            # 인덱스로 삭제
fruits.clear()           # 모두 삭제

# 검색
fruits = ["apple", "banana", "cherry"]
print("banana" in fruits)      # True
print(fruits.index("banana"))  # 1
print(fruits.count("apple"))   # 1

# 정렬
numbers = [3, 1, 4, 1, 5, 9, 2]
numbers.sort()              # 원본 수정
sorted_nums = sorted(numbers)  # 새 리스트 반환

numbers.sort(reverse=True)  # 내림차순
fruits.sort(key=len)        # 길이 기준 정렬

# 기타
numbers.reverse()  # 역순
length = len(fruits)

# 리스트 컴프리헨션
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]
matrix = [[i*j for j in range(3)] for i in range(3)]

# 중첩 컴프리헨션
flattened = [item for sublist in matrix for item in sublist]
```

### 6.2 튜플 (Tuple)

```python
# 튜플 생성 (불변)
point = (3, 4)
person = ("Alice", 25, "NYC")

# 단일 원소 튜플
single = (42,)  # 쉼표 필수
# single = (42)  # 이건 그냥 int

# 언패킹
x, y = point
name, age, city = person

# 나머지 언패킹 (Python 3)
first, *rest = (1, 2, 3, 4, 5)
print(first)  # 1
print(rest)   # [2, 3, 4, 5]

# 튜플은 불변
# point[0] = 5  # 에러!

# 인덱싱, 슬라이싱은 가능
print(person[0])   # Alice
print(person[1:])  # (25, 'NYC')

# 용도: 여러 값 반환, 딕셔너리 키
def get_coordinates():
    return (10, 20)

coords = {(0, 0): "origin", (1, 0): "x-axis"}
```

### 6.3 세트 (Set)

```python
# 세트 생성 (중복 없음, 순서 없음)
fruits = {"apple", "banana", "cherry"}
numbers = {1, 2, 3, 4, 5}

# 빈 세트
empty = set()
# empty = {}  # 이건 딕셔너리

# 중복 제거
numbers = {1, 2, 2, 3, 3, 3, 4, 5}
print(numbers)  # {1, 2, 3, 4, 5}

# 추가 및 삭제
fruits.add("date")
fruits.remove("banana")     # 없으면 에러
fruits.discard("banana")    # 없어도 에러 안남
popped = fruits.pop()       # 임의의 항목 제거
fruits.clear()

# 집합 연산
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a | b)  # {1, 2, 3, 4, 5, 6} (합집합)
print(a & b)  # {3, 4} (교집합)
print(a - b)  # {1, 2} (차집합)
print(a ^ b)  # {1, 2, 5, 6} (대칭 차집합)

# 메서드
print(a.union(b))
print(a.intersection(b))
print(a.difference(b))
print(a.symmetric_difference(b))

# 부분집합/상위집합
print({1, 2}.issubset({1, 2, 3}))    # True
print({1, 2, 3}.issuperset({1, 2}))  # True

# 세트 컴프리헨션
squares = {x**2 for x in range(10)}
```

### 6.4 딕셔너리 (Dictionary)

```python
# 딕셔너리 생성
person = {
    "name": "Alice",
    "age": 25,
    "city": "NYC"
}

# dict() 생성자
person2 = dict(name="Bob", age=30, city="LA")

# 빈 딕셔너리
empty = {}
empty2 = dict()

# 접근
print(person["name"])        # Alice
print(person.get("name"))    # Alice
print(person.get("email", "N/A"))  # 기본값

# 수정
person["age"] = 26
person["email"] = "alice@example.com"

# 삭제
del person["city"]
age = person.pop("age")      # 제거 및 반환
person.clear()

# 키 확인
person = {"name": "Alice", "age": 25}
print("name" in person)      # True
print("email" in person)     # False

# 순회
for key in person:
    print(key, person[key])

for key, value in person.items():
    print(f"{key}: {value}")

for key in person.keys():
    print(key)

for value in person.values():
    print(value)

# 병합 (Python 3.9+)
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
merged = dict1 | dict2  # {"a": 1, "b": 3, "c": 4}

# update
dict1.update(dict2)

# setdefault
person.setdefault("email", "default@example.com")

# defaultdict
from collections import defaultdict

dd = defaultdict(int)
dd["count"] += 1  # 키가 없어도 에러 안남

# 딕셔너리 컴프리헨션
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### 6.5 collections 모듈

```python
from collections import (
    Counter, defaultdict, OrderedDict,
    deque, namedtuple, ChainMap
)

# Counter
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
counter = Counter(words)
print(counter)  # Counter({'apple': 3, 'banana': 2, 'cherry': 1})
print(counter.most_common(2))  # [('apple', 3), ('banana', 2)]

# defaultdict
dd = defaultdict(list)
dd["fruits"].append("apple")
dd["fruits"].append("banana")

# OrderedDict (Python 3.7+는 dict도 순서 유지)
od = OrderedDict()
od["a"] = 1
od["b"] = 2

# deque (양방향 큐)
dq = deque([1, 2, 3])
dq.append(4)       # 오른쪽 추가
dq.appendleft(0)   # 왼쪽 추가
dq.pop()           # 오른쪽 제거
dq.popleft()       # 왼쪽 제거

# namedtuple
Point = namedtuple("Point", ["x", "y"])
p = Point(10, 20)
print(p.x, p.y)    # 10 20
print(p[0], p[1])  # 10 20

# ChainMap
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
chain = ChainMap(dict1, dict2)
print(chain["b"])  # 2 (dict1이 우선)
```

---

## 7. 문자열 처리

### 7.1 문자열 기본

```python
# 문자열 생성
s1 = 'single quotes'
s2 = "double quotes"
s3 = """triple
quotes
multiline"""

# 이스케이프
s = "Hello\nWorld"   # 줄바꿈
s = "Tab\there"      # 탭
s = "Quote: \"Hi\""  # 따옴표

# raw 문자열
path = r"C:\new\folder"  # 이스케이프 무시

# f-string (Python 3.6+)
name = "Alice"
age = 25
print(f"My name is {name} and I'm {age} years old")
print(f"Next year: {age + 1}")
print(f"Pi: {3.14159:.2f}")  # 소수점 2자리

# format 메서드
print("Hello, {}!".format(name))
print("{0} is {1} years old".format(name, age))
print("{name} is {age}".format(name=name, age=age))

# % 포맷팅 (구식)
print("Hello, %s!" % name)
print("%s is %d years old" % (name, age))
```

### 7.2 문자열 연산

```python
# 연결
s = "Hello" + " " + "World"

# 반복
s = "Ha" * 3  # "HaHaHa"

# 인덱싱
s = "Python"
print(s[0])   # P
print(s[-1])  # n

# 슬라이싱
print(s[0:3])   # Pyt
print(s[2:])    # thon
print(s[::-1])  # nohtyP (역순)

# 길이
print(len(s))  # 6

# 멤버십
print("Py" in s)      # True
print("Java" not in s)  # True
```

### 7.3 문자열 메서드

```python
s = "  Hello, World!  "

# 대소문자
print(s.upper())      # HELLO, WORLD!
print(s.lower())      # hello, world!
print(s.capitalize()) # Hello, world!
print(s.title())      # Hello, World!
print(s.swapcase())   # HELLO, wORLD!

# 공백 제거
print(s.strip())      # "Hello, World!"
print(s.lstrip())     # "Hello, World!  "
print(s.rstrip())     # "  Hello, World!"

# 검색
s = "Hello, World!"
print(s.find("World"))     # 7
print(s.find("Python"))    # -1 (없으면)
print(s.index("World"))    # 7
# print(s.index("Python"))  # ValueError

print(s.startswith("Hello"))  # True
print(s.endswith("!"))        # True
print(s.count("o"))           # 2

# 분할과 결합
words = "apple,banana,cherry".split(",")
print(words)  # ['apple', 'banana', 'cherry']

lines = "line1\nline2\nline3".splitlines()
print(lines)  # ['line1', 'line2', 'line3']

joined = " ".join(words)
print(joined)  # "apple banana cherry"

# 대체
s = "Hello, World!"
print(s.replace("World", "Python"))  # "Hello, Python!"
print(s.replace("o", "0", 1))        # "Hell0, World!" (1개만)

# 확인
print("123".isdigit())      # True
print("abc".isalpha())      # True
print("abc123".isalnum())   # True
print("   ".isspace())      # True

# 정렬
print("hello".center(20, "*"))  # "*******hello********"
print("hello".ljust(10, "*"))   # "hello*****"
print("hello".rjust(10, "*"))   # "*****hello"
print("42".zfill(5))            # "00042"
```

### 7.4 문자열 포맷팅 고급

```python
# 정렬
print(f"{'left':<10}|")     # "left      |"
print(f"{'right':>10}|")    # "     right|"
print(f"{'center':^10}|")   # "  center  |"

# 숫자 포맷
pi = 3.14159
print(f"{pi:.2f}")          # "3.14"
print(f"{pi:.4f}")          # "3.1416"

num = 1234567
print(f"{num:,}")           # "1,234,567"
print(f"{num:_}")           # "1_234_567"

# 진법
num = 255
print(f"{num:b}")           # "11111111" (2진수)
print(f"{num:o}")           # "377" (8진수)
print(f"{num:x}")           # "ff" (16진수)
print(f"{num:X}")           # "FF" (대문자)

# 백분율
ratio = 0.75
print(f"{ratio:.2%}")       # "75.00%"

# 표현식
x, y = 10, 20
print(f"{x + y = }")        # "x + y = 30" (Python 3.8+)
```

---

## 8. 모듈과 패키지

### 8.1 모듈 임포트

```python
# 전체 임포트
import math
print(math.sqrt(16))

# 특정 함수 임포트
from math import sqrt, pi
print(sqrt(16))
print(pi)

# 별칭
import numpy as np
import pandas as pd

from math import sqrt as square_root
print(square_root(16))

# 모두 임포트 (권장하지 않음)
from math import *

# 조건부 임포트
try:
    import numpy
except ImportError:
    print("NumPy not installed")
```

### 8.2 모듈 만들기

```python
# mymodule.py
def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b

PI = 3.14159

# main.py
import mymodule

print(mymodule.greet("Alice"))
print(mymodule.add(3, 5))
print(mymodule.PI)

# 또는
from mymodule import greet, add
print(greet("Bob"))
```

### 8.3 패키지

```
mypackage/
    __init__.py
    module1.py
    module2.py
    subpackage/
        __init__.py
        module3.py
```

```python
# __init__.py
# 패키지 초기화 코드
from .module1 import func1
from .module2 import func2

__all__ = ['func1', 'func2']

# 사용
import mypackage
mypackage.func1()

from mypackage import func1
func1()

from mypackage.subpackage import module3
```

### 8.4 __name__과 __main__

```python
# mymodule.py
def main():
    print("Running as main")

if __name__ == "__main__":
    main()

# 모듈로 임포트되면 __name__은 "mymodule"
# 직접 실행하면 __name__은 "__main__"
```

---

## 9. 파일 입출력

### 9.1 텍스트 파일

```python
# 파일 쓰기
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World!\n")
    f.write("Second line\n")

# 파일 읽기
with open("input.txt", "r", encoding="utf-8") as f:
    content = f.read()  # 전체 읽기
    print(content)

# 한 줄씩 읽기
with open("input.txt", "r") as f:
    for line in f:
        print(line.strip())

# 모든 줄을 리스트로
with open("input.txt", "r") as f:
    lines = f.readlines()

# 추가 모드
with open("output.txt", "a") as f:
    f.write("Appended line\n")

# 파일 모드
# "r": 읽기 (기본값)
# "w": 쓰기 (기존 내용 삭제)
# "a": 추가
# "x": 배타적 생성 (파일이 있으면 에러)
# "b": 바이너리 모드
# "+": 읽기/쓰기
```

### 9.2 바이너리 파일

```python
# 바이너리 쓰기
with open("data.bin", "wb") as f:
    f.write(b"\x00\x01\x02\x03")

# 바이너리 읽기
with open("data.bin", "rb") as f:
    data = f.read()
    print(data)

# 이미지 복사
with open("image.jpg", "rb") as src:
    with open("copy.jpg", "wb") as dst:
        dst.write(src.read())
```

### 9.3 파일 경로 다루기

```python
import os
from pathlib import Path

# os.path
path = os.path.join("folder", "subfolder", "file.txt")
print(os.path.exists(path))
print(os.path.isfile(path))
print(os.path.isdir(path))
print(os.path.basename(path))  # file.txt
print(os.path.dirname(path))   # folder/subfolder
print(os.path.splitext("file.txt"))  # ('file', '.txt')

# pathlib (Python 3.4+, 권장)
path = Path("folder") / "subfolder" / "file.txt"
print(path.exists())
print(path.is_file())
print(path.is_dir())
print(path.name)      # file.txt
print(path.stem)      # file
print(path.suffix)    # .txt
print(path.parent)    # folder/subfolder

# 파일 목록
for file in Path(".").glob("*.py"):
    print(file)

# 재귀적 검색
for file in Path(".").rglob("*.txt"):
    print(file)

# 파일 읽기/쓰기
path = Path("file.txt")
path.write_text("Hello, World!")
content = path.read_text()
```

### 9.4 JSON, CSV 파일

```python
import json
import csv

# JSON 쓰기
data = {
    "name": "Alice",
    "age": 25,
    "hobbies": ["reading", "coding"]
}

with open("data.json", "w") as f:
    json.dump(data, f, indent=4)

# JSON 읽기
with open("data.json", "r") as f:
    loaded = json.load(f)
    print(loaded)

# JSON 문자열
json_str = json.dumps(data, indent=2)
loaded = json.loads(json_str)

# CSV 쓰기
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age", "City"])
    writer.writerow(["Alice", 25, "NYC"])
    writer.writerow(["Bob", 30, "LA"])

# CSV 읽기
with open("data.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# DictReader/DictWriter
with open("data.csv", "w", newline="") as f:
    fieldnames = ["Name", "Age", "City"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({"Name": "Alice", "Age": 25, "City": "NYC"})

with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["Name"], row["Age"])
```

---

## 10. 예외 처리

```python
# 기본 예외 처리
try:
    x = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")

# 여러 예외
try:
    value = int(input("Enter a number: "))
    result = 10 / value
except ValueError:
    print("Invalid number")
except ZeroDivisionError:
    print("Cannot divide by zero")

# 예외 객체 접근
try:
    f = open("nonexistent.txt")
except FileNotFoundError as e:
    print(f"Error: {e}")

# else와 finally
try:
    f = open("file.txt", "r")
except FileNotFoundError:
    print("File not found")
else:
    print("File opened successfully")
    content = f.read()
    f.close()
finally:
    print("Cleanup code")

# 예외 발생
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age

# 커스텀 예외
class CustomError(Exception):
    def __init__(self, message, code):
        self.message = message
        self.code = code
        super().__init__(self.message)

raise CustomError("Something went wrong", 500)

# 예외 체이닝
try:
    # ...
except Exception as e:
    raise RuntimeError("Failed to process") from e
```

---

## 11. 객체지향 프로그래밍

### 11.1 클래스 기본

```python
class Person:
    # 클래스 변수
    species = "Homo sapiens"

    def __init__(self, name, age):
        # 인스턴스 변수
        self.name = name
        self.age = age

    # 인스턴스 메서드
    def greet(self):
        print(f"Hello, I'm {self.name}")

    # 클래스 메서드
    @classmethod
    def from_birth_year(cls, name, birth_year):
        age = 2024 - birth_year
        return cls(name, age)

    # 정적 메서드
    @staticmethod
    def is_adult(age):
        return age >= 18

# 사용
p = Person("Alice", 25)
p.greet()

p2 = Person.from_birth_year("Bob", 1990)
print(Person.is_adult(20))
```

### 11.2 상속

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

# 다중 상속
class Flyable:
    def fly(self):
        return "Flying..."

class Bird(Animal, Flyable):
    def speak(self):
        return f"{self.name} says Tweet!"

# super()
class Employee(Person):
    def __init__(self, name, age, employee_id):
        super().__init__(name, age)
        self.employee_id = employee_id
```

### 11.3 특수 메서드 (매직 메서드)

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Index out of range")

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)  # Vector(4, 6)
print(v1 * 3)   # Vector(3, 6)
```

### 11.4 프로퍼티

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value

    @property
    def area(self):
        return 3.14159 * self._radius ** 2

    @property
    def diameter(self):
        return self._radius * 2

    @diameter.setter
    def diameter(self, value):
        self._radius = value / 2

c = Circle(5)
print(c.area)      # 78.53975
c.diameter = 20
print(c.radius)    # 10.0
```

---

## 12. 함수형 프로그래밍

```python
# map
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))

# filter
evens = list(filter(lambda x: x % 2 == 0, numbers))

# reduce
from functools import reduce
sum_all = reduce(lambda x, y: x + y, numbers)

# 리스트 컴프리헨션 (더 pythonic)
squared = [x**2 for x in numbers]
evens = [x for x in numbers if x % 2 == 0]

# 딕셔너리 컴프리헨션
squared_dict = {x: x**2 for x in numbers}

# 집합 컴프리헨션
unique_squares = {x**2 for x in [1, 2, 2, 3, 3, 4]}

# 고차 함수
def apply_twice(func, x):
    return func(func(x))

result = apply_twice(lambda x: x * 2, 5)  # 20

# partial
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(3))    # 27
```

---

## 13. 반복자와 제너레이터

### 13.1 반복자

```python
class Counter:
    def __init__(self, max):
        self.max = max
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.max:
            self.count += 1
            return self.count
        raise StopIteration

for num in Counter(5):
    print(num)  # 1, 2, 3, 4, 5
```

### 13.2 제너레이터

```python
# 제너레이터 함수
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for i in countdown(5):
    print(i)

# 무한 제너레이터
def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1

# 제너레이터 표현식
squares = (x**2 for x in range(10))

# 피보나치 제너레이터
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# send() 메서드
def echo():
    while True:
        value = yield
        print(f"Received: {value}")

gen = echo()
next(gen)  # 시작
gen.send(10)
gen.send(20)
```

---

## 14. 데코레이터

```python
# 기본 데코레이터
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@my_decorator
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")

# 매개변수가 있는 데코레이터
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def say_hello():
    print("Hello!")

# 클래스 데코레이터
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Call {self.count}")
        return self.func(*args, **kwargs)

@CountCalls
def greet():
    print("Hello!")

# 내장 데코레이터
class MyClass:
    @staticmethod
    def static_method():
        print("Static method")

    @classmethod
    def class_method(cls):
        print("Class method")

    @property
    def my_property(self):
        return "Property value"

# functools.wraps
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

---

## 15. 컨텍스트 매니저

```python
# with 문
with open("file.txt", "r") as f:
    content = f.read()
# 파일 자동 닫힘

# 커스텀 컨텍스트 매니저
class MyContext:
    def __enter__(self):
        print("Entering context")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting context")
        if exc_type is not None:
            print(f"Exception: {exc_val}")
        return False  # 예외 전파

with MyContext() as ctx:
    print("Inside context")

# contextlib
from contextlib import contextmanager

@contextmanager
def my_context():
    print("Setup")
    try:
        yield
    finally:
        print("Cleanup")

with my_context():
    print("Inside")

# 여러 컨텍스트 매니저
with open("input.txt") as infile, open("output.txt", "w") as outfile:
    outfile.write(infile.read())
```

---

## 16. 정규표현식

```python
import re

# 매칭
pattern = r"\d+"  # 숫자
text = "I have 2 apples and 3 oranges"

# search
match = re.search(pattern, text)
if match:
    print(match.group())  # 2

# findall
numbers = re.findall(pattern, text)
print(numbers)  # ['2', '3']

# match (시작부터)
if re.match(r"I", text):
    print("Starts with I")

# 치환
new_text = re.sub(r"\d+", "X", text)
print(new_text)  # "I have X apples and X oranges"

# 그룹
pattern = r"(\w+)@(\w+)\.(\w+)"
email = "user@example.com"
match = re.search(pattern, email)
if match:
    print(match.group(1))  # user
    print(match.group(2))  # example
    print(match.group(3))  # com

# 플래그
re.search(r"hello", "HELLO", re.IGNORECASE)

# 컴파일
pattern = re.compile(r"\d+")
matches = pattern.findall(text)
```

---

## 17. 표준 라이브러리

```python
# datetime
from datetime import datetime, timedelta

now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

tomorrow = now + timedelta(days=1)
week_ago = now - timedelta(weeks=1)

# collections
from collections import Counter, defaultdict, deque, namedtuple

counter = Counter("hello world")
print(counter.most_common(3))

dd = defaultdict(int)
dd["count"] += 1

dq = deque([1, 2, 3])
dq.appendleft(0)
dq.append(4)

Point = namedtuple("Point", ["x", "y"])
p = Point(10, 20)

# itertools
from itertools import chain, cycle, repeat, combinations

# 무한 반복
for i, val in zip(range(5), cycle([1, 2, 3])):
    print(val)

# 조합
for combo in combinations([1, 2, 3, 4], 2):
    print(combo)

# random
import random

random.randint(1, 10)
random.choice([1, 2, 3, 4, 5])
random.shuffle([1, 2, 3, 4, 5])

# os
import os

os.getcwd()
os.listdir(".")
os.mkdir("new_folder")
os.environ["PATH"]

# sys
import sys

sys.argv  # 명령행 인자
sys.exit(0)
sys.version
```

---

## 18. 동시성과 병렬성

### 18.1 GIL (Global Interpreter Lock) - 필수 이해

**⚠️ Python의 가장 중요한 제약사항**

```python
# GIL이란?
# - CPython 인터프리터가 한 번에 하나의 스레드만 Python 바이트코드를 실행하도록 하는 뮤텍스
# - 멀티스레딩으로 CPU-bound 작업을 병렬화할 수 없음!
# - I/O-bound 작업에는 문제 없음

import threading
import time

# ❌ GIL 때문에 느린 예시 (CPU-bound)
def cpu_bound_task():
    total = 0
    for i in range(10_000_000):
        total += i
    return total

# 싱글 스레드
start = time.time()
cpu_bound_task()
cpu_bound_task()
print(f"Single thread: {time.time() - start:.2f}s")  # 약 1.2초

# 멀티 스레드 (GIL 때문에 더 느림!)
start = time.time()
threads = []
for _ in range(2):
    t = threading.Thread(target=cpu_bound_task)
    t.start()
    threads.append(t)
for t in threads:
    t.join()
print(f"Multi thread: {time.time() - start:.2f}s")  # 약 1.4초 (더 느림!)

# ✅ I/O-bound 작업은 GIL 영향 없음
import requests

def io_bound_task(url):
    response = requests.get(url)
    return len(response.content)

# 멀티 스레드가 훨씬 빠름
urls = ["https://example.com"] * 10
start = time.time()
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(io_bound_task, urls)
print(f"I/O-bound with threads: {time.time() - start:.2f}s")
```

**GIL 우회 방법:**

| 작업 종류 | 해결책 | 이유 |
|----------|--------|------|
| CPU-bound | `multiprocessing` | 별도 프로세스 = 별도 GIL |
| I/O-bound | `threading` | I/O 대기 중 GIL 해제 |
| 계산 집약적 | C 확장, Cython, NumPy | GIL 없이 실행 |
| 최신 비동기 | `asyncio` | 협력적 멀티태스킹 |

```python
# CPU-bound: multiprocessing 사용
from multiprocessing import Pool

def cpu_task(n):
    return sum(i*i for i in range(n))

if __name__ == '__main__':
    with Pool(4) as p:
        results = p.map(cpu_task, [10_000_000] * 4)
    print(results)  # 4배 빠름!
```

### 18.2 스레딩 (I/O-bound 작업용)

```python
import threading
import time

def worker(name):
    print(f"Worker {name} starting")
    time.sleep(2)  # I/O 대기 (GIL 해제됨)
    print(f"Worker {name} done")

threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

# Lock (데이터 레이스 방지)
lock = threading.Lock()
shared_data = 0

def increment():
    global shared_data
    with lock:
        shared_data += 1

# RLock (재진입 가능 락)
rlock = threading.RLock()

def recursive_function(n):
    with rlock:
        if n > 0:
            recursive_function(n - 1)

# Semaphore (동시 접근 제한)
semaphore = threading.Semaphore(3)  # 최대 3개 스레드

def limited_access():
    with semaphore:
        print(f"Accessing resource")
        time.sleep(1)

# Event (스레드 간 신호)
event = threading.Event()

def waiter():
    print("Waiting for event")
    event.wait()
    print("Event received!")

def setter():
    time.sleep(2)
    event.set()
```

### 18.2 멀티프로세싱

```python
from multiprocessing import Process, Pool, Queue

def worker(n):
    return n * n

# Process
p = Process(target=worker, args=(5,))
p.start()
p.join()

# Pool
with Pool(4) as pool:
    results = pool.map(worker, [1, 2, 3, 4, 5])
    print(results)

# Queue
def producer(q):
    for i in range(5):
        q.put(i)

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(item)

q = Queue()
```

### 18.3 asyncio

```python
import asyncio

async def say_hello():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

# 실행
asyncio.run(say_hello())

# 여러 작업 동시 실행
async def task(name, delay):
    await asyncio.sleep(delay)
    print(f"Task {name} done")

async def main():
    await asyncio.gather(
        task("A", 1),
        task("B", 2),
        task("C", 1.5)
    )

asyncio.run(main())
```

---

## 19. 타입 힌팅

```python
from typing import List, Dict, Tuple, Optional, Union, Callable

# 기본 타입 힌팅
def greet(name: str) -> str:
    return f"Hello, {name}"

# 컬렉션
def process_numbers(numbers: List[int]) -> int:
    return sum(numbers)

# 딕셔너리
def get_user_info() -> Dict[str, Union[str, int]]:
    return {"name": "Alice", "age": 25}

# Optional
def find_user(user_id: int) -> Optional[str]:
    # None을 반환할 수 있음
    return None

# Callable
def apply(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

# 제네릭
from typing import TypeVar, Generic

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self.items: List[T] = []

    def push(self, item: T) -> None:
        self.items.append(item)

    def pop(self) -> T:
        return self.items.pop()

# Protocol (구조적 서브타이핑)
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

# mypy로 타입 체크
# $ mypy program.py
```

---

## 20. 고급 기능과 패턴

### 20.1 메모리 관리 - Python 내부 동작

**참조 카운팅 (Reference Counting)**

```python
import sys

# 참조 카운트 확인
a = []
print(sys.getrefcount(a))  # 2 (a + getrefcount 매개변수)

b = a  # 참조 추가
print(sys.getrefcount(a))  # 3

del b  # 참조 제거
print(sys.getrefcount(a))  # 2

# 참조 카운트가 0이 되면 즉시 메모리 해제
```

**순환 참조 문제**

```python
# ❌ 순환 참조 - 메모리 누수 가능
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

# 순환 구조 생성
node1 = Node(1)
node2 = Node(2)
node1.next = node2
node2.next = node1  # 순환!

# node1, node2를 삭제해도 서로 참조하므로 ref count가 0이 안됨
del node1, node2  # 메모리 누수!

# ✅ 해결책 1: weakref 사용
import weakref

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None  # weakref.ref 사용

node1 = Node(1)
node2 = Node(2)
node1.next = weakref.ref(node2)
node2.next = weakref.ref(node1)

# ✅ 해결책 2: 가비지 컬렉터가 자동 처리
import gc
gc.collect()  # 순환 참조 수집
```

**가비지 컬렉터 (Garbage Collector)**

```python
import gc

# GC 상태 확인
print(gc.get_count())  # (threshold0, threshold1, threshold2)
print(gc.get_threshold())  # (700, 10, 10)

# GC 수동 실행
collected = gc.collect()
print(f"Collected {collected} objects")

# GC 비활성화/활성화 (성능 최적화)
gc.disable()
# ... 순환 참조 없는 코드 실행
gc.enable()

# 추적 가능한 객체 확인
print(len(gc.get_objects()))  # 모든 객체 수

# 순환 참조 찾기
gc.set_debug(gc.DEBUG_SAVEALL)
gc.collect()
for obj in gc.garbage:
    print(obj)
```

**__del__ 메서드의 위험성**

```python
# ❌ __del__은 피해야 함
class BadResource:
    def __del__(self):
        print("Cleaning up")
        # 문제점:
        # 1. 호출 시점 불확실
        # 2. 순환 참조 시 호출 안될 수 있음
        # 3. 예외 발생 시 무시됨

# ✅ 대신 컨텍스트 매니저 사용
class GoodResource:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Cleaning up")  # 확실히 호출됨
        return False

with GoodResource() as resource:
    # 사용
    pass
# __exit__ 자동 호출
```

**메모리 프로파일링**

```python
# memory_profiler 사용
# pip install memory-profiler

from memory_profiler import profile

@profile
def memory_heavy_function():
    huge_list = [i for i in range(1000000)]
    return sum(huge_list)

# 실행: python -m memory_profiler script.py

# tracemalloc (내장)
import tracemalloc

tracemalloc.start()

# 메모리 사용량 많은 코드
data = [i**2 for i in range(1000000)]

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.2f} MB")
print(f"Peak: {peak / 1024 / 1024:.2f} MB")

tracemalloc.stop()

# 스냅샷 비교
snapshot1 = tracemalloc.take_snapshot()
# ... 코드 실행
snapshot2 = tracemalloc.take_snapshot()

top_stats = snapshot2.compare_to(snapshot1, 'lineno')
for stat in top_stats[:10]:
    print(stat)
```

**성능 최적화 팁**

```python
# 1. 리스트 컴프리헨션 vs map/filter
# 리스트 컴프리헨션이 약간 빠름
import timeit

# 리스트 컴프리헨션
def list_comp():
    return [x*2 for x in range(1000)]

# map
def map_func():
    return list(map(lambda x: x*2, range(1000)))

print(timeit.timeit(list_comp, number=10000))  # 더 빠름
print(timeit.timeit(map_func, number=10000))

# 2. 제너레이터로 메모리 절약
def huge_range(n):
    for i in range(n):
        yield i * i

# 메모리 사용량 비교
import sys
list_version = [i*i for i in range(10000)]
gen_version = (i*i for i in range(10000))

print(sys.getsizeof(list_version))  # 약 87KB
print(sys.getsizeof(gen_version))   # 약 112 bytes!

# 3. __slots__ 메모리 절약
class WithoutSlots:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class WithSlots:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y

import sys
obj1 = WithoutSlots(1, 2)
obj2 = WithSlots(1, 2)

print(sys.getsizeof(obj1) + sys.getsizeof(obj1.__dict__))  # 더 큼
print(sys.getsizeof(obj2))  # 약 40% 작음

# 4. intern 문자열 메모리 절약
import sys

a = "hello"
b = "hello"
print(a is b)  # True (자동 intern)

# 동적 문자열
c = "".join(["h", "e", "l", "l", "o"])
print(a is c)  # False

# 명시적 intern
d = sys.intern(c)
print(a is d)  # True - 메모리 절약
```

### 20.2 메타클래스

```python
class Meta(type):
    def __new__(cls, name, bases, dct):
        print(f"Creating class {name}")
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=Meta):
    pass

# Singleton 패턴
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class MyClass(metaclass=Singleton):
    pass
```

### 20.2 디스크립터

```python
class Descriptor:
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        return obj.__dict__.get(self.name, None)

    def __set__(self, obj, value):
        if value < 0:
            raise ValueError("Cannot be negative")
        obj.__dict__[self.name] = value

class Person:
    age = Descriptor("age")

    def __init__(self, age):
        self.age = age

p = Person(25)
# p.age = -5  # ValueError
```

### 20.3 슬롯

```python
class Point:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

# 메모리 절약, 속성 추가 불가
p = Point(1, 2)
# p.z = 3  # AttributeError
```

### 20.4 데이터클래스 (Python 3.7+)

```python
from dataclasses import dataclass, field

@dataclass
class Person:
    name: str
    age: int
    email: str = "unknown@example.com"
    friends: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.age < 0:
            raise ValueError("Age cannot be negative")

p = Person("Alice", 25)
print(p)  # Person(name='Alice', age=25, email='unknown@example.com', friends=[])
```

### 20.5 주요 디자인 패턴

```python
# Factory Pattern
class ShapeFactory:
    @staticmethod
    def create_shape(shape_type):
        if shape_type == "circle":
            return Circle()
        elif shape_type == "square":
            return Square()

# Observer Pattern
class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def notify(self, data):
        for observer in self._observers:
            observer.update(data)

# Strategy Pattern
class Context:
    def __init__(self, strategy):
        self._strategy = strategy

    def execute_strategy(self, data):
        return self._strategy.execute(data)
```

---

## 결론

Python은 읽기 쉽고 강력한 다목적 언어입니다:

1. **간결한 문법**: 의사코드와 유사한 가독성
2. **풍부한 라이브러리**: 배터리 포함 철학
3. **다중 패러다임**: 객체지향, 함수형, 절차적
4. **동적 타이핑**: 빠른 개발, 타입 힌팅으로 보완
5. **커뮤니티**: 방대한 생태계와 지원

**학습 순서**: 1-7 → 5-6 → 9-10 → 11 → 12-15 → 18-19

Python으로 아이디어를 빠르게 구현하세요!
