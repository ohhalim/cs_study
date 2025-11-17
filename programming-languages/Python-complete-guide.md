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

**(계속됩니다...)**

이 Python 가이드는 계속해서 다음 주제들을 다룹니다:
- 10. 예외 처리
- 11. 객체지향 프로그래밍
- 12. 함수형 프로그래밍
- 13. 반복자와 제너레이터
- 14. 데코레이터
- 15. 컨텍스트 매니저
- 16. 정규표현식
- 17. 표준 라이브러리
- 18. 동시성과 병렬성
- 19. 타입 힌팅
- 20. 고급 기능과 패턴

각 섹션은 Python의 강력한 기능들을 실전 예제와 함께 심도 있게 다룹니다.
